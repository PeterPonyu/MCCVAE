"""
Unified base model interface for single-cell gene expression models
Provides consistent training, inference, and latent extraction
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any, Iterator
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import inspect


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for single-cell models
    
    Defines unified interface for:
    - Training with early stopping
    - Validation
    - Latent representation extraction
    - Model checkpointing
    
    Subclasses must implement: encode(), decode(), forward(), compute_loss()
    
    Example:
        >>> class MyVAE(BaseModel):
        ...     def encode(self, x): return self.encoder(x)
        ...     def decode(self, z): return self.decoder(z)
        ...     def forward(self, x, **kwargs):
        ...         z = self.encode(x)
        ...         return {'latent': z, 'reconstruction': self.decode(z)}
        ...     def compute_loss(self, x, outputs, **kwargs):
        ...         recon_loss = F.mse_loss(outputs['reconstruction'], x)
        ...         return {'total_loss': recon_loss, 'recon_loss': recon_loss}
    """
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 model_name: str = "base_model"):
        """
        Args:
            input_dim: Input feature dimension (number of genes)
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions (default: [512, 256])
            model_name: Model identifier
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.model_name = model_name
        
    @abstractmethod
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode x -> z"""
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode z -> reconstruction"""
        raise NotImplementedError

    def _filter_kwargs_for(self, fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs to match callable signature (if no **kwargs, only pass accepted args)"""
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return {}

        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input [batch_size, input_dim]
            **kwargs: Additional args (batch_id, labels, etc.)
            
        Returns:
            Dict with 'latent' and model-specific outputs (e.g., 'reconstruction')
        """
        pass
    
    @abstractmethod
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], 
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            x: Input [batch_size, input_dim]
            outputs: forward() output dict
            **kwargs: Additional args
            
        Returns:
            Dict with 'total_loss' (required) and other losses (e.g., 'recon_loss', 'kl_loss')
        """
        pass
    
    def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            lr: float = 1e-3,
            device: str = 'cuda',
            save_path: Optional[str] = None,
            patience: int = 25,
            verbose: int = 1,
            verbose_every: int = 1,
            **kwargs) -> Dict[str, list]:
        """
        Train with optional validation and early stopping
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (enables early stopping)
            epochs: Maximum epochs
            lr: Learning rate
            device: 'cuda' or 'cpu'
            save_path: Path to save best checkpoint (by val loss)
            patience: Early stopping patience
            verbose: 0=quiet, 1=epoch logs, 2=epoch+batch logs
            verbose_every: Print frequency for epoch logs
            **kwargs: Forwarded to forward()/compute_loss() and optimizer weight_decay
            
        Returns:
            history: Dict with train/val loss curves
        """
        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader, optimizer, device, verbose, **kwargs)
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_recon_loss'].append(train_metrics['recon_loss'])

            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )

            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, device, verbose, **kwargs)
                history['val_loss'].append(val_metrics['total_loss'])
                history['val_recon_loss'].append(val_metrics['recon_loss'])

                scheduler.step(val_metrics['total_loss'])

                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    patience_counter = 0
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1

                if do_print:
                    print(
                        f"Epoch {epoch+1:3d}/{epochs} | "
                        f"Train Loss: {train_metrics['total_loss']:8.4f} | "
                        f"Val Loss: {val_metrics['total_loss']:8.4f}"
                    )

                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"\n✓ Early stopping at epoch {epoch+1}")
                    break
            else:
                if do_print:
                    print(
                        f"Epoch {epoch+1:3d}/{epochs} | "
                        f"Train Loss: {train_metrics['total_loss']:8.4f}"
                    )

        if verbose >= 1:
            print("\n✓ Training finished!")
        return history
    
    def _prepare_batch(self, batch_data: Any, device: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Prepare batch for training/inference
        Supports Env DataLoader format: (x_norm, x_raw)
        - Returns x = x_norm
        - Passes x_raw via batch_kwargs["x_raw"] if shapes match
        """
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            batch_kwargs: Dict[str, Any] = {}

            if len(batch_data) >= 2 and torch.is_tensor(batch_data[1]):
                b1 = batch_data[1]
                if torch.is_floating_point(b1) and b1.shape == x.shape:
                    batch_kwargs["x_raw"] = b1.to(device).float()

            return x, batch_kwargs

        x = batch_data.to(device).float()
        return x, {}
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                     device: str, verbose: int = 1, **kwargs) -> Dict[str, float]:
        """Train single epoch"""
        self.train()
        total_loss = 0
        total_recon_loss = 0
        n_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            x, batch_kwargs = self._prepare_batch(batch_data, device)
            
            optimizer.zero_grad()
            outputs = self.forward(x, **batch_kwargs, **kwargs)
            losses = self.compute_loss(x, outputs, **batch_kwargs, **kwargs)
            
            loss = losses['total_loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += losses.get('recon_loss', loss).item()
            n_batches += 1
            
            if verbose >= 2:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        return {
            'total_loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader, device: str, 
                       verbose: int = 1, **kwargs) -> Dict[str, float]:
        """Validate single epoch"""
        self.eval()
        total_loss = 0
        total_recon_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                
                outputs = self.forward(x, **batch_kwargs, **kwargs)
                losses = self.compute_loss(x, outputs, **batch_kwargs, **kwargs)
                
                total_loss += losses['total_loss'].item()
                total_recon_loss += losses.get('recon_loss', losses['total_loss']).item()
                n_batches += 1
        
        return {
            'total_loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches
        }
    
    def _iter_loader(self, data_loader: DataLoader, device: str) -> Iterator[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Iterate over DataLoader yielding (x, batch_kwargs) on device"""
        for batch_data in data_loader:
            x, batch_kwargs = self._prepare_batch(batch_data, device)
            yield x, batch_kwargs

    def extract_latent(self, data_loader, device='cuda', return_reconstructions: bool = False):
        """
        Extract latent representations
        
        Args:
            data_loader: DataLoader
            device: Computation device
            return_reconstructions: If True, also return reconstructions via decode()
            
        Returns:
            Dict with 'latent' (and optionally 'reconstruction', 'labels')
        """
        self.eval()
        self.to(device)

        latents = []
        reconstructions = [] if return_reconstructions else None
        labels = []

        with torch.no_grad():
            for x, batch_kwargs in self._iter_loader(data_loader, device):
                enc_kwargs = self._filter_kwargs_for(self.encode, batch_kwargs)
                z = self.encode(x, **enc_kwargs)
                latents.append(z.detach().cpu().numpy())

                if "y" in batch_kwargs and batch_kwargs["y"] is not None:
                    labels.append(batch_kwargs["y"].detach().cpu().numpy())

                if return_reconstructions:
                    try:
                        dec_kwargs = self._filter_kwargs_for(self.decode, batch_kwargs)
                        recon = self.decode(z, **dec_kwargs)
                    except NotImplementedError:
                        out_kwargs = self._filter_kwargs_for(self.forward, batch_kwargs)
                        out = self.forward(x, **out_kwargs)
                        if isinstance(out, dict) and "reconstruction" in out:
                            recon = out["reconstruction"]
                        else:
                            raise NotImplementedError(
                                f"{self.__class__.__name__} does not implement decode(), "
                                f"and forward() did not return outputs['reconstruction']. "
                                f"Disable return_reconstructions or implement decode()/reconstruction."
                            )
                    reconstructions.append(recon.detach().cpu().numpy())

        result = {"latent": np.concatenate(latents, axis=0)}
        if len(labels) > 0:
            result["labels"] = np.concatenate(labels, axis=0)
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(reconstructions, axis=0)
        return result
    
    def save_model(self, path: str):
        """Save model weights and config"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'hidden_dims': self.hidden_dims,
                'model_name': self.model_name
            }
        }, path)
    
    def load_model(self, path: str) -> Dict[str, Any]:
        """Load model weights and config"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('config', {})