"""
Cell BLAST的PyTorch实现
基于DIRECTi的单细胞注释模型

Reference: Cao et al. (2020) Searching large-scale scRNA-seq databases via 
unbiased cell embedding with Cell BLAST. Nature Communications.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from .base_model import BaseModel


class Encoder(nn.Module):
    """DIRECTi编码器，使用ELU激活函数"""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, 
                 dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer_modules = [nn.Linear(prev_dim, hidden_dim)]
            if use_bn:
                layer_modules.append(nn.BatchNorm1d(hidden_dim))
            layer_modules.append(nn.ELU())
            if dropout > 0:
                layer_modules.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_modules))
            prev_dim = hidden_dim
        
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x):
        """返回 (mu, logvar)"""
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """DIRECTi解码器，输出ZINB/NB参数"""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float = 0.0, use_bn: bool = True,
                 output_distribution: str = 'zinb'):
        super().__init__()
        
        self.output_distribution = output_distribution
        
        self.layers = nn.ModuleList()
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layer_modules = [nn.Linear(prev_dim, hidden_dim)]
            if use_bn:
                layer_modules.append(nn.BatchNorm1d(hidden_dim))
            layer_modules.append(nn.ELU())
            if dropout > 0:
                layer_modules.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_modules))
            prev_dim = hidden_dim
        
        if output_distribution == 'zinb':
            self.mean_decoder = nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softmax(dim=-1))
            self.disp_decoder = nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softplus())
            self.dropout_decoder = nn.Linear(prev_dim, output_dim)
        elif output_distribution == 'nb':
            self.mean_decoder = nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softmax(dim=-1))
            self.disp_decoder = nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softplus())
        else:
            self.mean_decoder = nn.Linear(prev_dim, output_dim)
    
    def forward(self, z):
        """返回分布参数字典"""
        h = z
        for layer in self.layers:
            h = layer(h)
        
        if self.output_distribution == 'zinb':
            return {
                'mean': self.mean_decoder(h),
                'disp': self.disp_decoder(h),
                'dropout_logit': self.dropout_decoder(h)
            }
        elif self.output_distribution == 'nb':
            return {
                'mean': self.mean_decoder(h),
                'disp': self.disp_decoder(h)
            }
        return {'mean': self.mean_decoder(h)}


class BatchDiscriminator(nn.Module):
    """对抗性批次校正的判别器"""
    def __init__(self, latent_dim: int, n_batches: int, hidden_dim: int = 128):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_batches)
        )
    
    def forward(self, z):
        return self.discriminator(z)


class CellBLASTModel(BaseModel):
    """
    Cell BLAST完整实现：ZINB/NB重构 + 对抗性批次校正 + 概率潜在空间
    
    Features:
    - ZINB/NB reconstruction for count data
    - Adversarial batch correction
    - Probabilistic latent space (VAE)
    - ELU activation
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 dropout: float = 0.0,
                 use_bn: bool = True,
                 output_distribution: str = 'zinb',
                 use_batch_correction: bool = False,
                 n_batches: int = 1,
                 adversarial_weight: float = 1.0,
                 model_name: str = "CellBLAST"):
        """
        Args:
            input_dim: 基因数
            latent_dim: 潜在空间维度
            hidden_dims: 隐藏层维度，默认 [512, 256, 128]
            dropout: Dropout率
            use_bn: 是否使用BatchNorm
            output_distribution: 'zinb', 'nb', 或 'gaussian'
            use_batch_correction: 是否使用对抗性批次校正
            n_batches: 批次数量
            adversarial_weight: 对抗损失权重
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.dropout = dropout
        self.use_bn = use_bn
        self.output_distribution = output_distribution
        self.use_batch_correction = use_batch_correction
        self.n_batches = n_batches
        self.adversarial_weight = adversarial_weight
        
        self.encoder_net = Encoder(input_dim, hidden_dims, latent_dim, dropout, use_bn)
        self.decoder_net = Decoder(latent_dim, hidden_dims, input_dim, dropout, use_bn, output_distribution)
        
        if use_batch_correction and n_batches > 1:
            self.batch_discriminator = BatchDiscriminator(latent_dim, n_batches)
        else:
            self.batch_discriminator = None
    
    def _prepare_batch(self, batch_data, device):
        """处理批次数据，支持 (x_norm, x_raw) 和可选的 batch_id"""
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            metadata = {}

            if len(batch_data) >= 2 and torch.is_tensor(batch_data[1]):
                second_item = batch_data[1]

                if torch.is_floating_point(second_item) and second_item.shape == x.shape:
                    metadata["x_raw"] = second_item.to(device).float()
                elif second_item.dtype in (torch.int32, torch.int64) and second_item.ndim == 1:
                    if self.use_batch_correction and second_item.max() < self.n_batches:
                        metadata["batch_id"] = second_item.to(device).long()

            if self.use_batch_correction and "batch_id" not in metadata:
                metadata["batch_id"] = torch.zeros(x.size(0), dtype=torch.long, device=device)

            return x, metadata

        x = batch_data.to(device).float()
        metadata = {}
        if self.use_batch_correction:
            metadata["batch_id"] = torch.zeros(x.size(0), dtype=torch.long, device=device)
        return x, metadata
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间"""
        mu, logvar = self.encoder_net(x)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z: torch.Tensor):
        """从潜在空间解码"""
        decoder_output = self.decoder_net(z)
        return decoder_output['mean']
    
    def forward(self, x: torch.Tensor, batch_id: Optional[torch.Tensor] = None, 
                x_raw: Optional[torch.Tensor] = None, **kwargs):
        """
        前向传播
        
        Args:
            x: x_norm (用于编码器)
            x_raw: 原始counts (用于library size和似然计算)
            batch_id: 批次标签
        """
        x_counts = x_raw if x_raw is not None else x
        library_size = x_counts.sum(dim=1, keepdim=True)

        mu, logvar = self.encoder_net(x)
        z = self.reparameterize(mu, logvar)
        decoder_output = self.decoder_net(z)

        batch_logits = None
        if self.batch_discriminator is not None and batch_id is not None:
            batch_logits = self.batch_discriminator(z)

        output = {
            "latent": z,
            "mu": mu,
            "logvar": logvar,
            "library_size": library_size,
            "batch_logits": batch_logits,
        }
        output.update(decoder_output)
        return output
    
    def _zinb_loss(self, x: torch.Tensor, mean: torch.Tensor,
                   disp: torch.Tensor, dropout_logit: torch.Tensor,
                   library_size: torch.Tensor) -> torch.Tensor:
        """Zero-Inflated Negative Binomial loss"""
        eps = 1e-10
        mean_scaled = mean * library_size
        pi = torch.sigmoid(dropout_logit)
        
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean_scaled / (disp + eps))) + \
             (x * (torch.log(disp + eps) - torch.log(mean_scaled + eps)))
        nb_log_likelihood = -(t1 + t2)
        
        zero_nb = torch.pow(disp / (disp + mean_scaled + eps), disp)
        zero_case_log_prob = torch.log(pi + (1.0 - pi) * zero_nb + eps)
        non_zero_case_log_prob = torch.log(1.0 - pi + eps) + nb_log_likelihood
        
        loss = torch.where(x < 1e-8, -zero_case_log_prob, -non_zero_case_log_prob)
        return loss.mean()
    
    def _nb_loss(self, x: torch.Tensor, mean: torch.Tensor,
                 disp: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """Negative Binomial loss"""
        eps = 1e-10
        mean_scaled = mean * library_size
        
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean_scaled / (disp + eps))) + \
             (x * (torch.log(disp + eps) - torch.log(mean_scaled + eps)))
        
        return (t1 + t2).mean()
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], 
                    beta: float = 1.0, batch_id: Optional[torch.Tensor] = None, 
                    x_raw: Optional[torch.Tensor] = None, **kwargs):
        """计算总损失：重构 + KL + 对抗性批次校正"""
        x_counts = x_raw if x_raw is not None else x
        mu, logvar = outputs["mu"], outputs["logvar"]
        library_size = outputs["library_size"]

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        kl_loss = torch.clamp(kl_loss, min=0.0)

        if self.output_distribution == "zinb":
            recon_loss = self._zinb_loss(x_counts, outputs["mean"], outputs["disp"], 
                                        outputs["dropout_logit"], library_size)
        elif self.output_distribution == "nb":
            recon_loss = self._nb_loss(x_counts, outputs["mean"], outputs["disp"], library_size)
        else:
            recon_loss = F.mse_loss(outputs["mean"], x, reduction="mean")

        batch_disc_loss = torch.tensor(0.0, device=x.device)
        batch_adv_loss = torch.tensor(0.0, device=x.device)
        
        if self.batch_discriminator is not None and batch_id is not None and outputs["batch_logits"] is not None:
            batch_disc_loss = F.cross_entropy(outputs["batch_logits"], batch_id)
            batch_adv_loss = -batch_disc_loss

        total_loss = recon_loss + beta * kl_loss + self.adversarial_weight * batch_adv_loss
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "batch_disc_loss": batch_disc_loss,
            "batch_adv_loss": batch_adv_loss,
        }
    
    def compute_posterior_distance(self, z1: torch.Tensor, z2: torch.Tensor,
                                  mu1: torch.Tensor, mu2: torch.Tensor,
                                  logvar1: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
        """
        计算细胞间后验距离（用于Cell BLAST注释）
        使用KL散度: KL(q(z|x1) || q(z|x2))
        
        Returns:
            [batch1, batch2] 成对KL散度矩阵
        """
        mu1 = mu1.unsqueeze(1)
        mu2 = mu2.unsqueeze(0)
        logvar1 = logvar1.unsqueeze(1)
        logvar2 = logvar2.unsqueeze(0)
        
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        kl = 0.5 * (
            logvar2 - logvar1 +
            (var1 + (mu1 - mu2).pow(2)) / (var2 + 1e-10) - 1
        )
        
        return kl.sum(dim=-1)


def create_cellblast_model(input_dim: int, latent_dim: int = 10, **kwargs):
    """
    创建Cell BLAST模型
    
    Examples:
        >>> model = create_cellblast_model(2000, latent_dim=10, output_distribution='zinb')
        >>> model = create_cellblast_model(2000, use_batch_correction=True, n_batches=5)
    """
    return CellBLASTModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)