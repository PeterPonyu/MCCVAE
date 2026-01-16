
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Literal
from datetime import datetime

# Basic info models
class ShapeInfo(BaseModel):
    """Shape information for matrices"""
    n_obs: int = Field(description="Number of observations (cells)")
    n_vars: int = Field(description="Number of variables (genes)")

class ColumnInfo(BaseModel):
    """Information about DataFrame columns"""
    name: str
    dtype: str
    n_unique: int
    missing_count: int
    sample_values: Optional[List[Union[str, float, int]]] = Field(
        None, description="First few values for preview"
    )
    is_truncated: bool = Field(False, description="Whether display is truncated")

class LayerInfo(BaseModel):
    """Information about AnnData layers"""
    name: str
    dtype: str
    shape: ShapeInfo
    sparsity_ratio: Optional[float] = Field(None, description="Proportion of zero values")
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None

class QCMetrics(BaseModel):
    """Scanpy-style quality control metrics"""
    n_genes_by_counts: Dict[str, float] = Field(description="Genes detected per cell stats")
    total_counts: Dict[str, float] = Field(description="Total UMI counts per cell stats")  
    pct_counts_mt: Dict[str, float] = Field(description="Mitochondrial gene percentage stats")
    pct_counts_ribo: Dict[str, float] = Field(description="Ribosomal gene percentage stats")
    
    # Each dict contains: {"mean": X, "std": X, "min": X, "max": X, "median": X}

class AnnDataSummary(BaseModel):
    """Complete AnnData summary - Python __str__ style"""
    # Basic structure
    shape: ShapeInfo
    
    # Main data layer
    X_info: LayerInfo
    
    # Additional layers
    layers: Dict[str, LayerInfo] = Field(default_factory=dict)
    
    # Observation annotations (.obs)
    obs_columns: List[ColumnInfo] = Field(default_factory=list)
    
    # Variable annotations (.var) 
    var_columns: List[ColumnInfo] = Field(default_factory=list)
    
    # Quality control metrics
    qc_metrics: Optional[QCMetrics] = None
    
    # File info
    filename: Optional[str] = None
    file_size_mb: Optional[float] = None
    loaded_at: datetime = Field(default_factory=datetime.now)

class QCParams(BaseModel):
    """User-defined QC filtering params"""
    species: str = Field(default='human', description="Species for QC metrics (human or mouse)")
    min_genes_per_cell: Optional[int] = Field(None, ge=0)
    max_genes_per_cell: Optional[int] = Field(None, ge=0)  
    min_counts_per_cell: Optional[int] = Field(None, ge=0)
    max_counts_per_cell: Optional[int] = Field(None, ge=0)
    max_pct_mt: Optional[float] = Field(None, ge=0.0, le=100.0)
    max_pct_ribo: Optional[float] = Field(None, ge=0.0, le=100.0)

class AgentParameters(BaseModel):
    """Parameters for the MCCVAE agent"""
    
    # Data parameters
    layer: str = Field(default='counts', description="AnnData layer for training")
    percent: float = Field(default=0.01, ge=0.001, le=1.0, description="Batch sampling fraction")
    
    # Loss weights
    recon: float = Field(default=1.0, ge=0.0, description="Reconstruction loss weight")
    irecon: float = Field(default=0.0, ge=0.0, description="Information bottleneck reconstruction weight")
    beta: float = Field(default=1.0, ge=0.0, description="KL divergence weight (β-VAE)")
    dip: float = Field(default=0.0, ge=0.0, description="DIP regularization weight")
    tc: float = Field(default=0.0, ge=0.0, description="Total correlation weight")
    info: float = Field(default=0.0, ge=0.0, description="InfoVAE MMD weight")
    vae_reg: float = Field(default=0.5, ge=0.0, le=1.0, description="VAE latent regularization weight")
    
    # MoCo contrastive learning parameters
    use_moco: bool = Field(default=False, description="Enable Momentum Contrast learning")
    moco_weight: float = Field(default=1.0, ge=0.0, description="Contrastive learning loss weight")
    moco_T: float = Field(default=0.2, gt=0.0, description="Temperature parameter for MoCo contrastive loss")
    
    # Data augmentation parameters
    aug_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability of applying data augmentation")
    mask_prob: float = Field(default=0.2, ge=0.0, le=1.0, description="Probability of masking each gene during augmentation")
    noise_prob: float = Field(default=0.7, ge=0.0, le=1.0, description="Probability of adding noise during augmentation")
    
    # Model architecture parameters
    hidden_dim: int = Field(default=128, gt=0, description="Hidden layer dimension")
    latent_dim: int = Field(default=10, gt=0, description="Primary latent dimension")
    i_dim: int = Field(default=2, gt=0, description="Information bottleneck latent dimension")
    use_bn: bool = Field(default=True, description="Use batch normalization in encoder/decoder")
    
    # Training parameters
    lr: float = Field(default=1e-4, gt=0.0, description="Learning rate")
    loss_mode: Literal["mse", "nb", "zinb"] = Field(default="nb", description="Loss function mode for reconstruction")
    use_qm: bool = Field(default=True, description="Use mean (True) or samples (False) for latent representations")


class TrainingConfig(BaseModel):
    """Training configuration"""
    epochs: int = Field(default=1000, gt=0, description="Number of training epochs")
    qcparams: Optional[QCParams] = Field(None, description="QC filtering params")

class TrainingMetrics(BaseModel):
    """Training metrics at specific epoch"""
    epoch: int
    loss: float
    ari: float   # Adjusted Rand Index
    nmi: float   # Normalized Mutual Information  
    asw: float   # Average Silhouette Width
    ch: float    # Calinski-Harabasz Index
    db: float    # Davies-Bouldin Index
    pc: float    # Graph connectivity score

class TrainingState(BaseModel):
    """Current training state"""
    is_running: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    latest_metrics: Optional[TrainingMetrics] = None
    history: List[TrainingMetrics] = Field(default_factory=list)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class EmbeddingResult(BaseModel):
    """Embedding extraction result"""
    embedding_type: str  # "interpretable" or "latent"
    shape: ShapeInfo
    download_ready: bool = False
    csv_path: Optional[str] = None
    extracted_at: Optional[datetime] = None

# Response models
class UploadResponse(BaseModel):
    """File upload response"""
    success: bool
    message: str
    data_summary: Optional[AnnDataSummary] = None

class TrainingResponse(BaseModel):
    """Training start response"""
    success: bool
    message: str

