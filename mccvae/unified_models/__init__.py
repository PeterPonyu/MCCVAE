"""
统一模型接口 - 导入所有模型

约定：
- 每个模型提供统一接口（BaseModel 子类）
- 每个模型提供 create_* 工厂函数
"""
from .base_model import BaseModel

from .cellblast_model import CellBLASTModel, create_cellblast_model
from .gmvae_model import GMVAEModel, create_gmvae_model
from .scalex_model import SCALEXModel, create_scalex_model
from .scdiffusion_model import scDiffusionModel, create_scdiffusion_model
from .sivae_model import siVAEModel, create_sivae_model

from .clear_model import CLEARModel, create_clear_model
from .scdac_model import scDACModel, create_scdac_model
from .scdeepcluster_model import scDeepClusterModel, create_scdeepcluster_model
from .scdhmap_model import scDHMapModel, create_scdhmap_model
from .scgnn_model import scGNNModel, create_scgnn_model
from .scgcc_model import scGCCModel, create_scgcc_model
from .scsmd_model import scSMDModel, create_scsmd_model

# New graph contrastive learning models
from .scagcl_model import scAGCLModel, create_scagcl_model
from .scgpcl_model import scGPCLModel, create_scgpcl_model
from .schsc_model import scHSCModel, create_schsc_model
from .scsimgcl_model import scSimGCLModel, create_scsimgcl_model

__all__ = [
    # base
    "BaseModel",
    # models
    "CellBLASTModel",
    "GMVAEModel",
    "SCALEXModel",
    "scDiffusionModel",
    "siVAEModel",
    "CLEARModel",
    "scDACModel",
    "scDeepClusterModel",
    "scDHMapModel",
    "scGNNModel",
    "scGCCModel",
    "scSMDModel",
    "scAGCLModel",
    "scGPCLModel",
    "scHSCModel",
    "scSimGCLModel",
    # factories
    "create_cellblast_model",
    "create_gmvae_model",
    "create_scalex_model",
    "create_scdiffusion_model",
    "create_sivae_model",
    "create_clear_model",
    "create_scdac_model",
    "create_scdeepcluster_model",
    "create_scdhmap_model",
    "create_scgnn_model",
    "create_scgcc_model",
    "create_scsmd_model",
    "create_scagcl_model",
    "create_scgpcl_model",
    "create_schsc_model",
    "create_scsimgcl_model",
]