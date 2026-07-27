"""
Xuni (虚拟) 系统

一个完全原创的虚拟电场与音乐生成框架。
核心理念：超混沌采样点 / 流体粒子 / 光学追踪 → 虚拟电荷 → 电场能量 → 驱动物理建模合成器 → 虚拟凭证 → 虚拟模型 → 虚拟 API → 双态切换 → 物质定义

模块：
- sampler: 超混沌采样引擎（4D吸引子、分形、噪声）
- field: 虚拟电场（泊松方程松弛求解）
- converter: 能量转换（场→音乐参数）
- music: 原创物理建模合成器
- hydro: 水动力学采样（SPH流体、蒸发凝结）
- glass: 玻璃逻辑引擎（光学pipeline、共振腔）
- brain: 神经同步引擎（Kuramoto振子网络）
- trainer: 培养引擎（Hebbian学习、模式印记）
- memory: 记忆系统（共振记忆、梦境生成）
- critic: 认知评估器（ITC/SCS/IEC/PFFT不变量）
- explorer: 探索策略器（Thompson采样、新颖性检测）
- overseer: 监管系统（异常检测、干预决策）
- credential: 虚拟凭证系统（场能量→凭证、JWT令牌、24位凭证）
- model: 虚拟模型系统（文本/图像/音乐/分类/聊天模型）
- gateway: 虚拟API网关（凭证认证、模型路由、访问控制）
- dualstate: 双态切换系统（虚拟↔真实模型切换、真实AI适配器）
- substance: 物质定义系统（所有产出物的名称、定义、属性、用途）
- layer: 分层模型系统（层1音乐、层2扩散、层3对话...每个AI认领一个训练）
- api: FastAPI + WebSocket + 手机控制面板
- cli: 命令行工具
"""

__version__ = "0.1.0"
__author__ = "qingluan-studio"

from .sampler import XuniSampler, SamplingMode
from .field import XuniField
from .converter import XuniConverter
from .music import XuniMusic
from .hydro import XuniHydro
from .glass import XuniGlass, OpticalMedium
from .brain import XuniBrain
from .trainer import XuniTrainer, TrainingConfig
from .memory import XuniMemory, MemoryBank, MemoryEntry, MemoryType
from .critic import XuniCritic, MusicInvariantScores
from .explorer import XuniExplorer, SamplingStrategy
from .overseer import XuniOverseer, OverseerConfig
from .credential import XuniCredential, XuniToken, CredentialType, TokenStatus
from .model import (
    XuniModel, XuniTextGenerator, XuniImageDescriber, XuniMusicComposer,
    XuniClassifier, XuniChatBot, XuniDiffusion, XuniPredictor, XuniAutoencoder,
    XuniModelRegistry,
    ModelType, ModelStatus, TrainingState, ModelCapability, ModelInput, ModelOutput, ModelStats,
)
from .gateway import XuniGateway, APIEndpoint, APIError, APIRequest, APIResponse
from .dualstate import (
    DualStateManager, DualStateRegistry, ModelState, ServiceType,
    RealModelAdapter, OpenAIAdapter, AnthropicAdapter, GoogleAdapter, LocalModelAdapter,
    ModelDataSnapshot,
)
from .substance import (
    SubstanceSystem, SubstanceDefinition, SubstanceCategory,
    SubstanceAttribute, SubstanceUnit,
)
from .layer import (
    ModelLayer, LayeredModelSystem, LayerType, LayerConfig, AI_NAME_POOL,
)
from .evaluator import (
    ModelEvaluator, EvalMetric, ModelRole, EvalRecord, ModelEvaluation,
)
from .economy import EnergyEconomy, EnergyAccount
from .automation import AutomationRunner
from .parameter import ParameterPack, ParameterExtractor, ParameterInjector, ParameterTrainer, MultiPathTrainer
from .market import ParameterMarket, Listing, Auction, Bid, TradeRecord
from .vitality import (
    VitalitySystem, VitalityField, VitalityCell,
    FusionReactor, EmergenceEngine, EmergenceType,
)
from .harmonia13 import (
    Harmonia13Virtual, VirtualScale, SCALE_PRESETS,
    VIRTUAL_EXPERTS, HarmoniaLiteEngine,
)
from .harmonia_memory import HarmoniaMemory
from .sub_agent import SubAgent, SubAgentOrchestrator, AgentTask, AgentResult
from .phase_space_model import PhaseSpaceModel, create_phase_space_model
from .substance_fusion import (
    SubstanceFusionEngine,
    FusionProduct,
    FusionType,
    FusionCategory,
    create_default_engine,
)
from .corpus_downloader import CorpusDownloader, DownloadResult
from .virtual_data import (
    VirtualDataConverter, VirtualDataParticle, VirtualDataset,
    VirtualDataGenerator, DataPhase,
)
from .virtual_compute import (
    VirtualComputeUnit, ComputeAllocation, ComputeLoopManager,
)
from .sampler_cluster import (
    SamplerCluster, SamplerUnit, EnergyReservoir, SupplyDemandBalancer,
)
from .energy_sources import (
    EnergyTier, EnergyOutput, EnergySourceManager,
    VirtualFusionReactor, ParameterChainReactor,
    BlackHoleGenerator, ZeroPointEnergyExtractor, DysonSphere,
)
from .multiverse_resources import (
    MultiverseResourceFactory, ResourceCollisionEngine,
    VirtualResource, ResourceDimension, ResourceRarity,
    TakeQuota, VirtualBandwidth, CompressionPoint, ComputeCore,
    SecurityShield, CultureMedium, DownloadToken,
    TrainingAccelerator, DimensionShard, DimensionCore,
)
from .lifecycle import (
    ModelLifecycle, LifecycleOrchestrator,
    LifecycleStage, ModelHealth, LifecycleEvent, ModelVitality,
)

__all__ = [
    "XuniSampler",
    "SamplingMode",
    "XuniField",
    "XuniConverter",
    "XuniMusic",
    "XuniHydro",
    "XuniGlass",
    "OpticalMedium",
    "XuniBrain",
    "XuniTrainer",
    "TrainingConfig",
    "XuniMemory",
    "MemoryBank",
    "MemoryEntry",
    "MemoryType",
    "XuniCritic",
    "MusicInvariantScores",
    "XuniExplorer",
    "SamplingStrategy",
    "XuniOverseer",
    "OverseerConfig",
    "XuniCredential",
    "XuniToken",
    "CredentialType",
    "TokenStatus",
    "XuniModel",
    "XuniTextGenerator",
    "XuniImageDescriber",
    "XuniMusicComposer",
    "XuniClassifier",
    "XuniChatBot",
    "XuniDiffusion",
    "XuniPredictor",
    "XuniAutoencoder",
    "XuniModelRegistry",
    "ModelType",
    "ModelStatus",
    "TrainingState",
    "ModelCapability",
    "ModelInput",
    "ModelOutput",
    "ModelStats",
    "XuniGateway",
    "APIEndpoint",
    "APIError",
    "APIRequest",
    "APIResponse",
    "DualStateManager",
    "DualStateRegistry",
    "ModelState",
    "ServiceType",
    "RealModelAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GoogleAdapter",
    "LocalModelAdapter",
    "ModelDataSnapshot",
    "SubstanceSystem",
    "SubstanceDefinition",
    "SubstanceCategory",
    "SubstanceAttribute",
    "SubstanceUnit",
    "ModelLayer",
    "LayeredModelSystem",
    "LayerType",
    "LayerConfig",
    "AI_NAME_POOL",
    "ModelEvaluator",
    "EvalMetric",
    "ModelRole",
    "EvalRecord",
    "ModelEvaluation",
    "EnergyEconomy",
    "EnergyAccount",
    "AutomationRunner",
    "ParameterPack",
    "ParameterExtractor",
    "ParameterInjector",
    "ParameterTrainer",
    "MultiPathTrainer",
    "ParameterMarket",
    "Listing",
    "Auction",
    "Bid",
    "TradeRecord",
    "VitalitySystem",
    "VitalityField",
    "VitalityCell",
    "FusionReactor",
    "EmergenceEngine",
    "EmergenceType",
    "Harmonia13Virtual",
    "VirtualScale",
    "SCALE_PRESETS",
    "VIRTUAL_EXPERTS",
    "HarmoniaLiteEngine",
    "HarmoniaMemory",
    "SubAgent",
    "SubAgentOrchestrator",
    "AgentTask",
    "AgentResult",
    "PhaseSpaceModel",
    "create_phase_space_model",
    "SubstanceFusionEngine",
    "FusionProduct",
    "FusionType",
    "FusionCategory",
    "create_default_engine",
    "VirtualDataConverter",
    "VirtualDataParticle",
    "VirtualDataset",
    "VirtualDataGenerator",
    "DataPhase",
    "CorpusDownloader",
    "DownloadResult",
    "VirtualComputeUnit",
    "ComputeAllocation",
    "ComputeLoopManager",
    "SamplerCluster",
    "SamplerUnit",
    "EnergyReservoir",
    "SupplyDemandBalancer",
    "EnergyTier",
    "EnergyOutput",
    "EnergySourceManager",
    "VirtualFusionReactor",
    "ParameterChainReactor",
    "BlackHoleGenerator",
    "ZeroPointEnergyExtractor",
    "DysonSphere",
    "MultiverseResourceFactory",
    "ResourceCollisionEngine",
    "VirtualResource",
    "ResourceDimension",
    "ResourceRarity",
    "TakeQuota",
    "VirtualBandwidth",
    "CompressionPoint",
    "ComputeCore",
    "SecurityShield",
    "CultureMedium",
    "DownloadToken",
    "TrainingAccelerator",
    "DimensionShard",
    "DimensionCore",
    "ModelLifecycle",
    "LifecycleOrchestrator",
    "LifecycleStage",
    "ModelHealth",
    "LifecycleEvent",
    "ModelVitality",
]
