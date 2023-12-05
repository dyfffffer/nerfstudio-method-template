"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

#####################################################
from esnerf.esnerf_datamanager import (
    ESNerfDataManagerConfig,
)
from esnerf.esnerf_model import ESNerfModelConfig
from esnerf.esnerf_pipeline import (
    ESNerfPipelineConfig,
)
from esnerf.esnerf_dataparser import ESNerfDataParserConfig
from esnerf.esnerf_datamanager import ESNerfDataManager
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datasets.semantic_dataset import SemanticDataset
#####################################################





ESNerf = MethodSpecification(
    config=TrainerConfig(
        method_name="ESNerf",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=1 * 5000,
        mixed_precision=True,
        pipeline=ESNerfPipelineConfig(
            datamanager=ESNerfDataManagerConfig(
                _target=ESNerfDataManager[SemanticDataset],
                dataparser=ESNerfDataParserConfig(),
                train_num_rays_per_batch=1,
                eval_num_rays_per_batch=4096,
                #camera_optimizer=CameraOptimizerConfig(
                #    mode="SO3xR3",
                #    #optimizer=RAdamOptimizerConfig(lr=6e-3, eps=1e-8, weight_decay=1e-3),
                #    optimizer=RAdamOptimizerConfig(lr=1e-3, eps=1e-8, weight_decay=1e-5),
                #    #1optimizer=AdamOptimizerConfig(lr=1e-2, eps=1e-8, weight_decay=1e-5),
                #    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                #),
            ),
            model=ESNerfModelConfig(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            # TODO: consider changing the optimizers depending on your custom Model
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2.2e-3, eps=1e-15),
                #"optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                #"optimizer": AdamOptimizerConfig(lr=3e-3, eps=1e-15),
                #"optimizer": AdamOptimizerConfig(lr=6e-3, eps=1e-15),
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(), #lr_final=1e-5, max_steps=200000),
            },
            "pos_thre": {
                "optimizer": AdamOptimizerConfig(lr=3e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "neg_thre": {
                "optimizer": AdamOptimizerConfig(lr=3e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12,
                            websocket_port_default=7000),
        vis="viewer",
    ),
    description="Event Nerf method.",
)
