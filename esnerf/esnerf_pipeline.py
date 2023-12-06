"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.models.base_model import ModelConfig, Model
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)

#####################################################
from esnerf.esnerf_datamanager import ESNerfDataManagerConfig
from esnerf.esnerf_model import ESNerfModelConfig
#####################################################



@dataclass
class ESNerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ESNerfPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = ESNerfDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ESNerfModelConfig()
    """specifies the model config"""


class ESNerfPipeline(VanillaPipeline):
    """ESNerf Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: ESNerfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            num_eval_data=len(self.datamanager.eval_dataset),
            grad_scaler=grad_scaler,
        )
        # print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        # print(device)
        # print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])
