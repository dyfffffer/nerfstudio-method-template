from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Optional, Callable, Any, cast, List, Generic
from functools import cached_property
from pathlib import Path
import numpy as np

import torch
import random
from torch.nn import Parameter
from copy import deepcopy

from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.cameras import Cameras

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

#####################################################
from esnerf.esnerf_dataparser import (
    ESNerfDataParserConfig,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from esnerf.esnerf_sampler import ESNerfSampler
from typing_extensions import TypeVar
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_orig_class
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    ForwardRef,
    get_origin,
    get_args,
)
from nerfstudio.data.pixel_samplers import (
    PixelSampler,
    PixelSamplerConfig,
    PatchPixelSamplerConfig,
)
from nerfstudio.cameras.cameras import CameraType
#####################################################

@dataclass
class ESNerfDataManagerConfig(VanillaDataManagerConfig): 
    """ESNerf DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: ESNerfDataManager)
    dataparser: ESNerfDataParserConfig = ESNerfDataParserConfig()
    is_colored = True
    neg_ratio = 0.1
    max_winsize = 50 * 2


class ESNerfDataManager(VanillaDataManager):
    """ESNerf DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ESNerfDataManagerConfig


    def __init__(
        self,
        config: ESNerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs,
        )

    

    def setup_train(self):
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        # print(self.train_dataset)
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        # for item in self.iter_train_image_dataloader:
        #     print(item)

        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device), self.train_camera_optimizer,)
        
        
        cams = self.train_dataset.cameras
        rotations = cams.camera_to_worlds[..., :3, :3] # [N, 3, 3]
        intrinsics = cams.get_intrinsics_matrices() 
        ray_matrices = torch.Tensor()
        for i in range(len(rotations)):
            ray_matrix = rotations[i] @ intrinsics[i].inverse()
            ray_matrices = torch.concat((ray_matrices, ray_matrix[None, ...]), dim=0)
        self.ray_matrices = ray_matrices.to(self.device)
        self.ray_jitter_noise_make()

        image_batch = next(self.iter_train_image_dataloader)
        image_idx = image_batch["image_idx"]
        image = image_batch["image"]
        images = torch.zeros_like(image, device=self.device)
        for i in range(len(image_idx)):
            images[image_idx[i]] = image[i]
        self.images = images
        # CONSOLE.print(image_idx.shape, image_idx[:5], image_idx[-5:])
        # CONSOLE.print(image.shape)
        self.pos_thre = torch.ones((170, 540, 960), device=self.device)
        self.neg_thre = torch.ones((170, 540, 960), device=self.device)
        print(self.train_dataparser_outputs)
        self.event_sampler = ESNerfSampler(self.train_dataparser_outputs.metadata["event_files"]["event_files"][0], pos_thre=self.pos_thre, 
                                          neg_thre=self.neg_thre, device=self.device, batch_size=self.get_train_rays_per_batch(),
                                          neg_ratio=self.config.neg_ratio, max_winsize=self.config.max_winsize)
        self.event_iter = iter(self.event_sampler)
        # print(self.event_sampler.sample_method)
        #CONSOLE.print("cameras.size", self.train_dataset.cameras.size)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param = super().get_param_groups()
        #param["pos_thre"] = list(self.pos_thre)
        #param["neg_thre"] = list(self.neg_thre)
        return param

    # 返回下一个训练的batch
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        
        self.train_count += 1
        if self.train_count == 1:
            self.event_sampler.sample_method = "sample_3d"
        elif self.train_count == 170:    # 图像总数-1
            self.event_sampler.sample_method = "sample_1d"
            self.event_sampler.neg_ratio = 0.2
        
        # 从这个batch的images中采样pixels，这个next是sampler的函数

        ray_indices, batch = next(self.event_iter)
        batch["image"] = self.images[ray_indices[:, 0], ray_indices[:, 1], ray_indices[:, 2]]

        # 从image和像素indices产生ray
        ray_bundle = self.train_ray_generator(ray_indices)
        #if self.train_count % self.noises.shape[1] == 0:
        #    self.ray_jitter_noise_make()
        #self.ray_jitter(ray_bundle=ray_bundle)

        #image_batch = next(self.iter_train_image_dataloader)
        #batch = self.train_pixel_sampler.sample(image_batch)
        #ray_indices = batch["indices"]
        #ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch
    
    # ref arXiv: https://4dqv.mpi-inf.mpg.de/NeRF-OSR/
    def ray_jitter_noise_make(self, number : int = 5000):
        self.noises = torch.Tensor().to(self.device)
        cam_cnt = self.train_dataset.cameras.size
        assert cam_cnt != 0
        for i in range(cam_cnt):
            #noise = torch.normal(mean=0.0, std=1e-5, size=(2, number)) # normal distribution
            #noise = torch.randn(2, number) * 0.1  # normal distribution
            noise = torch.rand(2, number) - 0.5
            #noise[abs(noise)>1] = 0
            #noise[noise>1] = 1
            #noise[noise<-1] = -1
            noise = torch.stack((noise[0], noise[1], torch.zeros(number)), axis=0).to(self.device)
            noise = torch.mm(self.ray_matrices[i], noise)
            noise = noise.transpose(1, 0)[None, ...] # [1, N, 3]
            self.noises = torch.concat((self.noises, noise))

    # ref arXiv: https://4dqv.mpi-inf.mpg.de/NeRF-OSR/
    def ray_jitter(self, ray_bundle: RayBundle):  
        dev = ray_bundle.directions.device
        cam_indices = ray_bundle.camera_indices[..., 0].to(dev)
        #print (self.noises[cam_indices, self.train_count % len(ray_bundle)].shape)
        #print(self.noises.shape)
        ray_bundle.directions += self.noises[cam_indices, self.train_count % self.noises.shape[1]]

    # ref arXiv: https://4dqv.mpi-inf.mpg.de/NeRF-OSR/
    def ray_jitter_single_frame(self, ray_bundle: RayBundle):  
        # only ray_bundle on samle event frame
        dev = ray_bundle.directions.device
        ray_d = ray_bundle.directions
        #print(ray_bundle.camera_indices.shape)
        cam_indices = [ray_bundle.camera_indices[-1, 0], ray_bundle.camera_indices[0, 0]]
        #print(self.ray_matrices[cam_indices[1]].shape)
        noise = torch.randn(2, len(ray_d))-0.5 # [2, N_rand]
        split_len = int(len(ray_d)//2)
        noise = torch.stack((noise[0], noise[1], torch.zeros(len(ray_d))), axis=0).to(dev) # [3, N_rand]
        noise[:, :split_len] = torch.mm(self.ray_matrices[cam_indices[1]], noise[:, :split_len])
        noise[:, split_len:-1] = torch.mm(self.ray_matrices[cam_indices[0]], noise[:, split_len:-1])
        noise = noise.transpose(1, 0)  # [N_rand, 3]
        ray_bundle.directions = ray_d + noise

if __name__ == "__main__":

    cfg = ESNerfDataManagerConfig()
    data_manager = cfg.setup(device="cuda:0")
    ray, batch = data_manager.next_train(1)
    #print (ray)
    
