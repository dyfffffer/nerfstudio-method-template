import os
import glob
import json
from abc import abstractclassmethod
import numpy as np
from torch import Tensor
from torchtyping import TensorType
import torch

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type, List, Any, Literal
from jaxtyping import Float

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras import camera_utils
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json
#################### SemanticNerf ############################
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.dataparsers.base_dataparser import Semantics
#################### SemanticNerf ############################

# 将所有不同的数据集转换为通用格式
@dataclass
class ESNerfDataParserConfig(DataParserConfig):

    _target: Type = field(default_factory=lambda: ESNerfDataParser)
    data: Path = Path("/home/dengyufei/DATA/room")

    # dyfwhether or not to include loading of semantics data
    include_semantics: bool = True
    # downscale_factor: int = 4
    
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Sitcoms3D axis-aligned bbox will be scaled to this value.
    """

    # 相机原点缩放多少
    scale_factor: float = 1.0

    # 对感兴趣的区域缩放多少  semantic设置为2.0
    #scene_scale: float = 1.0
    scene_scale: float = 0.35
    # scene_scale: float = 2.0

    # 用于定向的方法
    orientation_menthod: Literal["pca", "up", "vertical", "none"] = "up"

    # 用于使位姿居中的方法
    center_method: Literal["poses", "focus", "none"] = "poses"

    # 是否会自动缩放姿势以适合+/- 1边界框
    auto_scale_poses: bool = True

    eval_mode: Literal["fraction", "filename", "interval", "all"] = "all"

    # 用于训练的图像的比例。其余的图像用于eval
    train_split_fraction: float = 0.9
    eval_interval: int = 125

    # 将深度值缩放为米。毫米到米转换的默认值为0.001
    depth_unit_scale_factor: float = 1e-3

# DataParser将根据split参数生成一个训练并计算DataparserOutputs
@dataclass
class ESNerfDataParser(DataParser):

    config: ESNerfDataParserConfig

    def _generate_dataparser_outputs(self, split="train", **kwargs: Optional[Dict]):

        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        if split != "train":
            split = "test"
        dir_path = self.config.data / f"{split}"

        #rgb的宽和高 
        Height = 960 # 540 480
        Width = 960 #960 640

        # event files
        enames = self.__find_files(dir_path / "events", ["*.npz"])
        assert len(enames) == 1, "event files not 1"
        event_files = {"event_files": enames}

        # rgb files
        image_filenames = self.__find_files(dir_path / "rgb", ["*.png"])
        fnames = []
        for filepath in image_filenames:
            fnames.append(Path(filepath).name)
        inds = np.argsort(fnames)
        image_filenames = [image_filenames[ind] for ind in inds]

        # pose filies
        pose_files = self.__find_files(dir_path/"pose", ["*.txt"])
        fnames = []
        for filepath in pose_files:
            fnames.append(Path(filepath).name)
        inds = np.argsort(fnames)
        pose_files = [pose_files[ind] for ind in inds]
       
        cam_cnt = len(pose_files)
        poses = []
        for i in range(0, cam_cnt):
            pose = self.__parse_txt(pose_files[i], (4, 4)).reshape(1, 4, 4)
            poses.append(pose)
        poses = torch.from_numpy(np.array(poses).astype(np.float32)).reshape((-1, 4, 4))
        poses[..., 0:3, 1:3] *= -1
     

        if self.config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
        elif self.config.eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif self.config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
        elif self.config.eval_mode == "all":
            CONSOLE.log(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")
        
        if split == "train":
            indices = i_train
        else:
            indices = i_eval

        # 旋转方向
        transform_matrix = torch.eye(4)
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
           poses, 
           method=self.config.orientation_menthod, 
           center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        #CONSOLE.print(scale_factor)

        poses[:, :3, 3] *= scale_factor

        # Chose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        # CONSOLE.print(image_filenames)
        #CONSOLE.print(poses)


        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32)
        )


        # intrinsic files
        intrinsic_files = self.__find_files(dir_path/"intrinsics", ["*.txt"])
        intrinsic = Tensor(self.__parse_txt(intrinsic_files[0], (-1, 4)))
        fx = intrinsic[0][0].clone()
        fy = intrinsic[1][1].clone()
        cx = intrinsic[0][2].clone()  
        cy = intrinsic[1][2].clone()
        distortion_params = torch.zeros(6)
        if intrinsic.shape[0] == 5:
            distortion_params[0] = intrinsic[4][0].clone()
            distortion_params[1] = intrinsic[4][1].clone()


        # --- semantics ---
        semantics = None
        if self.config.include_semantics:
            empty_path = Path()
            replace_this_path = str(empty_path / "rgb" / empty_path)
            with_this_path = str(empty_path / "semantic_class" / empty_path)
            filenames = [
                Path(str(image_filename).replace('rgb', 'semantic_class'))
                for image_filename in image_filenames
            ]
            panoptic_classes = load_from_json(self.config.data / "semantic.json")
            classes = panoptic_classes["classes"]
            colors = torch.tensor(panoptic_classes["colors"], dtype=torch.float32) / 255.0
            semantics = Semantics(filenames=filenames, colors=colors, classes=classes) #, colors=colors , mask_classes=["person"]

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4], 
            fx=fx, fy=fy, cx=cx, cy=cy, 
            distortion_params=distortion_params, 
            height=Height, width=Width,
        )


        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,  #dataparser_scale=scale,
            dataparser_transform=transform_matrix,
            metadata = {"event_files": event_files, "semantics": semantics},
            # metadata={"semantics": semantics} if self.config.include_semantics else {},
        )
        
        return dataparser_outputs

    def __find_files(self, dir, exts):
        if os.path.isdir(dir):
            files_grabbed = []
            for ext in exts:
                files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
            if len(files_grabbed) > 0:
                files_grabbed = sorted(files_grabbed)
            return files_grabbed
        else:
            return []
        
    def __parse_txt(self, filename, shape):
        assert os.path.isfile(filename), "file not exist"
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums], dtype=np.float32).reshape(shape)


if __name__ == "__main__":
    #print(Path("Hello ")/"s")
    config = ESNerfDataParserConfig()
    parser = config.setup()
    outputs = parser.get_dataparser_outputs()
    print(outputs.cameras.fx)