"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, Any
from jaxtyping import Float
import numpy as np

import nerfacc
import torch
from torch.nn import Parameter
from torch import Tensor
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import VolumetricSampler, UniformSampler, PDFSampler, LogSampler, NeuSSampler, ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import colormaps, colors, misc



#####################################################
from esnerf.esnerf_field import ESNerfField
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.renderers import SemanticRenderer, SemanticRenderer, UncertaintyRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
#####################################################



@dataclass
class ESNerfModelConfig(ModelConfig):
    """ESNerf Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: ESNerfModel)

    # dyf.Whether to use transient embedding.dyf
    use_transient_embedding: bool = False
    
    pass_semantic_gradients: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    use_average_appearance_embedding: bool = True
    use_same_proposal_network: bool = False
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""



    # 权重
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0, "event_loss": 0.0, "semantic_loss": 0.0})
    

    num_layers: int = 2
    hidden_dim: int = 64
    num_layers_color: int = 3
    hidden_dim_color: int = 64

    num_levels:int = 16 # 64 #32 #16
    features_per_level: int = 2
    grid_resolution: int = 128
    grid_levels: int = 4
    max_res: int = 1024
    log2_hashmap_size: int = 19
    alpha_thre: float = 1e-3
    cone_angle: float = 0 #0.004

    render_step_size: Optional[float] = None
    near_plane: float = 0.05
    far_plane: float = 2 #1e3
    enable_collider: bool = False
    collider_params: Optional[Dict[str, float]] = None

    """Whether to randomize the background color."""
    background_color: Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]] = Tensor([0.62, 0.62, 0.62])
    #background_color: Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]] = "random"
    
    disable_scene_contraction: bool = True 


class ESNerfModel(Model):
    """ESNerf Model."""

    config: ESNerfModelConfig

    # dyf.新增了一个metadata的参数.dyf 新增了对输入数据中的semantics的存储和colormap的处理
    def __init__(self, config: ESNerfModelConfig, metadata: Dict,  **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        super().__init__(config=config, **kwargs)
        self.colormap = self.semantics.colors.clone().detach().to(self.device)

    # 设置field和模块（sampler\render等
    def populate_modules(self):
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf"))

        if self.config.use_transient_embedding:
            raise ValueError("Transient embedding is not fully working for semantic nerf-w.")

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = ESNerfField(
            aabb=self.scene_box.aabb,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            num_layers_color=self.config.num_layers_color,
            hidden_dim_color=self.config.hidden_dim_color,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
            # dyfadd
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_transient_embedding=self.config.use_transient_embedding,
            use_semantics=True,
            num_semantic_classes=len(self.semantics.classes),
            pass_semantic_gradients=self.config.pass_semantic_gradients,
        )



        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # dyfadd 改变他的采样方式，先不改
        # Build the proposal network(s) 构建一个先验网络？
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
            self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for _ in range(self.config.num_proposal_iterations)]
        else:
            for _ in range(self.config.num_proposal_iterations):
                network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
                self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for network in self.proposal_networks]

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        # dyfadd

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) **2).sum().sqrt().item() / 1000
        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )
        # dyfadd 采样方法应该用一个就够了 可能可以各用各的？
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        # dyfadd
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()
        
        # losses
        self.event_loss = MSELoss()
        self.rgb_loss = MSELoss()
        # dyfadd
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    # 返回训练回调函数，例如更新密度grid
    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x) * self.config.render_step_size,
            )



        # dyfadd semantic这里的callback是List，而event不是
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            # def set_anneal(step):
            #     # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            #     train_frac = np.clip(step / N, 0, 1)

            #     def bias(x, b):
            #         return b * x / ((b - 1) * x + 1)

            #     anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
            #     self.proposal_sampler.set_anneal(anneal)

            # callbacks.append(
            #     TrainingCallback(
            #         where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],  # 表示回调函数将在每个训练迭代之前执行
            #         update_every_num_iters=1,
            #         func=set_anneal,  # 回调函数执行时调用
            #     )
            # )

            # 融合了event和semantic的
            def set_anneal_updata_occupancy_grid(step):
                self.occupancy_grid.update_every_n_steps(
                    step=step,
                    occ_eval_fn=lambda x: self.field.density_fn(x) * self.config.render_step_size,
                )
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],  # 表示回调函数将在每个训练迭代之前执行
                    update_every_num_iters=1,
                    func=set_anneal_updata_occupancy_grid,  # 回调函数执行时调用
                )
            )
        return callbacks

        # 原来eventnerf的
        # return [
        #     TrainingCallback(
        #         where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
        #         update_every_num_iters=1,
        #         func=update_occupancy_grid,
        #     ),
        # ]
        
    
    # 返回优化模型组件的参数组
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        #dyfadd
        # param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        #dyfadd
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups
    
    # 产生一个RayBundle对象，并返回用于描述每条射线的quanties的RayOutputs
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor | List]:
        assert self.field is not None
        num_rays = len(ray_bundle)


        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        ray_samples_2, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)



        

        field_outputs = self.field(ray_samples)
        field_outputs_2= self.field(ray_samples_2)

        # print("field_outputs",field_outputs[FieldHeadNames.RGB].size())
        # flag = 1 
        # assert flag != 1

        #dyfadd
        if self.training and self.config.use_transient_embedding:
            density = field_outputs[FieldHeadNames.DENSITY] + field_outputs[FieldHeadNames.TRANSIENT_DENSITY]
            weights = ray_samples_2.get_weights(density)
            weights_static = ray_samples_2.get_weights(field_outputs[FieldHeadNames.DENSITY])
            rgb_static_component = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            rgb_transient_component = self.renderer_rgb(
                rgb=field_outputs[FieldHeadNames.TRANSIENT_RGB], weights=weights
            )
            rgb = rgb_static_component + rgb_transient_component
        else:

            weights_static = ray_samples_2.get_weights(field_outputs_2[FieldHeadNames.DENSITY])
            weights = weights_static
            # print(weights_static.size())  torch.Size([8192, 48, 1])
            # rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        weights_list.append(weights_static)
        ray_samples_list.append(ray_samples_2)
        #dyfadd

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        # print("//////////////////////////??")
        # print(field_outputs[FieldHeadNames.RGB].size())
        # print(weights.size())
        # print(ray_indices.size())
        # print(num_rays.size())
        # print("//////////////////////////??")

        # flag = 1
        # assert flag != 1

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays,
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            #"num_samples_per_ray": packed_info[:, 1],
        }
        outputs["weights_list"] = weights_list
        outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        #dyfadd
        # transients 瞬变
        if self.training and self.config.use_transient_embedding:
            weights_transient = ray_samples_2.get_weights(field_outputs_2[FieldHeadNames.TRANSIENT_DENSITY])
            uncertainty = self.renderer_uncertainty(field_outputs_2[FieldHeadNames.UNCERTAINTY], weights_transient)
            outputs["uncertainty"] = uncertainty + 0.03  # NOTE(ethan): this is the uncertainty min
            outputs["density_transient"] = field_outputs_2[FieldHeadNames.TRANSIENT_DENSITY]

        # semantics
        semantic_weights = weights_static
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        '''
        field_outputs[FieldHeadNames.SEMANTICS]  torch.Size([8192, 48, 81])
        semantic_weights  torch.Size([8192, 48, 1])
        可以相乘，先不管
        '''

        outputs["semantics"] = self.renderer_semantics(
            field_outputs_2[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        print(outputs["semantics"].shape)
        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]
        #dyfadd

        return outputs
    
    # 返回指标字典， which will be plotted with comet, wandb or tensorboard."
    def get_metrics_dict(self, outputs, batch):
        return {}
        '''
        image = batch["image"].to(self.deivce)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}outputs
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict
        '''
    
    # 返回图像和指标的字典到plot。这里可以用自己的colormap 没改
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)

        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}

        images_dict = { "img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        
        # dyfadd
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        # semantics
        semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        images_dict["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        # valid mask
        images_dict["mask"] = batch["mask"].repeat(1, 1, 3).to(self.device)

        return metrics_dict, images_dict
    
    # 返回loss的字典，用于加和得到最终的loss
    def get_loss_dict(self, outputs, batch, metric_dict=None):
        image = batch["image"][..., :3].to(self.device)
        print(outputs["rgb"])
        print('////////////////////////')
        print(outputs["accumulation"].shape)
        print('////////////////////////')
        print(image.shape)
        print('////////////////////////')
        assert 0


        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )


        rgb_loss = self.rgb_loss(image, pred_rgb)

        
        ef = batch["event_frame_selected"].to(self.device)
        # print("ef.size",ef.size())
        

        log_rgb = torch.log(pred_rgb)
        log_rgb = log_rgb.reshape(2, len(log_rgb) // 2, 3)
        diff = (log_rgb[1] - log_rgb[0]) * (ef != 0)
        event_loss = self.event_loss(ef / ef.max(), diff / diff.max()) + \
                    self.event_loss(ef / ef.min(), diff / diff.min()) + \
                    self.event_loss(ef / (ef.max() - ef.min()), diff / (diff.max() - diff.min()))
        #event_loss = self.event_loss(ef * 0.25, diff * 2.2)

        # outputs["semantics"]的批次大小为8192 是field中网络的输出
        # batch["semantics"][..., 0].long().to(self.device)的批次大小为4096 config规定的，是datamanager读到的
        # 原来的eventnerf就是这样, 但是为什么呢 为什么batch["image"]的大小为8192 event和semantic的大小都是4096
        #  恒定的2倍关系
        semantics_loss = 0
        # semantics_loss = self.cross_entropy_loss(outputs["semantics"], batch["semantics"][..., 0].long().to(self.device))

        loss_dict = {"event_loss:": event_loss, "rgb_loss": rgb_loss, "semantics_loss": semantics_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

        # transient loss
        # if self.training and self.config.use_transient_embedding:
        #     betas = outputs["uncertainty"]
        #     loss_dict["uncertainty_loss"] = 3 + torch.log(betas).mean()
        #     loss_dict["density_loss"] = 0.01 * outputs["density_transient"].mean()
        #     loss_dict["rgb_loss"] = (((image - outputs["rgb"]) ** 2).sum(-1) / (betas[..., 0] ** 2)).mean()
        # else:
        #     loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        

        return loss_dict

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
