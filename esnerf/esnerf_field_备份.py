"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""
from typing import Literal, Optional, Dict, Tuple, Type
from dataclasses import dataclass, field
from jaxtyping import Float, Shaped

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.field_components.encodings import SHEncoding, HashEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion, SceneContraction
from nerfstudio.fields.base_field import Field, get_normalized_directions  # for custom Field

try:
    import tinycudann as tcnn
except ImportError:
    pass

##################################################################
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
)
from nerfstudio.field_components.encodings import  Encoding, Identity
##################################################################

# Field是一个模型组件，它将空间区域与某种quantity联系起来
# 最典型的情况下，字段的输入是3D位置和观看方向，输出是密度和颜色
class ESNerfField(Field):
    """ref: NerfactoField"""
    """ESNerf Field
    
    Args:
        num_semantic_classes: Number of distinct semantic classes.
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
    """

    aabb: Tensor

    def __init__(
        self,
        # axis-algned bounding box 轴对齐边界框
        aabb: Tensor,   
        
        num_semantic_classes: int,
        # dyfadd
        num_images: int,
        use_average_appearance_embedding: bool = False,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        hidden_dim_transient: int = 64,
        pass_semantic_gradients: bool = False,

        position_encoding: Encoding = Identity(in_dim=3),
        # direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        
        # mlp网络层数 ？
        num_layers: int = 2,
        # mlp网络的隐藏层维度 ？
        hidden_dim: int = 64,
        # 地理特征的维度 ？
        geo_feat_dim: int = 15,
        # 表示哈希表的层级数 ？
        num_levels: int = 16,
        # 基本分辨率 用在哪
        base_res: int = 16,
        # 最大分辨率
        max_res: int = 2048,
        # 哈希表大小的对数？
        log2_hashmap_size: int = 19,
        # 颜色处理的MLP层数
        num_layers_color: int = 3,
        # 每个层级的特征数
        features_per_level: int = 2,
        # 颜色处理MLP的隐藏层维度
        hidden_dim_color: int = 64,
        # 用于处理空间失真
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torcn"] = "tcnn",
        
        

        
        
    ) -> None:
        super().__init__()
        # dyfadd
        self.num_images = num_images
        self.use_average_appearance_embedding = use_average_appearance_embedding
        use_transient_embedding = use_transient_embedding
        self.num_semantic_classes = num_semantic_classes
        self.use_semantics = use_semantics
        self.pass_semantic_gradients = pass_semantic_gradients
        
        # 执行方向编码的模块
        self.position_encoding = position_encoding
        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        """将一些张量参数注册为模块的缓冲区，以确保它们在模块的状态字典中。这些缓冲区在模型的训练过程中不会更新，但可以被访问"""
        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hash_map_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.base_res = base_res


        #self.direction_encoding = NeRFEncoding(
        #    in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        #)

        '''
        mlp_base_grid, hashencoding的一个实例 
        哈希编码(HashEncoding)模块, 用于对基本网格数据执行哈希编码。
        执行某种哈希编码的操作 其中的参数用于控制编码的行为
        这个编码结果的输出维度将用于后续的层'''
        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
            # 线性插值
            interpolation="Linear",
        )

        '''
        mlp_base_mlp是个MLP模型
        mlp_base_grid的编码输出 是该模型的输入
        通过几个全连接层处理
        '''
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,            # num_layers=base_mlp_num_layers, event:2; se:8
            layer_width=hidden_dim,           # layer_width=base_mlp_layer_width, event:64; se:256
            out_dim=1 + self.geo_feat_dim,
            # 这是应用于隐藏层（中间层）的输出。激活函数是ReLU,这使神经元的输出大于0时保持不变，否则输出为0
            activation=nn.ReLU(),       
            # out_activation=nn.ReLU()这是semantic_nerf的设置。这个应用于神经网络的输出层。输出层的激活函数取决于任务类型。ReLU可以用于回归任务、分类任务      
            out_activation=None,              
            implementation=implementation,
            # skip_connections=skip_connections,  跳跃是指将某一层的输出连接到非连续的后续曾。参数解释：Where to add skip connection in base MLP.
        )

        '''
        mlp_base是一个mlp_base_grid和mlp_base_mlp堆叠起来的Sequential对象
        这样mlp_base_grid的输出就是mlp_base_mlp的输入了
        '''
        self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)
        
        '''
        mlp_head是一个MLP模型, 用于颜色处理
        输入维度由direction_encoding的输出维度 + geo_feat_dim(通常表示与几何特征相关的维度)
        num_layers: 隐藏层数量
        layer_width: 隐藏层中神经元的数量 加了color 不知道什么意思
        implementation: 这是有关实现细节的参数，通常用于优化和性能方面的调整。
        '''
        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
            num_layers=num_layers_color,   # num_layers=head_mlp_num_layers, event:3; se:2
            layer_width=hidden_dim_color,  # layer_width=head_mlp_layer_width, event:64; se:128
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),   # out_activation=nn.ReLU(),
            implementation=implementation,
        )
        
        self.mlp_semantic = MLP(
            in_dim=self.mlp_head.get_out_dim(),
            layer_width=self.mlp_head.layer_width // 2,
            num_layers=1,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )

        # self.field_head_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())  # semantic_nerf中用于求密度的，这里不用
        # self.field_head_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim())  #同理
        self.field_head_semantic = SemanticFieldHead(
            in_dim=self.mlp_semantic.get_out_dim(), num_classes=self.num_semantic_classes
        )

        # semantics
        if self.use_semantics:
            self.mlp_semantics = MLP(
                in_dim=self.geo_feat_dim,
                num_layers=2,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.get_out_dim(), num_classes=num_semantic_classes
            )

    '''
    计算并返回密度。返回密度和特征的张量
    ray_samples:采样位置以计算密度
    '''
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim = -1)
        self._density_before_activation = density_before_activation

        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out
    
    
    '''
    计算并返回颜色
    ray_samples:采样位置以计算密度
    density_embedding: Density embeddings to condition on 密度嵌入条件？
    '''
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None

        # dyfadd
        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)  # 把ray_samples_2带入就出错
        mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
        outputs = {}
        # rgb 用event的了
        # outputs[self.field_head_rgb.field_head_name] = self.field_head_rgb(mlp_out)
        # semantic
        mlp_out_sem = self.mlp_semantic(mlp_out)
        outputs[self.field_head_semantic.field_head_name] = self.field_head_semantic(mlp_out_sem)
        # dyfover

        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        h = torch.cat(
            [
                d, 
                density_embedding.view(-1, self.geo_feat_dim),
            ],
            dim = -1,
        )

        # rgb 没有改
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})
        
        # 这个不是semantic_filed的get_output，不知大是哪的
        # # appearance 不知道apperance代表什么 dyfadd 但是默认值是false，还没用到
        # camera_indices = ray_samples.camera_indices.squeeze()
        # if self.training:
        #     embedded_appearance = self.embedding_appearance(camera_indices)
        # else:
        #     if self.use_average_appearance_embedding:
        #         embedded_appearance = torch.ones(
        #             (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
        #         ) * self.embedding_appearance.mean(dim=0)
        #     else:
        #         embedded_appearance = torch.zeros(
        #             (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
        #         )

        # # semantics
        # if self.use_semantics:
        #     semantics_input = density_embedding.view(-1, self.geo_feat_dim)
        #     if not self.pass_semantic_gradients:
        #         semantics_input = semantics_input.detach()

        #     x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
        #     outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        return outputs


