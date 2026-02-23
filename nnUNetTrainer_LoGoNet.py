import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.distributions.uniform import Uniform
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

import numpy as np
import torch
from torch import autocast
from typing import Tuple, Union, List
import warnings

from nnunetv2.training.loss.compound_losses import DC_SkelREC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache, dummy_context

from nnunetv2.training.dataloading.data_loader_2d_skel import nnUNetDataLoader2DSkel
from nnunetv2.training.dataloading.data_loader_3d_skel import nnUNetDataLoader3DSkel
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.skeletonization import SkeletonTransform


BNNorm2d = nn.BatchNorm2d
LNNorm = nn.LayerNorm
Activation = nn.GELU

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation()
        )

    def forward(self, x):
        x = self.up(x)
        return x

class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation()
        )

    def forward(self, x):
        x = self.down(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, groups=1):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1)
        self.bn1 = BNNorm2d(planes)
        self.act = Activation()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, groups=groups, padding=1)
        self.bn2 = BNNorm2d(planes)

        if self.inplanes != self.planes:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=1),
                BNNorm2d(planes)
            )
    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)

        if self.inplanes != self.planes:
            identity = self.down(x)

        out = self.bn2(out) + identity
        out = self.act(out)

        return out

class OPE(nn.Module):
    def __init__(self, inplanes, planes):
        super(OPE, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, 3, stride=1, padding=1)
        self.bn1 = BNNorm2d(inplanes)
        self.act = Activation()
        self.down = down_conv(inplanes, planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.down(out)

        return out

class local_block(nn.Module):
    def __init__(self, inplanes, hidden_planes, planes, groups=1, down_or_up=None):
        super(local_block, self).__init__()
        if down_or_up is None:
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
            )

        elif down_or_up == 'down':
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
                down_conv(hidden_planes, planes)
            )
        elif down_or_up == 'up':
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
                up_conv(hidden_planes, planes),
            )

    def forward(self, x):
        out = self.BasicBlock(x)
        return out


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = Activation()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class global_block(nn.Module):
    def __init__(self, in_dim, dim, num_heads, pool_size=3, mlp_ratio=4., drop=0., drop_path=0., sr_ratio=1):
        super().__init__()

        self.in_dim = in_dim
        self.dim = dim

        self.proj = nn.Conv2d(in_dim, dim, kernel_size=3,  padding=1)
        self.norm1 = GroupNorm(dim)
        self.attn = Pooling(pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = GroupNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PAM_Module(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class PositionEmbeddingLearned(nn.Module):
    """
    可学习的位置编码
    """
    def __init__(self, num_pos_feats=256, len_embedding=128):   # len_embedding=64/128
        super().__init__()
        self.row_embed = nn.Embedding(len_embedding, num_pos_feats)
        self.col_embed = nn.Embedding(len_embedding, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos

class ScaledDotProductAttention(nn.Module):
    '''自注意力模块'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature ** 0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        q = x.view(m_batchsize, d, -1)
        k = x.view(m_batchsize, d, -1)
        k = k.permute(0, 2, 1)
        v = x.view(m_batchsize, d, -1)

        attn = torch.matmul(q / self.temperature, k)

        if mask is not None:
            # 给需要mask的地方设置一个负无穷
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        output = output.view(m_batchsize, d, height, width)

        return output

class LoGoNet2D(nn.Module):
    def __init__(self, in_channel, num_classes, deep_supervised, layer_channel=[16, 32, 64, 128, 256], global_dim=[8, 16, 32, 64, 128], num_heads=[1, 2, 4, 8], sr_ratio=[8, 4, 2, 1], dropout_op=None, dropout_op_kwargs=None):
        super(LoGoNet2D, self).__init__()

        self.deep_supervised = deep_supervised

        self.input_l0 = nn.Sequential(
            nn.Conv2d(in_channel, layer_channel[0], kernel_size=3, stride=1, padding=1),
            BNNorm2d(layer_channel[0]),
            Activation(),
            nn.Conv2d(layer_channel[0], layer_channel[0], kernel_size=3, stride=1, padding=1),
            BNNorm2d(layer_channel[0]),
            Activation()
        )

        self.encoder1_l1_local = OPE(layer_channel[0], layer_channel[1])
        self.encoder1_l1_global = global_block(layer_channel[0], global_dim[0], num_heads=num_heads[0], sr_ratio=sr_ratio[0])

        self.encoder1_l2_local = OPE(layer_channel[1], layer_channel[2])
        self.encoder1_l2_global = global_block(layer_channel[1], global_dim[1], num_heads=num_heads[1], sr_ratio=sr_ratio[1])

        self.encoder1_l3_local = OPE(layer_channel[2], layer_channel[3])
        self.encoder1_l3_global = global_block(layer_channel[2], global_dim[2], num_heads=num_heads[2], sr_ratio=sr_ratio[2])

        
        self.encoder1_l4_local = OPE(layer_channel[3], layer_channel[4])
        self.encoder1_l4_global = global_block(layer_channel[3], global_dim[3], num_heads=num_heads[3], sr_ratio=sr_ratio[3])

        self.decoder1_l4_local_output = nn.Conv2d(layer_channel[4], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder1_l4_local = local_block(layer_channel[4], layer_channel[4], layer_channel[3], down_or_up='up')
        
        self.decoder1_l3_local_output = nn.Conv2d(layer_channel[3], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder1_l3_local = local_block(layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[2], down_or_up='up')
        
        self.decoder1_l2_local_output = nn.Conv2d(layer_channel[2], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder1_l2_local = local_block(layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[1], down_or_up='up')
        
        self.decoder1_l1_local_output = nn.Conv2d(layer_channel[1], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder1_l1_local = local_block(layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[0], down_or_up='up')

        self.output_l0 = nn.Sequential(
            local_block(layer_channel[0] + global_dim[0], layer_channel[0], layer_channel[0], down_or_up=None),
            nn.Conv2d(layer_channel[0], num_classes, kernel_size=1, stride=1, padding=0)
        )
        

        # 位置编码
        self.pos = PositionEmbeddingLearned(layer_channel[4] // 2)

        # 空间注意力机制
        self.pam = PAM_Module(layer_channel[4])

        # 自注意力机制
        self.sdpa = ScaledDotProductAttention(layer_channel[4])
        
        # 添加dropout层  
        if dropout_op is not None:  
            self.dropout = dropout_op(**dropout_op_kwargs)  
        else:  
            self.dropout = None  
        
    def _init_weights(self, m):
        # initialization
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):

        outputs = []
        # encoder-decoder 1
        x_e1_l0 = self.input_l0(x)
        
        # if self.dropout is not None:  
        #     x_e1_l0 = self.dropout(x_e1_l0)  
        
        x_e1_l1_local = self.encoder1_l1_local(x_e1_l0)
        x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)

        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local)
        x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)

        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local)
        x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)

        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local)
        x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

        x_l1 = x_e1_l4_local

        # TSA l-latent space feature
        x_l1_pam = self.pam(x_l1)
        
        # GSA
        x_l1_pos = self.pos(x_l1)
        x_l1 = x_l1 + x_l1_pos
        x_l1_sdpa = self.sdpa(x_l1)
        
        # 融合
        x_l1 = x_l1_sdpa + x_l1_pam
        
        x_e1_l4_local = x_l1
        
        outputs.append(self.decoder1_l4_local_output(x_e1_l4_local))
        x_d1_l3_local = self.decoder1_l4_local(x_e1_l4_local)


        x_d1_l3 = torch.cat((x_d1_l3_local, x_e1_l3_global), dim=1)
        outputs.append(self.decoder1_l3_local_output(x_d1_l3_local))
        x_d1_l2_local = self.decoder1_l3_local(x_d1_l3)
        
        x_d1_l2 = torch.cat((x_d1_l2_local, x_e1_l2_global), dim=1)
        outputs.append(self.decoder1_l2_local_output(x_d1_l2_local))
        x_d1_l1_local = self.decoder1_l2_local(x_d1_l2)
        
        x_d1_l1 = torch.cat((x_d1_l1_local, x_e1_l1_global), dim=1)
        outputs.append(self.decoder1_l1_local_output(x_d1_l1_local))
        x_d1_l0_local = self.decoder1_l1_local(x_d1_l1)
 
             
        
        x_d1_l0 = torch.cat((x_d1_l0_local, x_e1_l0_global), dim=1)
        
        outputs.append(self.output_l0(x_d1_l0))
        outputs = outputs[::-1]

        if self.deep_supervised:
            r = outputs
        else:
            r = outputs[0]
        return r



from torch._dynamo import OptimizedModule
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
\
class nnUNetTrainer_LoGoNet(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        # unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.initial_lr = 1e-2
        # self.weight_decay = 3e-5
        # self.num_epochs = 1000
        self.num_epochs = 100
        # self.num_epochs = 50

        
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.deep_supervised = enabled

    @staticmethod
    def build_network_architecture(architecture_class_name,
                                   arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision):
        # patch_size = self.configuration_manager.patch_size
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He
        model = LoGoNet2D(in_channel=num_input_channels, num_classes=num_output_channels, deep_supervised=enable_deep_supervision)
        model.apply(InitWeights_He(1e-2))
        return model
    
class nnUNetTrainer_LoGoNet_SRL(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        # unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.initial_lr = 1e-2
        # 调节weight_decay防止过拟合,轻度过拟合尝试1e-4 ~ 1e-3
        self.weight_decay = 3e-5
        # self.weight_decay = 1e-3
        # self.num_epochs = 1000
        self.num_epochs = 100
        # self.num_epochs = 50
        
        # 骨架损失权重，需调节参数
        # === 1. 定义目标权重和 Warm-up 时长 ===
        self.weight_srec = 0.01  # 最终想要达到的权重
        # ==========================================
        # 是否启用管状膨胀
        # self.do_tube = True
        # self.do_tube = False
        
    ################ add SRLoss ##############
    def _build_loss(self):
        from nnunetv2.training.loss.compound_losses import DC_SkelREC_and_CE_loss  
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper  
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        
        if self.label_manager.ignore_label is not None:
            warnings.warn('Support for ignore label with Skeleton Recall is experimental and may not work as expected')
        loss = DC_SkelREC_and_CE_loss(soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 
                                                        'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
                                      soft_skelrec_kwargs={'batch_dice': self.configuration_manager.batch_dice, 
                                                           'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
                                      ce_kwargs={}, weight_ce=1, weight_dice=1, weight_srec=self.weight_srec, 
                                      ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    
    
    def get_dataloaders(self):
        from nnunetv2.training.dataloading.data_loader_2d_skel import nnUNetDataLoader2DSkel  
        from nnunetv2.training.data_augmentation.custom_transforms.skeletonization import SkeletonTransform
    
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size=patch_size, 
            rotation_for_DA=rotation_for_DA, 
            deep_supervision_scales=deep_supervision_scales, 
            mirror_axes=mirror_axes, 
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DSkel(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader2DSkel(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        else:
            dl_tr = nnUNetDataLoader3DSkel(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader3DSkel(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
    
    def get_tr_and_val_datasets(self):  
        # create dataset split  
        tr_keys, val_keys = self.do_split()  
    
        # load the datasets for training and validation using standard nnUNetDataset  
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetB2nd  
        dataset_tr = nnUNetDatasetB2nd(self.preprocessed_dataset_folder, tr_keys,  
                                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,  
                                num_images_properties_loading_threshold=0)  
        dataset_val = nnUNetDatasetB2nd(self.preprocessed_dataset_folder, val_keys,  
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,  
                                    num_images_properties_loading_threshold=0)  
        return dataset_tr, dataset_val
    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        transforms.append(SkeletonTransform(do_tube=False))
        # transforms.append(SkeletonTransform(do_tube=True))
        
        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        transforms.append(SkeletonTransform(do_tube=False))
        # transforms.append(SkeletonTransform(do_tube=True))
        
        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        skel = batch['skel']

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0].cpu().numpy(), name='data')
        # viewer.add_image(target[0][0].cpu().numpy(), name='target')
        # viewer.add_image(skel[0][0].cpu().numpy(), name='skel')
        # napari.run()

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            skel = [i.to(self.device, non_blocking=True) for i in skel]
        else:
            target = target.to(self.device, non_blocking=True)
            skel = skel.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target, skel)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        skel = batch['skel']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            skel = [i.to(self.device, non_blocking=True) for i in skel]
        else:
            target = target.to(self.device, non_blocking=True)
            skel = skel.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target, skel)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    #######################################
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.deep_supervised = enabled

    @staticmethod
    def build_network_architecture(architecture_class_name,
                                   arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision):
        # 添加dropout
        arch_init_kwargs['dropout_op'] = 'torch.nn.Dropout'  
        arch_init_kwargs['dropout_op_kwargs'] = {'p': 0.1}
        # 导入dropout操作  
        from pydoc import locate  
        dropout_op = locate(arch_init_kwargs['dropout_op'])  
        dropout_op_kwargs = arch_init_kwargs['dropout_op_kwargs'] 
        
        # patch_size = self.configuration_manager.patch_size
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He
        model = LoGoNet2D(in_channel=num_input_channels, num_classes=num_output_channels, deep_supervised=enable_deep_supervision,
                       dropout_op=dropout_op,  
                       dropout_op_kwargs=dropout_op_kwargs)
        model.apply(InitWeights_He(1e-2))
        return model
