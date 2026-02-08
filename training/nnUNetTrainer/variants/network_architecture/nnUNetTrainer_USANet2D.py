import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.distributions.uniform import Uniform
import numpy as np
from timm.models.layers import DropPath, trunc_normal_


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
    def __init__(self, num_pos_feats=256, len_embedding=32):
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

class USANet2D(nn.Module):
    def __init__(self, in_channel, num_classes, deep_supervised, layer_channel=[16, 32, 64, 128, 256], global_dim=[8, 16, 32, 64, 128], num_heads=[1, 2, 4, 8], sr_ratio=[8, 4, 2, 1]):
        super(USANet2D, self).__init__()

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
        # self.apply(self._init_weights)

        # 位置编码
        self.pos = PositionEmbeddingLearned(layer_channel[4] // 2)

        # 空间注意力机制
        self.pam = PAM_Module(layer_channel[4])

        # 自注意力机制
        self.sdpa = ScaledDotProductAttention(layer_channel[4])
        
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

# 消融实验-w/o Global Branch
class USANet2D_NG(nn.Module):
    def __init__(self, in_channel, num_classes, deep_supervised, layer_channel=[16, 32, 64, 128, 256], 
                 global_dim=[16, 32, 64, 128, 256], 
                 num_heads=[1, 2, 4, 8], sr_ratio=[8, 4, 2, 1], dropout_op=None, dropout_op_kwargs=None):
        super(USANet2D_NG, self).__init__()

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
        # x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)

        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local)
        # x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)
        
        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local)
        # x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)

        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local)
        # x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

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
        
        # --------------Decoder--------------
        outputs.append(self.decoder1_l4_local_output(x_e1_l4_local))
        x_d1_l3_local = self.decoder1_l4_local(x_e1_l4_local)


        # x_d1_l3 = torch.cat((x_d1_l3_local, x_e1_l3_global), dim=1)
        x_d1_l3 = torch.cat((x_d1_l3_local, x_e1_l3_local), dim=1)
        outputs.append(self.decoder1_l3_local_output(x_d1_l3_local))
        x_d1_l2_local = self.decoder1_l3_local(x_d1_l3)
        
        x_d1_l2 = torch.cat((x_d1_l2_local, x_e1_l2_local), dim=1)
        outputs.append(self.decoder1_l2_local_output(x_d1_l2_local))
        x_d1_l1_local = self.decoder1_l2_local(x_d1_l2)
        
        x_d1_l1 = torch.cat((x_d1_l1_local, x_e1_l1_local), dim=1)
        outputs.append(self.decoder1_l1_local_output(x_d1_l1_local))
        x_d1_l0_local = self.decoder1_l1_local(x_d1_l1)
 
             
        
        x_d1_l0 = torch.cat((x_d1_l0_local, x_e1_l0), dim=1)
        
        outputs.append(self.output_l0(x_d1_l0))
        outputs = outputs[::-1]

        if self.deep_supervised:
            r = outputs
        else:
            r = outputs[0]
        return r

from torch._dynamo import OptimizedModule
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainer_VSANet2D(nnUNetTrainer):
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
        model = USANet2D(in_channel=num_input_channels, num_classes=num_output_channels, deep_supervised=enable_deep_supervision)
        model.apply(InitWeights_He(1e-2))
        return model
    
class nnUNetTrainer_USANet2D(nnUNetTrainer):
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
        model = USANet2D(in_channel=num_input_channels, num_classes=num_output_channels, deep_supervised=enable_deep_supervision)
        model.apply(InitWeights_He(1e-2))
        return model

class nnUNetTrainer_USANet2D_NG(nnUNetTrainer):
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
        # self.weight_decay = 3e-5
        # self.num_epochs = 1000
        self.num_epochs = 100
        # self.num_epochs = 50

class NNUNet_Block_Up(nn.Module):
    """
    仿照 nnU-Net 的基础卷积模块：
    [Conv3x3 - IN - LeakyReLU] x 2  -->  [ConvTranspose2d (Upsample)]
    """
    def __init__(self, in_ch, mid_ch, out_ch, down_or_up='up'):
        super().__init__()
        
        # nnU-Net 标准的双层卷积块
        self.conv = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            
            # 第二层卷积
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # 根据指令决定是否上采样
        if down_or_up == 'up':
            # 使用转置卷积进行 2x 上采样，同时调整通道数到 out_ch
            self.project = nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=2, stride=2, bias=False)
        else:
            # 如果不进行上采样 (用于 output_l0)，仅使用 1x1 卷积调整通道数
            if mid_ch != out_ch:
                self.project = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
            else:
                self.project = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.project(x)
        return x
    
class USANet2D_NL(nn.Module):
    def __init__(self, in_channel, num_classes, deep_supervised, layer_channel=[16, 32, 64, 128, 256], global_dim=[8, 16, 32, 64, 128], num_heads=[1, 2, 4, 8], sr_ratio=[8, 4, 2, 1]):
        super(USANet2D_NL, self).__init__()

        self.deep_supervised = deep_supervised

        # Input Block 保持不变 (也可以换成 nnUNet 风格，但通常影响较小)
        self.input_l0 = nn.Sequential(
            nn.Conv2d(in_channel, layer_channel[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer_channel[0]), # 注意：原代码用的 BNNorm2d，此处假设为 BN
            nn.ReLU(inplace=True),            # 注意：原代码用的 Activation
            nn.Conv2d(layer_channel[0], layer_channel[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer_channel[0]),
            nn.ReLU(inplace=True)
        )

        # Encoder (OPE) 保持不变，还是同时做 Local 和 Global 分支
        self.encoder1_l1_local = OPE(layer_channel[0], layer_channel[1])
        self.encoder1_l1_global = global_block(layer_channel[0], global_dim[0], num_heads=num_heads[0], sr_ratio=sr_ratio[0])

        self.encoder1_l2_local = OPE(layer_channel[1], layer_channel[2])
        self.encoder1_l2_global = global_block(layer_channel[1], global_dim[1], num_heads=num_heads[1], sr_ratio=sr_ratio[1])

        self.encoder1_l3_local = OPE(layer_channel[2], layer_channel[3])
        self.encoder1_l3_global = global_block(layer_channel[2], global_dim[2], num_heads=num_heads[2], sr_ratio=sr_ratio[2])
        
        self.encoder1_l4_local = OPE(layer_channel[3], layer_channel[4])
        self.encoder1_l4_global = global_block(layer_channel[3], global_dim[3], num_heads=num_heads[3], sr_ratio=sr_ratio[3])

        # -------------------------------------------------------------------------
        # [修改区域] Decoder 部分：将 local_block 替换为 nnUNet_Block_Up
        # -------------------------------------------------------------------------
        
        # L4 -> L3
        self.decoder1_l4_local_output = nn.Conv2d(layer_channel[4], num_classes, kernel_size=1, stride=1, padding=0)
        # 替换 local_block
        self.decoder1_l4_local = NNUNet_Block_Up(layer_channel[4], layer_channel[4], layer_channel[3], down_or_up='up')
        
        # L3 -> L2
        self.decoder1_l3_local_output = nn.Conv2d(layer_channel[3], num_classes, kernel_size=1, stride=1, padding=0)
        # 替换 local_block
        self.decoder1_l3_local = NNUNet_Block_Up(layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[2], down_or_up='up')
        
        # L2 -> L1
        self.decoder1_l2_local_output = nn.Conv2d(layer_channel[2], num_classes, kernel_size=1, stride=1, padding=0)
        # 替换 local_block
        self.decoder1_l2_local = NNUNet_Block_Up(layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[1], down_or_up='up')
        
        # L1 -> L0
        self.decoder1_l1_local_output = nn.Conv2d(layer_channel[1], num_classes, kernel_size=1, stride=1, padding=0)
        # 替换 local_block
        self.decoder1_l1_local = NNUNet_Block_Up(layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[0], down_or_up='up')

        # Output L0
        self.output_l0 = nn.Sequential(
            # 这里的 local_block 不做上采样 (down_or_up=None)，也替换掉
            NNUNet_Block_Up(layer_channel[0] + global_dim[0], layer_channel[0], layer_channel[0], down_or_up=None),
            nn.Conv2d(layer_channel[0], num_classes, kernel_size=1, stride=1, padding=0)
        )
        
        # 保持其他辅助模块不变
        self.pos = PositionEmbeddingLearned(layer_channel[4] // 2)
        self.pam = PAM_Module(layer_channel[4])
        self.sdpa = ScaledDotProductAttention(layer_channel[4])
        
    def forward(self, x):
        # forward 函数与原代码完全一致，因为我们保持了模块的输入输出维度兼容性
        outputs = []
        x_e1_l0 = self.input_l0(x)

        x_e1_l1_local = self.encoder1_l1_local(x_e1_l0)
        x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)

        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local)
        x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)

        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local)
        x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)

        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local)
        x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

        x_l1 = x_e1_l4_local

        # Bottleneck Attention
        x_l1_pam = self.pam(x_l1)
        x_l1_pos = self.pos(x_l1)
        x_l1 = x_l1 + x_l1_pos
        x_l1_sdpa = self.sdpa(x_l1)
        x_l1 = x_l1_sdpa + x_l1_pam
        
        x_e1_l4_local = x_l1
        
        # Decoder 1
        outputs.append(self.decoder1_l4_local_output(x_e1_l4_local))
        x_d1_l3_local = self.decoder1_l4_local(x_e1_l4_local) # Uses NNUNet Block
        
        x_d1_l3 = torch.cat((x_d1_l3_local, x_e1_l3_global), dim=1)
        outputs.append(self.decoder1_l3_local_output(x_d1_l3_local))
        x_d1_l2_local = self.decoder1_l3_local(x_d1_l3) # Uses NNUNet Block
        
        x_d1_l2 = torch.cat((x_d1_l2_local, x_e1_l2_global), dim=1)
        outputs.append(self.decoder1_l2_local_output(x_d1_l2_local))
        x_d1_l1_local = self.decoder1_l2_local(x_d1_l2) # Uses NNUNet Block
        
        x_d1_l1 = torch.cat((x_d1_l1_local, x_e1_l1_global), dim=1)
        outputs.append(self.decoder1_l1_local_output(x_d1_l1_local))
        x_d1_l0_local = self.decoder1_l1_local(x_d1_l1) # Uses NNUNet Block
        
        x_d1_l0 = torch.cat((x_d1_l0_local, x_e1_l0_global), dim=1)
        
        outputs.append(self.output_l0(x_d1_l0)) # Uses NNUNet Block inside Sequential
        outputs = outputs[::-1]

        if self.deep_supervised:
            r = outputs
        else:
            r = outputs[0]
        return r

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

class nnUNetTrainer_USANet2D_NL(nnUNetTrainer):
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
        model = USANet2D_NL(in_channel=num_input_channels, num_classes=num_output_channels, deep_supervised=enable_deep_supervision)
        model.apply(InitWeights_He(1e-2))
        return model
    
class NNUNet_Encoder_Block(nn.Module):
    """
    nnU-Net 风格的编码器模块：
    1. 使用 stride=2 的卷积进行下采样 (Downsampling)
    2. 使用 InstanceNorm 和 LeakyReLU
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # 第一层：负责下采样 (stride=2) 和升维
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            
            # 第二层：特征提取 (stride=1)
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class USANet2D_NoOPE(nn.Module):
    def __init__(self, in_channel, num_classes, deep_supervised, layer_channel=[16, 32, 64, 128, 256], global_dim=[8, 16, 32, 64, 128], num_heads=[1, 2, 4, 8], sr_ratio=[8, 4, 2, 1]):
        super(USANet2D_NoOPE, self).__init__()

        self.deep_supervised = deep_supervised

        # Input Block (Level 0) - 保持全分辨率
        # 也可以替换为 nnUNet 风格 (Conv stride=1)，但为了仅消融 OPE，此处可保持不变
        self.input_l0 = nn.Sequential(
            nn.Conv2d(in_channel, layer_channel[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer_channel[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(layer_channel[0], layer_channel[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer_channel[0]),
            nn.ReLU(inplace=True)
        )

        # -----------------------------------------------------------------
        # [修改区域] Encoder 部分：将 OPE 替换为 NNUNet_Encoder_Block
        # -----------------------------------------------------------------
        
        # Level 0 -> Level 1 (Downsample)
        # self.encoder1_l1_local = OPE(layer_channel[0], layer_channel[1])
        self.encoder1_l1_local = NNUNet_Encoder_Block(layer_channel[0], layer_channel[1])
        self.encoder1_l1_global = global_block(layer_channel[0], global_dim[0], num_heads=num_heads[0], sr_ratio=sr_ratio[0])

        # Level 1 -> Level 2 (Downsample)
        # self.encoder1_l2_local = OPE(layer_channel[1], layer_channel[2])
        self.encoder1_l2_local = NNUNet_Encoder_Block(layer_channel[1], layer_channel[2])
        self.encoder1_l2_global = global_block(layer_channel[1], global_dim[1], num_heads=num_heads[1], sr_ratio=sr_ratio[1])

        # Level 2 -> Level 3 (Downsample)
        # self.encoder1_l3_local = OPE(layer_channel[2], layer_channel[3])
        self.encoder1_l3_local = NNUNet_Encoder_Block(layer_channel[2], layer_channel[3])
        self.encoder1_l3_global = global_block(layer_channel[2], global_dim[2], num_heads=num_heads[2], sr_ratio=sr_ratio[2])

        # Level 3 -> Level 4 (Downsample)
        # self.encoder1_l4_local = OPE(layer_channel[3], layer_channel[4])
        self.encoder1_l4_local = NNUNet_Encoder_Block(layer_channel[3], layer_channel[4])
        self.encoder1_l4_global = global_block(layer_channel[3], global_dim[3], num_heads=num_heads[3], sr_ratio=sr_ratio[3])

        # -----------------------------------------------------------------
        # Decoder 部分保持不变 (除非你想同时消融 Decoder)
        # -----------------------------------------------------------------
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
        
        # 辅助模块
        self.pos = PositionEmbeddingLearned(layer_channel[4] // 2)
        self.pam = PAM_Module(layer_channel[4])
        self.sdpa = ScaledDotProductAttention(layer_channel[4])
        
    def forward(self, x):
        # Forward 逻辑完全不需要改变，因为接口一致
        outputs = []
        x_e1_l0 = self.input_l0(x)

        x_e1_l1_local = self.encoder1_l1_local(x_e1_l0)
        x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)

        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local)
        x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)

        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local)
        x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)

        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local)
        x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

        x_l1 = x_e1_l4_local

        # Bottleneck
        x_l1_pam = self.pam(x_l1)
        x_l1_pos = self.pos(x_l1)
        x_l1 = x_l1 + x_l1_pos
        x_l1_sdpa = self.sdpa(x_l1)
        x_l1 = x_l1_sdpa + x_l1_pam
        
        x_e1_l4_local = x_l1
        
        # Decoder
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

class nnUNetTrainer_USANet2D_NO(nnUNetTrainer):
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
        model = USANet2D_NoOPE(in_channel=num_input_channels, num_classes=num_output_channels, deep_supervised=enable_deep_supervision)
        model.apply(InitWeights_He(1e-2))
        return model