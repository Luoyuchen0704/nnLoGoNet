import numpy as np  
import torch  
import torch.nn as nn
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper  
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  
from nnunetv2.training.loss.cldice_loss import soft_cldice, soft_dice_cldice  
from nnunetv2.utilities.helpers import softmax_helper_dim1  
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

class nnUNetTrainerCLDice(nnUNetTrainer):  
    """使用clDice损失的训练器"""  
      
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,  
                 device: torch.device = torch.device('cuda')):  
        super().__init__(plans, configuration, fold, dataset_json, device)  
  
        self.alpha = 0.1
        # self.alpha = 1
        self.num_epochs = 100
      
    def _build_loss(self):  
        assert not self.label_manager.has_regions, "regions not supported by this trainer" 
         
        # dice + ce loss
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
            
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
                
        cld_loss = soft_cldice(  
            iter_=3,    # 骨架
            smooth=1e-5,
            exclude_background=True
        )  
        
        # 公式：L = (1-alpha) * (Dice+CE) + alpha * clDice
        # todo:
        loss = loss + self.alpha * cld_loss
        
          
        if self.enable_deep_supervision:  
            deep_supervision_scales = self._get_deep_supervision_scales()  
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:  
                weights[-1] = 0  
            weights = weights / weights.sum()  
            loss = DeepSupervisionWrapper(loss, weights)  
          
        return loss  
 
class Combined_DC_CE_and_clDice_loss(nn.Module):
    def __init__(self, dc_ce_loss, cldice_loss, alpha):
        super().__init__()
        self.dc_ce_loss = dc_ce_loss
        self.cldice_loss = cldice_loss
        self.alpha = alpha

    def forward(self, net_output, target):
        # 1. 计算原本的 Dice + CE Loss
        # DC_and_CE_loss 内部会自动处理 net_output 的 softmax/sigmoid
        dc_ce_res = self.dc_ce_loss(net_output, target)

        # 2. 计算 clDice Loss
        # 注意：需要手动处理数据格式以适配 clDice (soft_cldice 通常需要概率图和One-hot GT)
        # 这里假设 soft_cldice 内部已经处理好了，或者我们在 forward 里处理
        # 为保险起见，建议在这里进行预处理（参考之前的对话）
        
        # 预处理：Logits -> Probabilities
        from nnunetv2.utilities.helpers import softmax_helper_dim1
        probs = softmax_helper_dim1(net_output)
        
        # 预处理：Target -> One-Hot (因为 clDice 需要每个通道的骨架)
        # 且需要处理 ignore_label
        with torch.no_grad():
            target_one_hot = torch.zeros_like(net_output)
            # 假设 target 是 (B, 1, X, Y, Z)
            # 如果有 ignore_label，需要先掩膜掉
            if self.dc_ce_loss.ignore_label is not None:
                mask = target != self.dc_ce_loss.ignore_label
                target_valid = torch.where(mask, target, torch.tensor(0, device=target.device, dtype=target.dtype))
                target_one_hot.scatter_(1, target_valid.long(), 1)
                target_one_hot = target_one_hot * mask.float()
            else:
                target_one_hot.scatter_(1, target.long(), 1)
            
            # 转换类型避免报错
            target_one_hot = target_one_hot.float()
            probs = probs.float()

        cld_res = self.cldice_loss(target_one_hot, probs)

        # 3. 融合 (注意公式)
        result = dc_ce_res + self.alpha * cld_res
        
        return result

class DC_CE_and_clDice_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cldice_kwargs, 
                 weight_ce=1, weight_dice=1, weight_cldice=1, 
                 ignore_label=None, dice_class=SoftDiceLoss):
        """
        三项融合损失：CE + Dice + clDice
        :param cldice_kwargs: 传递给 soft_cldice 的参数，例如 {'iter_': 3, 'smooth': 1e-5}
        """
        super(DC_CE_and_clDice_loss, self).__init__()
        
        # 1. 初始化参数
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.ignore_label = ignore_label

        # 2. 初始化子 Loss 模块
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        
        # 初始化 clDice (确保传入了正确的参数)
        # 注意：这里不需要传入 alpha，因为权重在外部控制
        self.cldice = soft_cldice(**cldice_kwargs) 

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        :param net_output: (B, C, X, Y, Z) Logits
        :param target: (B, 1, X, Y, Z) Integer labels
        """
        
        # --- 预处理：处理 ignore_label (保持 nnUNet 原生逻辑) ---
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label not implemented for one hot encoded target'
            mask = target != self.ignore_label
            # 将 ignore 区域替换为 0，防止索引越界或计算错误
            # 这些区域在 CE 中会被 ignore_index 忽略，在 Dice 中会被 loss_mask 忽略
            target_dice = torch.where(mask, target, torch.tensor(0, device=target.device, dtype=target.dtype))
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # --- 1. 计算 Dice Loss (使用 Logits, 内部自动 Softmax) ---
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        # --- 2. 计算 CE Loss (使用 Logits, 内部自动处理) ---
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        
        # --- 3. 计算 clDice Loss (需要手动处理输入) ---
        cldice_loss = 0
        if self.weight_cldice != 0:
            # A. Logits -> Probabilities (clDice 需要 0-1 范围的值来软骨架化) [cite: 140, 155]
            # softmax_helper_dim1 自动识别是 softmax 还是 sigmoid
            probs = softmax_helper_dim1(net_output)
            
            # B. Target Integer -> One-Hot Encoding
            # clDice 需要 (B, C, X, Y, Z) 形状的 GT 来分别提取每个类别的骨架
            num_classes = net_output.shape[1]
            
            # 创建 One-Hot 编码 (注意要使用处理过 ignore label 的 target_dice)
            target_one_hot = torch.zeros_like(net_output)
            target_one_hot.scatter_(1, target_dice.long(), 1)
            
            # C. 强制转为 float (解决之前的 RuntimeError: not implemented for 'Short')
            target_one_hot = target_one_hot.float()
            probs = probs.float()
            
            # D. 如果有 Mask，需要将 Mask 应用到 One-Hot 上 (将 ignore 区域设为 0)
            if mask is not None:
                # mask shape: (B, 1, X, Y, Z) -> 广播到 One-Hot
                target_one_hot = target_one_hot * mask.float()

            # E. 计算 clDice
            # 注意：soft_cldice 内部通常会处理多通道求和/平均
            cldice_loss = self.cldice(target_one_hot, probs)

        # --- 4. 融合 ---
        result = self.weight_ce * ce_loss + \
                 self.weight_dice * dc_loss + \
                 self.weight_cldice * cldice_loss
                 
        return result