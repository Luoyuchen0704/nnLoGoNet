import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
from scipy import ndimage  
from typing import Optional  
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper  
  
class SoftSkeletonLoss(nn.Module):  
    """  
    Soft-Skeleton Loss for maintaining vessel trunk connectivity  
    """  
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):  
        super().__init__()  
        self.alpha = alpha  # weight for skeleton loss  
        self.beta = beta    # weight for connectivity loss  
          
    def _soft_skeletonize(self, x: torch.Tensor, iterations: int = 10) -> torch.Tensor:  
        """  
        Differentiable skeletonization using morphological operations  
        """  
        # Ensure input is in [0, 1] range  
        x = torch.sigmoid(x)  
          
        # Create structuring element for morphological operations  
        kernel = torch.ones(1, 1, 3, 3, 3, device=x.device)  
          
        skeleton = x.clone()  
        for _ in range(iterations):  
            # Erosion  
            eroded = F.conv3d(skeleton, kernel, padding=1) > 0  
            eroded = eroded.float()  
              
            # Opening  
            opened = F.conv3d(eroded, kernel, padding=1) > 0  
            opened = opened.float()  
              
            # Update skeleton  
            skeleton = torch.clamp(skeleton - opened, 0, 1)  
              
        return skeleton  
      
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        # Apply sigmoid to predictions  
        pred_prob = torch.sigmoid(pred)  
          
        # Extract skeletons  
        pred_skeleton = self._soft_skeletonize(pred_prob)  
        target_skeleton = self._soft_skeletonize(target)  
          
        # Skeleton loss (Dice loss on skeletons)  
        intersection = (pred_skeleton * target_skeleton).sum()  
        union = pred_skeleton.sum() + target_skeleton.sum()  
        skeleton_loss = 1 - (2 * intersection + 1e-8) / (union + 1e-8)  
          
        # Connectivity loss  
        connectivity_loss = self._compute_connectivity_loss(pred_prob, target)  
          
        return self.alpha * skeleton_loss + self.beta * connectivity_loss  
      
    def _compute_connectivity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        """  
        Compute connectivity loss based on connected components  
        """  
        # Convert to binary  
        pred_binary = (pred > 0.5).float()  
        target_binary = target.float()  
          
        # Count connected components (simplified version)  
        pred_components = self._count_components(pred_binary)  
        target_components = self._count_components(target_binary)  
          
        # Penalize difference in number of components  
        component_loss = F.l1_loss(pred_components, target_components)  
          
        return component_loss  
      
    def _count_components(self, binary: torch.Tensor) -> torch.Tensor:  
        """  
        Count connected components in binary volume  
        """  
        # Simplified component counting using convolution  
        kernel = torch.ones(1, 1, 3, 3, 3, device=binary.device)  
        neighbors = F.conv3d(binary, kernel, padding=1)  
          
        # Estimate number of components (simplified)  
        components = torch.sum(binary > 0) / torch.mean(neighbors[binary > 0] + 1e-8)  
        return components.unsqueeze(0)  
  
class ConnectivityConstraintLoss(nn.Module):  
    """  
    Connectivity Constraint Loss for reducing vessel breaks and false branches  
    """  
    def __init__(self, connectivity_weight: float = 1.0, topology_weight: float = 0.5):  
        super().__init__()  
        self.connectivity_weight = connectivity_weight  
        self.topology_weight = topology_weight  
          
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        pred_prob = torch.sigmoid(pred)  
          
        # Connectivity loss  
        connectivity_loss = self._connectivity_loss(pred_prob, target)  
          
        # Topology preservation loss  
        topology_loss = self._topology_loss(pred_prob, target)  
          
        return self.connectivity_weight * connectivity_loss + self.topology_weight * topology_loss  
      
    def _connectivity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        """  
        Penalize disconnected vessel segments  
        """  
        pred_binary = (pred > 0.5).float()  
        target_binary = target.float()  
          
        # Compute Euler characteristic (simplified)  
        pred_euler = self._compute_euler_characteristic(pred_binary)  
        target_euler = self._compute_euler_characteristic(target_binary)  
          
        return F.mse_loss(pred_euler, target_euler)  
      
    def _topology_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        """  
        Preserve topological properties  
        """  
        # Persistent homology inspired loss (simplified)  
        pred_binary = (pred > 0.5).float()  
        target_binary = target.float()  
          
        # Count holes and handles  
        pred_holes = self._count_holes(pred_binary)  
        target_holes = self._count_holes(target_binary)  
          
        return F.l1_loss(pred_holes, target_holes)  
      
    def _compute_euler_characteristic(self, binary: torch.Tensor) -> torch.Tensor:  
        """  
        Compute Euler characteristic: V - E + F  
        """  
        # Simplified Euler characteristic computation  
        kernel = torch.ones(1, 1, 3, 3, 3, device=binary.device)  
          
        # Count vertices (voxels)  
        vertices = binary.sum()  
          
        # Count edges (adjacent voxel pairs)  
        edges = F.conv3d(binary, kernel, padding=1)  
        edges = (edges > 0).sum() / 2  # Divide by 2 to avoid double counting  
          
        # Simplified Euler characteristic  
        euler = vertices - edges * 0.5  
        return euler.unsqueeze(0)  
      
    def _count_holes(self, binary: torch.Tensor) -> torch.Tensor:  
        """  
        Count holes in binary volume  
        """  
        # Simplified hole counting using morphological operations  
        kernel = torch.ones(1, 1, 5, 5, 5, device=binary.device)  
          
        # Fill operation  
        filled = F.conv3d(binary, kernel, padding=2) > 0  
        filled = filled.float()  
          
        # Estimate holes  
        holes = torch.sum(filled - binary)  
        return holes.unsqueeze(0)  
  
class TreeEnergyLoss(nn.Module):  
    """  
    Tree-energy Loss for vessel tree hierarchy optimization  
    """  
    def __init__(self, tree_weight: float = 1.0, hierarchy_weight: float = 0.5):  
        super().__init__()  
        self.tree_weight = tree_weight  
        self.hierarchy_weight = hierarchy_weight  
          
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        pred_prob = torch.sigmoid(pred)  
          
        # Tree structure loss  
        tree_loss = self._tree_structure_loss(pred_prob, target)  
          
        # Hierarchy preservation loss  
        hierarchy_loss = self._hierarchy_loss(pred_prob, target)  
          
        return self.tree_weight * tree_loss + self.hierarchy_weight * hierarchy_loss  
      
    def _tree_structure_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        """  
        Preserve tree-like structure  
        """  
        pred_binary = (pred > 0.5).float()  
        target_binary = target.float()  
          
        # Compute branching points  
        pred_branches = self._detect_branching_points(pred_binary)  
        target_branches = self._detect_branching_points(target_binary)  
          
        # Branch point preservation loss  
        branch_loss = F.mse_loss(pred_branches, target_branches)  
          
        # Tree energy (simplified)  
        pred_energy = self._compute_tree_energy(pred_binary)  
        target_energy = self._compute_tree_energy(target_binary)  
          
        energy_loss = F.mse_loss(pred_energy, target_energy)  
          
        return branch_loss + energy_loss  
      
    def _hierarchy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        """  
        Preserve vessel hierarchy (main branches vs small branches)  
        """  
        # Multi-scale analysis  
        scales = [1, 3, 5]  
        hierarchy_loss = 0  
          
        for scale in scales:  
            kernel_size = 2 * scale + 1  
            kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=pred.device)  
              
            pred_smooth = F.conv3d(pred, kernel, padding=scale) / (kernel_size ** 3)  
            target_smooth = F.conv3d(target, kernel, padding=scale) / (kernel_size ** 3)  
              
            hierarchy_loss += F.mse_loss(pred_smooth, target_smooth)  
          
        return hierarchy_loss / len(scales)  
      
    def _detect_branching_points(self, binary: torch.Tensor) -> torch.Tensor:  
        """  
        Detect branching points in binary vessel tree  
        """  
        kernel = torch.ones(1, 1, 3, 3, 3, device=binary.device)  
        neighbors = F.conv3d(binary, kernel, padding=1)  
          
        # Branching points have more than 2 neighbors  
        branches = (neighbors > 2.5).float() * binary  
        return branches.sum().unsqueeze(0)  
      
    def _compute_tree_energy(self, binary: torch.Tensor) -> torch.Tensor:  
        """  
        Compute tree energy based on connectivity and branching  
        """  
        # Simplified tree energy  
        kernel = torch.ones(1, 1, 3, 3, 3, device=binary.device)  
        connectivity = F.conv3d(binary, kernel, padding=1)  
          
        # Energy based on local connectivity patterns  
        energy = torch.sum(binary * connectivity) / (torch.sum(binary) + 1e-8)  
        return energy.unsqueeze(0)  
  
# Combined vascular loss  
class VascularTreeLoss(nn.Module):  
    """  
    Combined loss for vessel segmentation with topology preservation  
    """  
    def __init__(self,   
                 skeleton_weight: float = 0.3,  
                 connectivity_weight: float = 0.3,   
                 tree_weight: float = 0.4):  
        super().__init__()  
          
        self.skeleton_loss = SoftSkeletonLoss(alpha=skeleton_weight, beta=0.5)  
        self.connectivity_loss = ConnectivityConstraintLoss(  
            connectivity_weight=connectivity_weight, topology_weight=0.5)  
        self.tree_loss = TreeEnergyLoss(tree_weight=tree_weight, hierarchy_weight=0.5)  
          
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
        skeleton_l = self.skeleton_loss(pred, target)  
        connectivity_l = self.connectivity_loss(pred, target)  
        tree_l = self.tree_loss(pred, target)  
          
        return skeleton_l + connectivity_l + tree_l