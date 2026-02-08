import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class SoftSkeletonize(torch.nn.Module):  
    def __init__(self, num_iter=40):  
        super(SoftSkeletonize, self).__init__()  
        self.num_iter = num_iter  
  
    def soft_erode(self, img):  
        if len(img.shape)==4:  
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))  
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))  
            return torch.min(p1,p2)  
        elif len(img.shape)==5:  
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))  
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))  
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))  
            return torch.min(torch.min(p1, p2), p3)  
  
    def soft_dilate(self, img):  
        if len(img.shape)==4:  
            return F.max_pool2d(img, (3,3), (1,1), (1,1))  
        elif len(img.shape)==5:  
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))  
  
    def soft_open(self, img):  
        return self.soft_dilate(self.soft_erode(img))  
  
    def soft_skel(self, img):  
        img1 = self.soft_open(img)  
        skel = F.relu(img-img1)  
        for j in range(self.num_iter):  
            img = self.soft_erode(img)  
            img1 = self.soft_open(img)  
            delta = F.relu(img-img1)  
            skel = skel + F.relu(delta - skel * delta)  
        return skel  
  
    def forward(self, img):  
        # return self.soft_skel(img)
        # 转换 float
        return self.soft_skel(img.float())  
  
class soft_cldice(nn.Module):  
    def __init__(self, iter_=3, smooth=1., exclude_background=False):  
        super(soft_cldice, self).__init__()  
        self.iter = iter_  
        self.smooth = smooth  
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)  
        self.exclude_background = exclude_background  
  
    def forward(self, y_true, y_pred):  
        if self.exclude_background:  
            y_true = y_true[:, 1:, :, :]  
            y_pred = y_pred[:, 1:, :, :]  
        skel_pred = self.soft_skeletonize(y_pred)  
        skel_true = self.soft_skeletonize(y_true)  
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)      
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)      
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)  
        return cl_dice  
  
def soft_dice(y_true, y_pred):  
    smooth = 1  
    intersection = torch.sum((y_true * y_pred))  
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)  
    return (1. - coeff)  
  
class soft_dice_cldice(nn.Module):  
    def __init__(self, iter_=3, alpha=0.5, smooth=1., exclude_background=False, ignore_label=None):  
        super(soft_dice_cldice, self).__init__()  
        self.iter = iter_  
        self.smooth = smooth  
        self.alpha = alpha  
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)  
        self.exclude_background = exclude_background
        self.ignore_label = ignore_label   
  
    def forward(self, y_true, y_pred): 

        if self.exclude_background:  
            y_true = y_true[:, 1:, :, :]  
            y_pred = y_pred[:, 1:, :, :]  
        # 处理忽略标签
        if self.ignore_label is not None:  
            mask = y_true != self.ignore_label  
            y_true = torch.where(mask, y_true, 0)  
        # 激活函数
        y_pred_prob = torch.sigmoid(y_pred)  
        y_true_prob = y_true.float()  
        
        dice = soft_dice(y_true_prob, y_pred_prob)  
        skel_pred = self.soft_skeletonize(y_pred_prob)  
        skel_true = self.soft_skeletonize(y_true_prob)  
        tprec = (torch.sum(torch.multiply(skel_pred, y_true_prob))+self.smooth)/(torch.sum(skel_pred)+self.smooth)      
        tsens = (torch.sum(torch.multiply(skel_true, y_pred_prob))+self.smooth)/(torch.sum(skel_true)+self.smooth)      
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)  
        return (1.0-self.alpha)*dice+self.alpha*cl_dice