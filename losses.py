import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class CBLoss(nn.Module):
    def __init__(self, cls_num_list, beta=0.999):  # cls_num_list为每一类样本数组成的数组，beta为超参数通常设置为0.99.0.999等
        super(CBLoss, self).__init__()
        self.cls_num_list = cls_num_list
        self.beta = beta
        effective_num = 1.0 - np.power(self.beta, self.cls_num_list)
        self.weights = (1.0 - self.beta) / np.array(effective_num)
        self.weights = self.weights / np.sum(self.weights) * len(self.cls_num_list)
        self.weights = torch.cuda.FloatTensor(self.weights)

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weights) # F.cross_entropy()交叉熵损失

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class GCLLoss(nn.Module):

    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul=1., gamma=0.):
        super(GCLLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max() - m_list
        self.m_list = m_list
        assert s > 0
        self.m = m
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1 / 3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma

    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(
            cosine.device)  # self.scale(torch.randn(cosine.shape).to(cosine.device))

        # cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list
        cosine = cosine - self.noise_mul * noise.abs() / self.m_list.max() * self.m_list
        output = torch.where(index, cosine - self.m, cosine)
        if self.train_cls:
            return focal_loss(F.cross_entropy(self.s * output, target, reduction='none', weight=self.weight),
                              self.gamma)
        else:
            return F.cross_entropy(self.s * output, target, weight=self.weight)


class CBRLoss(nn.Module):

    def __init__(self, cls_num_list, weight=None, alpha=0.3, beta=0.4):
        super(CBRLoss, self).__init__()
        m_list = cls_num_list / np.max(cls_num_list)
        m_list = torch.cuda.FloatTensor(m_list)
        self.weight = weight
        self.alpha = alpha
        self.beta = beta
        self.lambda_weight = self.alpha * (m_list ** self.beta)


    def forward(self, input, target, model):
        w = model.linear.weight
        w_norm = torch.linalg.norm(w, ord=2, dim=1)
        return F.cross_entropy(input, target, weight=self.weight) + torch.matmul(self.lambda_weight, w_norm ** 2)
