'''
Define heterogeneous transform domain layers.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_HMT = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]], dtype=np.float32)
_IHMT = 4 * np.linalg.inv(_HMT)
_IHMT = _IHMT[1:3, :]
_H = torch.from_numpy(_HMT)
_IH = torch.from_numpy(_IHMT)

_HMT2 =_HMT[[1,0,2,3],:]
_H2 = torch.from_numpy(_HMT2)
_IHMT2 = 4 * np.linalg.inv(_HMT2.transpose())
_IHMT2 = _IHMT2[1:3, :]
_IH2 = torch.from_numpy(_IHMT2)

_HMT3 =_HMT[[3,1,2,0],:]
_H3 = torch.from_numpy(_HMT3)
_IHMT3 = 4 * np.linalg.inv(_HMT3.transpose())
_IHMT3 = _IHMT3[1:3, :]
_IH3 = torch.from_numpy(_IHMT3)


'''
HTNN layer with 2 transforms
'''
class MultLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, bias=True):
        super(MultLayer, self).__init__()
        self.stride = stride
        self.kernel = kernel_size 
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.padding = torch.nn.ZeroPad2d(padding=1)
        self.weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels, self.kernel, self.kernel), requires_grad=True)

        self.bias_flag = bias
        if self.bias_flag:
            self.bias = torch.nn.Parameter(data=torch.Tensor(1, out_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        batch, _, ih, iw = list(x.size())
        bsize = self.kernel - self.stride
        x = self.padding(x)

        patches = x.unfold(3, self.kernel, self.stride).unfold(2, self.kernel, self.stride)
        input_hmt1 = _H.cuda().matmul(patches).matmul(_H.cuda())
        input_hmt2 = _H2.t().cuda().matmul(patches).matmul(_H2.cuda())

        hmult1 = (input_hmt1.unsqueeze(1)*self.weight[:self.out_channels//2].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        hmult1 = _IH.cuda().matmul(hmult1).matmul(_IH.t().cuda()) # batch by n_out//2 by n_block by n_block by bsize by bsize tensor
        hmult2 = (input_hmt2.unsqueeze(1)*self.weight[self.out_channels//2:].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        hmult2 = _IH2.cuda().matmul(hmult2).matmul(_IH2.t().cuda()) # batch by n_out//2 by n_block by n_block by bsize by bsize tensor
        hmult = torch.cat([hmult1, hmult2], dim=1) # batch by n_out by n_block by n_block by kernel by kernel tensor

        hmult = hmult.permute(0,1,2,4,3,5).reshape(batch,self.out_channels,-1,bsize,iw) # batch by n_out by n_block by bsize by iw tensor
        out = hmult.reshape(batch,self.out_channels,ih,iw) # batch by n_out by ih by iw tensor

        if self.bias_flag:
            out = out + self.bias

        return out


'''
HTNN layer with 3 transforms
'''
class MultLayer3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, bias=True):
        super(MultLayer3, self).__init__()
        self.stride = stride
        self.kernel = kernel_size 
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.padding = torch.nn.ZeroPad2d(padding=1)
        self.weight = torch.nn.Parameter(data=torch.Tensor(self.out_channels, self.in_channels, self.kernel, self.kernel), requires_grad=True)

        self.bias_flag = bias
        if self.bias_flag:
            self.bias = torch.nn.Parameter(data=torch.Tensor(1, out_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        batch, _, ih, iw = list(x.size())
        bsize = self.kernel - self.stride
        x = self.padding(x)

        patches = x.unfold(3, self.kernel, self.stride).unfold(2, self.kernel, self.stride)
        input_hmt1 = _H.cuda().matmul(patches).matmul(_H.cuda())
        input_hmt2 = _H2.t().cuda().matmul(patches).matmul(_H2.cuda())
        input_hmt3 = _H3.t().cuda().matmul(patches).matmul(_H3.cuda())

        hmult1 = (input_hmt1.unsqueeze(1)*self.weight[:self.out_channels//3].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        hmult1 = _IH.cuda().matmul(hmult1).matmul(_IH.t().cuda()) # batch by n_out//3 by n_block by n_block by bsize by bsize tensor
        hmult2 = (input_hmt2.unsqueeze(1)*self.weight[self.out_channels//3:self.out_channels//3*2].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        hmult2 = _IH2.cuda().matmul(hmult2).matmul(_IH2.t().cuda()) # batch by n_out//3 by n_block by n_block by bsize by bsize tensor
        hmult3 = (input_hmt3.unsqueeze(1)*self.weight[self.out_channels//3*2:].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        hmult3 = _IH3.cuda().matmul(hmult3).matmul(_IH3.t().cuda()) # batch by n_out//3 by n_block by n_block by bsize by bsize tensor
        hmult = torch.cat([hmult1, hmult2, hmult3], dim=1) # batch by n_out by n_block by n_block by kernel by kernel tensor

        hmult = hmult.permute(0,1,2,4,3,5).reshape(batch,self.out_channels,-1,bsize,iw) # batch by n_out by n_block by bsize by iw tensor
        out = hmult.reshape(batch,self.out_channels,ih,iw) # batch by n_out by ih by iw tensor

        if self.bias_flag:
            out = out + self.bias

        return out