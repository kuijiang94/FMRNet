"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
from subpixel import shuffle_down, shuffle_up###################
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math

from einops import rearrange
from Blanced_attention import BlancedAttention, BlancedAttention_CAM_SAM_ADD


##########################################################################
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return [x_LL, x_HL, x_LH, x_HH]#torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 使用哈尔 haar 小波变换来实现二维逆向离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width]) #[1, 12, 56, 56]
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r**2)), r * in_height, r * in_width
    # print(out_batch, out_channel, out_height, out_width) #1 3 112 112
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    # print(x1.shape) #torch.Size([1, 3, 56, 56])
    # print(x2.shape) #torch.Size([1, 3, 56, 56])
    # print(x3.shape) #torch.Size([1, 3, 56, 56])
    # print(x4.shape) #torch.Size([1, 3, 56, 56])
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
##########################################################################
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
def st_conv(in_channels, out_channels, kernel_size, bias=False, stride = 2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)	
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        self.act1 = nn.PReLU()
        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act1(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        #self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, k_fea, v_fea, q_fea):
        b,c,h,w = q_fea.shape
        q = self.q(q_fea)
        k = self.k(k_fea)
        v = self.v(v_fea)
        #qkv = self.qkv_dwconv(self.qkv(x))
        #q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
        
##########################################################################
class dualAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(dualAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        #self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, k_fea, v_fea, q_fea):
        b,c,h,w = q_fea.shape
        q = self.q(q_fea)
        k = self.k(k_fea)
        v = self.v(v_fea)
        #qkv = self.qkv_dwconv(self.qkv(x))
        #q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm_key = LayerNorm(dim, LayerNorm_type)
        self.norm_query = LayerNorm(dim, LayerNorm_type)
        self.norm_value = LayerNorm(dim, LayerNorm_type)
        
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, in1, in2):
        # print('in1', in1.shape)
        # print('in2', in2.shape)
        # a = self.norm_key(in1)
        # b = self.norm_query(in2)
        # print('norm_key(in1)', a.shape)
        # print('norm_query(in2)', b.shape)
        x = in2 + self.attn(self.norm_key(in1),self.norm_value(in1),self.norm_query(in2))
        x = x + self.ffn(self.norm2(x))

        return x
        
##########################################################################
class COBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(COBlock, self).__init__()

        self.norm_key = LayerNorm(dim, LayerNorm_type)
        self.norm_query = LayerNorm(dim, LayerNorm_type)
        self.norm_value = LayerNorm(dim, LayerNorm_type)
        
        self.COattn = Attention(dim, num_heads, bias)
        #self.norm2 = LayerNorm(dim, LayerNorm_type)

    def forward(self, in1, in2):
        x = in2 + self.COattn(self.norm_key(in1),self.norm_value(in1),self.norm_query(in2))

        return x
        
        
##########################################################################
class COBlock2(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(COBlock2, self).__init__()

        self.norm_key1 = LayerNorm(dim, LayerNorm_type)
        self.norm_query1 = LayerNorm(dim, LayerNorm_type)
        self.norm_value1 = LayerNorm(dim, LayerNorm_type)
        
        self.norm_key2 = LayerNorm(dim, LayerNorm_type)
        self.norm_query2 = LayerNorm(dim, LayerNorm_type)
        self.norm_value2 = LayerNorm(dim, LayerNorm_type)
        
        self.COattn1 = Attention(dim, num_heads, bias)
        self.COattn2 = Attention(dim, num_heads, bias)
        #self.norm2 = LayerNorm(dim, LayerNorm_type)

    def forward(self, in1, in2, in3):
        x_12 = in2 + self.COattn1(self.norm_key1(in1),self.norm_value1(in1),self.norm_query1(in2))
        x_13 = in3 + self.COattn2(self.norm_key2(in1),self.norm_value2(in1),self.norm_query2(in3))
        return x_12, x_13
        
##########################################################################
class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(RestormerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
        
##########################################################################
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def st_conv(in_channels, out_channels, kernel_size, bias=False, stride = 2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)		
##########################################################################
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
		
# contrast-aware channel attention module
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
	
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
##########################################################################
## Global context Layer
# class GCLayer(nn.Module):
    # def __init__(self, channel, reduction=16, bias=False):
        # super(GCLayer, self).__init__()
        # # global average pooling: feature --> point
        # #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # # feature channel downscale and upscale --> channel weight
        # self.conv_phi = nn.Conv2d(channel, 1, 1, stride=1,padding=0, bias=False)
        # self.softmax = nn.Softmax(dim=1)
		
        # self.conv_du = nn.Sequential(
                # nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                # nn.Sigmoid()
        # )

    # def forward(self, x):
        # b, c, h, w = x.size()
        # #y = self.avg_pool(x)
        # y_1 = self.conv_phi(x).view(b, 1, -1).permute(0, 2, 1).contiguous()### b,hw,1
        # y_1_att = self.softmax(y_1)
        # print(y_1.size)
        # x_1 = x.view(b, c, -1)### b,c,hw
        # mul_context = torch.matmul(x_1, y_1_att)#### b,c,1
        # mul_context = mul_context.view(b, c, 1, -1)

        # y = self.conv_du(mul_context)
        # return x * y
		
##########################################################################
## Semantic-guidance Texture Enhancement Module
class STEM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=False):
        super(STEM, self).__init__()
        # global average pooling: feature --> point
        
        act=nn.PReLU()
        #num_blocks = 1
        heads = 4
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'

        #self.down_img = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        self.conv_stem0 = st_conv(3, n_feat, kernel_size, bias=bias)
        #self.conv_stem1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv_stem2 = conv(3, n_feat, kernel_size, bias=bias)
        self.former = TransformerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type=LayerNorm_type)
        self.conv_stem3 = conv(3, n_feat, kernel_size, bias=bias)
        self.S2FB = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CA_fea = CALayer(n_feat, reduction, bias=bias)

    def forward(self, img_rain, res, img):
        #img_down = self.down_img(img_rain)
        img_down = self.conv_stem0(img_rain)
        #img_down_fea = self.conv_stem1(img_down)
        res_fea = self.conv_stem2(res)
        #rain_mask = torch.sigmoid(res_fea)
        #rain_mask = self.CA_fea(res_fea)
        #att_fea = img_down * rain_mask + img_down
        att_fea = self.former(res_fea, img_down, img_down)
        img_fea = self.conv_stem3(img)
        S2FB2_FEA = self.S2FB(img_fea, att_fea)
        return S2FB2_FEA
        #return torch.cat([img_down_fea * rain_mask, img_fea],1) 
		
class STEM_att(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=False):
        super(STEM_att, self).__init__()
        # global average pooling: feature --> point
        
        act=nn.PReLU()
        #num_blocks = 1
        heads = 4
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'

        #self.down_img = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        self.conv_stem0 = conv(n_feat, n_feat//2, kernel_size=1, bias=bias)
        #self.conv_stem1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv_stem2 = conv(n_feat, n_feat//2, kernel_size=1, bias=bias)
        self.former = TransformerBlock(dim=n_feat//2, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type=LayerNorm_type)
        # self.conv_stem3 = conv(n_feat, n_feat//2, kernel_size, bias=bias)
        # self.S2FB = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        self.conv_stem3 = conv(n_feat//2, n_feat, kernel_size=1, bias=bias)
        
    def forward(self, img_rain, res):
        #img_down = self.down_img(img_rain)
        img_down = self.conv_stem0(img_rain)
        #img_down_fea = self.conv_stem1(img_down)
        res_fea = self.conv_stem2(res)
        #rain_mask = torch.sigmoid(res_fea)
        #rain_mask = self.CA_fea(res_fea)
        #att_fea = img_down * rain_mask + img_down
        att_fea = self.conv_stem3(self.former(res_fea, img_down))
        # img_fea = self.conv_stem3(img)
        # S2FB2_FEA = self.S2FB(img_fea, att_fea)
        return att_fea
        #return torch.cat([img_down_fea * rain_mask, img_fea],1) 
##########################################################################
## S2FB
class S2FB_2(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_2, self).__init__()
        
        # # 设置可学习参数
        # self.fuse_weight_ATOB = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # # self.fuse_weight_RTOB = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # # 初始化
        # self.fuse_weight_ATOB.data.fill_(0.2)
        # # self.fuse_weight_RTOB.data.fill_(0.2)
        # self.conv_fuse_ATOB = nn.Sequential(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=False), nn.Sigmoid())
        
        # self.DSC = depthwise_separable_conv(n_feat, n_feat)
        # # self.DSC1 = depthwise_separable_conv(n_feat//2, n_feat)
        # #self.CON_FEA = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        # self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        # #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        # #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    # def forward(self, x1, x2):
        # FEA_1to2 = self.DSC(x1*self.conv_fuse_ATOB(x2)*self.fuse_weight_ATOB)
        # #FEA_2 = self.CON_FEA(torch.cat((x1,x2), 1))
        # #resin = FEA_1 + FEA_2
        # out= self.CA_fea(FEA_1to2) + x2
        # #res += resin
        # return out#x1 + resin
        
        
        self.DSC = depthwise_separable_conv(n_feat*2, n_feat)
        # self.DSC1 = depthwise_separable_conv(n_feat//2, n_feat)
        #self.CON_FEA = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2):
        FEA_1 = self.DSC(torch.cat((x1,x2), 1))
        #FEA_2 = self.CON_FEA(torch.cat((x1,x2), 1))
        #resin = FEA_1 + FEA_2
        res= self.CA_fea(FEA_1) + x1
        #res += resin
        return res#x1 + resin

##########################################################################
## S2FB
class S2FB_4(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_4, self).__init__()
        #self.CA_fea1 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea2 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea3 = CALayer(n_feat, reduction, bias=bias)
        self.DSC = depthwise_separable_conv(n_feat*4, n_feat)
        #self.CON_FEA = nn.Conv2d(n_feat*3, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2, x3, x4):
        FEA_1 = self.DSC(torch.cat((x1, x2,x3,x4), 1))
        #FEA_2 = self.CON_FEA(torch.cat((x2,x3,x4), 1))
        #resin = FEA_1 + FEA_2
        res= self.CA_fea(FEA_1) + FEA_1
        #res += resin
        #res1= self.CA_fea1(FEA_1)
        #res2= self.CA_fea2(x3)
        #res3= self.CA_fea3(x4)
        return res#x1 + res

class S2FB_p(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_p, self).__init__()
        #self.CA_fea1 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea2 = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea3 = CALayer(n_feat, reduction, bias=bias)
        self.DSC1 = depthwise_separable_conv(n_feat*2, n_feat)
        self.DSC2 = depthwise_separable_conv(n_feat*2, n_feat)
        self.DSC3 = depthwise_separable_conv(n_feat*2, n_feat)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2, x3, x4):
        FEA_34 = self.DSC1(torch.cat((x3, x4), 1))
        FEA_34_2 = self.DSC2(torch.cat((x2, FEA_34), 1))
        FEA_34_2_1 = self.DSC3(torch.cat((x1, FEA_34_2), 1))
        res= self.CA_fea(FEA_34_2_1) + FEA_34_2_1
        #res += resin
        #res1= self.CA_fea1(FEA_1)
        #res2= self.CA_fea2(x3)
        #res3= self.CA_fea3(x4)
        return res#x1 + res
##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
		
## Enhanced Channel Attention Block (ECAB)
class ECAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ECAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res
		
## Channel Attention Block (CAB)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = []
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))
        modules_body.append(act)
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res


class DownSample(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x
        
class UpSample(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
        
##########################################################################
## Coupled Representaion Module (CRM)
class CRM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias):
        super(CRM, self).__init__()
        #num_blocks = num_cab
        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'
        #modules_body = []
        #self.CAB = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        #self.down1 = DownSample(n_feat)
        #self.down2 = DownSample(n_feat)
        self.CAB1 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.CAB2 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.CAB3 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.CAB4 = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.CAB5 = nn.Sequential(conv(n_feat*2, n_feat//2, kernel_size, bias=bias), CAB_dsc(n_feat//2, kernel_size, reduction, bias=bias, act=act), conv(n_feat//2, n_feat, kernel_size, bias=bias))#, CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act))
        #self.former = TransformerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type=LayerNorm_type)
        # self.coblock_1 = COBlock(dim=n_feat, num_heads=heads, bias=False, LayerNorm_type=LayerNorm_type)
        # self.coblock_2 = COBlock(dim=n_feat, num_heads=heads, bias=False, LayerNorm_type=LayerNorm_type)
        # self.coblock2 = COBlock2(dim=n_feat, num_heads=heads, bias=False, LayerNorm_type=LayerNorm_type)
        self.up1 = UpSample(n_feat)
        self.up2 = UpSample(n_feat)
        self.STEM_att12 = STEM_att(n_feat, kernel_size, bias=bias)
        self.STEM_att21 = STEM_att(n_feat, kernel_size, bias=bias)
        self.S2FB2_1 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        self.S2FB2_2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        self.S2FB2_3 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        self.S2FB2_4 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.S2FB2_3 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.conv = conv(n_feat, n_feat, kernel_size)
        #self.body = nn.Sequential(*modules_body) S2FB_2

    def forward(self, x):
        x0 = x[0]  ### ll
        x1 = x[1]  ### rain
        x2 = x[2]  ### back
        #x1_down = self.down1(x1)
        #x2_down = self.down2(x2)
        # res01, res02 = self.coblock2(x0, x1, x2)
		
        res01  = self.S2FB2_1(x0, x1) ### ll rain
        res02 = self.S2FB2_2(x0, x2)  ### ll rain
        #res13 = self.COBlock(x1, x2)
        x1v1 = self.CAB1(res01)  + x1 ### rain
        x2v1 = self.CAB2(res02)  + x2 ### back
        # print('x2v1', x2v1.shape)
        res12 = self.STEM_att12(x1v1, x2v1)
        # print('res12', res12.shape)
        x2v2 = self.CAB3(res12) + x2v1 ### back
        res21 = self.STEM_att21(x2v2, x1v1)
        x1v2 = self.CAB4(res21) + x1v1 ### rain
        
        res10 = self.CAB5(torch.cat([x1v2, x0],1) )
        x0v1 = res10 + x0
        #res = self.coblock(x2v1, x1v1)
        #x1v2 = res + x1v1
        x[0] = x0v1 #+ x1 #self.up1(x1v2) + x1
        x[1] = x1v2 #+ x2 #self.up2(x2v1) + x2
        x[2] = x2v2 #+ x2 #self.up2(x2v1) + x2
        return x

##########################################################################
class COmodule(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(COmodule, self).__init__()
        modules_body = []
        modules_body = [CRM(n_feat, kernel_size, reduction, act, bias) for i in range(num_cab)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x1, x2, x3):
        res = self.body([x1, x2, x3])
        return res
        
##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        num_blocks = 5
        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        #self.modules_body = nn.Sequential(*[RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])

        #self.conv = conv(n_feat, n_feat, kernel_size)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        #res = self.conv(res)
        res += x
        return res
##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORSNet, self).__init__()

        self.orb = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)

    def forward(self, x):
        x = self.orb(x)
        return x
##########################################################################
## Reconstruction and Reproduction Block (RRB)
class RRB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(RRB, self).__init__()
        self.iwt_rain = IWT()
        self.iwt_back = IWT()
        self.iwt_res = IWT()
        self.recon_B =  conv(12, 12, kernel_size, bias=bias)
        self.recon_R = conv(12, 12, kernel_size, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(16, 32, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 12, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )
    def forward(self, x):
        xB = x[0]
        xR = x[1]

        recon_B = self.recon_B(xB)
        recon_R = self.recon_R(xR)
        # res = self.avg_pool(recon_B + recon_R)
        # res_att = self.conv_du(res)
        # re_rain = xB*res_att + xR*(1-res_att)
        #rain_img = self.iwt_rain(re_rain)
        #back_img = self.iwt_back(xB)
        # rain_img = self.iwt_rain(re_rain)
        rain_res = self.iwt_res(recon_R)
        back_img = self.iwt_back(xB)
        return [back_img, rain_res]
        
        
##########################################################################
class DSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_orb, num_cab):
        super(DSNet, self).__init__()
        
        self.dwt = DWT()
        #self.iwt_rain = IWT()
        #self.iwt_back = IWT()
        act=nn.GELU()#nn.ReLU()
        #num_blocks = 1
        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'
        
        # self.down_1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        # self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        # self.shallow_feat2 = nn.Sequential(conv(3*4, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        # self.fuse_conv  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.former = nn.Sequential(CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act), RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type), CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act))
        #self.down_R = st_conv(n_feat, n_feat*2, kernel_size, bias=bias)
        #self.down_B = st_conv(n_feat, n_feat*2, kernel_size, bias=bias)
        #self.orsnet = ORSNet(n_feat, kernel_size, reduction, act, bias, num_orb)
        self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))#, RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        # self.shallow_feat1_1 = conv(n_feat//2, n_feat, kernel_size, bias=bias)#, RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        # self.shallow_feat2_1 = nn.Sequential(conv(3, n_feat//2, kernel_size, bias=bias), CAB_dsc(n_feat//2,kernel_size, reduction, bias=bias, act=act))
        # self.shallow_feat2_2 = nn.Sequential(conv(3, n_feat//2, kernel_size, bias=bias), CAB_dsc(n_feat//2,kernel_size, reduction, bias=bias, act=act))
        # self.shallow_feat2_3 = nn.Sequential(conv(3, n_feat//2, kernel_size, bias=bias), CAB_dsc(n_feat//2,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(4*3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        
        self.shallow_feat_R = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.shallow_feat_B = conv(n_feat, n_feat, kernel_size, bias=bias)
        #self.fuse_conv  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        #self.former = nn.Sequential(*[RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        
        self.comodule = COmodule(n_feat, kernel_size, reduction, act, bias, num_cab)
        #self.UP_B = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat,  kernel_size=3, stride=2, padding=1,output_padding=1, bias= False), act)#, conv(n_feat, n_feat, 1, bias=bias))
        #self.UP_R = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat,  kernel_size=3, stride=2, padding=1,output_padding=1, bias= False), act)#, conv(n_feat, n_feat, 1, bias=bias))

        #self.tail_LL_r = conv(n_feat, 3, kernel_size, bias=bias)
        
        self.cat_rain = nn.Sequential(CAB_dsc(n_feat*2, kernel_size, reduction, bias=bias, act=act), conv(n_feat*2, 3*4, kernel_size, bias=bias)) 
        self.cat_back = nn.Sequential(CAB_dsc(n_feat*2, kernel_size, reduction, bias=bias, act=act), conv(n_feat*2, 3*4, kernel_size, bias=bias)) 
        
        #conv(n_feat*2, 3*4, kernel_size, bias=bias)
        #self.tail_LL_r = conv(n_feat*2, 3*4, kernel_size, bias=bias)
        #self.tail_HH_b = conv(n_feat, 3, kernel_size, bias=bias)
        #self.tail_V_b = conv(n_feat, 3, kernel_size, bias=bias)
        #self.tail_H_b = conv(n_feat, 3, kernel_size, bias=bias)
        
        #self.tail_HH_r = conv(n_feat, 3, kernel_size, bias=bias)
        #self.tail_V_r = conv(n_feat, 3, kernel_size, bias=bias)
        #self.tail_H_r = conv(n_feat, 3, kernel_size, bias=bias)
        
        #self.S2FB_fuse = S2FB_2(n_feat, reduction, bias=bias, act=act)
        self.recon = RRB(n_feat, kernel_size, act, bias=bias)
        #self.tail     = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x):

        #H = x.size(2)
        #W = x.size(3)
        dwt_fea = self.dwt(x)
        
        x_LL = dwt_fea[0]
        x_V = dwt_fea[1]
        x_H = dwt_fea[2]
        x_HH = dwt_fea[3]
        # Two Patches for Stage 2
        #xtop_img  = x[:,:,0:int(H/2),:]
        #xbot_img  = x[:,:,int(H/2):H,:]
		
        #x2_img_down = self.down_1(x)
        #x2_img_down_fea = self.shallow_feat1(x2_img_down)
		
        # Four Patches for Stage 1
        # x1ltop_img = xtop_img[:,:,:,0:int(W/2)]
        # x1rtop_img = xtop_img[:,:,:,int(W/2):W]
        # x1lbot_img = xbot_img[:,:,:,0:int(W/2)]
        # x1rbot_img = xbot_img[:,:,:,int(W/2):W]
        
        # stage1_input = torch.cat([x1ltop_img, x1rtop_img, x1lbot_img, x1rbot_img],1) 
        # x1fea = self.shallow_feat2(stage1_input)
        # former_fea = self.former(x1fea)
        
        
        # stage1_fuse = torch.cat([x2_img_down_fea, former_fea],1) 
        # fuse_fea = self.fuse_conv(stage1_fuse)
        
        x_LL_fea = self.shallow_feat1(x_LL)
        # x_LL_fea = self.shallow_feat1_1(x_LL_fea)
        #LL_fea = self.orsnet(x_LL_fea)

        # x_V_fea = self.shallow_feat2_1(x_V) 
        # x_H_fea = self.shallow_feat2_2(x_H) 
        # x_HH_fea = self.shallow_feat2_3(x_HH) 
        stage_fuse = torch.cat([x_LL, x_V, x_H, x_HH],1)

        x_fuse_fea = self.shallow_feat2(stage_fuse)
        
        former_fea = self.former(x_fuse_fea)
        #R_down = self.down_R(former_fea)
        #B_down = self.down_B(former_fea)
        
        xB_fea = self.shallow_feat_B(former_fea)
        xR_fea = self.shallow_feat_R(former_fea)

        [x_LL_R, rain_fea, back_fea] = self.comodule(x_LL_fea, xR_fea, xB_fea)
        
        #x_V_rain = self.tail_V_r(rain_fea)
        #x_HH_rain = self.tail_HH_r(rain_fea)
        #x_H_rain = self.tail_H_r(rain_fea)
        #x_LL_rain = self.tail_LL_r(LL_fea)
        
        b_cat = self.cat_back(torch.cat((x_LL_fea-x_LL_R, back_fea), 1))
        r_cat = self.cat_rain(torch.cat((x_LL_R, rain_fea), 1))
        
        #x_V_b = self.tail_V_b(back_fea)
        #x_HH_b = self.tail_HH_b(back_fea)
        #x_H_b = self.tail_H_b(back_fea)
        #fea_B = self.UP_B(back_fea)
        #fea_R = self.UP_R(rain_fea)
        #fused_fea = self.S2FB_fuse(xGE_fea, xLE_fea)
        #fused_fea = self.S2FB_fuse(or_fea, fused_fea)
        #b_cat = torch.cat((x_LL-x_LL_rain, x_V_b, x_H_b, x_HH_b), 1)
        #r_cat = torch.cat((x_LL_rain, x_V_rain, x_H_rain, x_HH_rain), 1)
        
        #rain_img = self.iwt_rain(r_cat)
        #back_img = self.iwt_back(b_cat)
        
        [img_B, img_R] = self.recon([b_cat, r_cat])
        #recon_up  = shuffle_up(fused_fea, 2)

        #out = self.tail(fused_fea)
		
        return img_B, img_R
		
##########################################################################
class COformer(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=48, kernel_size=3, reduction=4, num_orb=3, num_cab=10, bias=False):
        super(COformer, self).__init__()

        act=nn.PReLU()
        self.dsnet = DSNet(n_feat, kernel_size, reduction, act, bias, num_orb, num_cab)

        
    def forward(self, x_img): #####b,c,h,w
        #print(x_img.shape)

        imitation, rain_res = self.dsnet(x_img)
        # print("x_img",x_img.device)
        # print("rain_res",rain_res.device)
        # print("imitation",imitation.device)
        #imitation = x_img - res
        return [imitation, rain_res]