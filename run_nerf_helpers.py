import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2) # 计算均方差
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])) # 计算psnr，越大越好
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8) # 归一化图像转为uint8


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs # 编码配置参数
        self.create_embedding_fn() # 创建编码函数
        
    def create_embedding_fn(self):
        # 所有编码函数
        embed_fns = []
        # 输入维度
        d = self.kwargs['input_dims']
        # 输出维度
        out_dim = 0
        # 如果保存输入, 则append返回原始输入的函数，输出维度+3
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
        
        # 最大编码阶数
        max_freq = self.kwargs['max_freq_log2']
        # 编码数量
        N_freqs = self.kwargs['num_freqs']
        
        # 如果使用对数采样，在0-max_freq取N_freqs个数据并作为2的指数计算出编码频率
        # 否则，在2.**0.-2.**max_freq取N_freqs个数据
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        # 遍历freq_bands和周期函数，生成一系列编码函数[sin(freq*x),cos(freq*x),...], 同时更新输出维度
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    # 对坐标进行编码(在最后一个维度进行拼接)
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# 获取位置编码函数和通道，对应paper中的
# [sin(2**0*pi*p),cos(2**0*pi*p), ... sin(2**(L-1)*pi*p), cos(2**(L-1)*pi*p)]
# return: (1) 位置编码函数，(2) 位置编码后的输出通道数(L * 2 * 3 + 3=6L+3)
def get_embedder(multires, i=0):
    # 如果i==-1, 直接返回原始坐标，即不进行编码
    if i == -1:
        return nn.Identity(), 3
    # 编码配置参数
    embed_kwargs = {
                'include_input' : True, # 是否包含输入，即编码后的数据包含原始3维坐标
                'input_dims' : 3, # 输入维度，点坐标(x,y,z)或方向(dx,dy,dz)
                'max_freq_log2' : multires-1, # 最大编码阶数
                'num_freqs' : multires, # 编码数量
                'log_sampling' : True, # 是否对数采样
                'periodic_fns' : [torch.sin, torch.cos], # 周期函数
    }
    # 返回编码函数和输出维度
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D # 模型深度
        self.W = W # 模型宽度
        self.input_ch = input_ch # 点位置输入通道数
        self.input_ch_views = input_ch_views # 视角方向输入通道
        self.skips = skips # 在上一层输出的基础上加入原始坐标输入的网络层
        self.use_viewdirs = use_viewdirs # 是否使用视角方向参与训练
        
        # 对应paper中的连续8层FC
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # 视角方向对应的FC层
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            # 输出256 feature map 的FC层
            self.feature_linear = nn.Linear(W, W)
            # 输出体密度的FC层
            self.alpha_linear = nn.Linear(W, 1)
            # 输出RGB的FC层
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            # 不使用视角方向时，输出5维：[rgb, sigma, other]
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        # 输入x是经过编码的点和方向坐标，按照最后一维将点和方向分开
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        # 遍历8层FC网络，进行前向传播计算
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # 对于skip层，在上一层输出的基础上加入原始坐标作为输入
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # 使用方向向量时，计算feature map, 体密度和rgb
        if self.use_viewdirs:
            # 计算体密度sigma
            alpha = self.alpha_linear(h)
            # 计算feature map
            feature = self.feature_linear(h)
            # 将feature map和方向拼接做为下一层输入
            h = torch.cat([feature, input_views], -1)

            # 计算加入方向向量后的输出
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            # 计算rgb
            rgb = self.rgb_linear(h)
            # 拼接输出结果
            outputs = torch.cat([rgb, alpha], -1)
        else:
            # 不使用方向向量时的输出
            outputs = self.output_linear(h)

        return outputs    

    # 从keras加载权重(没用到)
    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    # 类似get_rays_np函数功能
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    # 根据图像宽高计算xy mesh坐标，i是x, j是y, shape: [H, W]
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # 计算方向，注意：这里相机坐标系定义是x向右，y向上，z向后(即相机看向坐标轴的负z方向)，所以将像素坐标转为归一化平面坐标时，y和z都加了负号
    # 但这里dir的norm不是1，仅仅是转到了归一化平面坐标, dirs shape: [H, W, 3]
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # 将dir旋转到world坐标系, rays_d shape: [H, W, 3]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 将相机在world系下的位置rays_o维度扩展，与rays_d shape相同, rays_o shape: [H, W, 3]
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    # 返回rays_o和rays_d, shape: [H, W, 3]
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
