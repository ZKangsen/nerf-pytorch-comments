import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

# 训练设备：GPU或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机种子为0（固定值产生相同随机序列）
np.random.seed(0)
# 用于测试
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

# 用来创建nerf的MLP网络
def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # 获取位置编码函数和通道数，对应paper中的
    # [sin(2**0*pi*p),cos(2**0*pi*p), ... sin(2**(L-1)*pi*p), cos(2**(L-1)*pi*p)]
    # multires就是L，i_embed为0时使用默认位置编码，为1时不使用位置编码
    # embed_fn:位置编码函数; input_ch: 位置编码后的输出通道数(L * 2 * 3 + 3=6L+3)
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    # 如果使用视角方向，则获取方向编码函数和通道数，类似embed_fn, input_ch
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    # 如果N_importance>0(即精网络采样点数>0)，则设置输出通道为5(rgb+sigma+权重/置信度之类的)，否则设为4(rgb+sigma)
    output_ch = 5 if args.N_importance > 0 else 4
    # skip=4表示将MLP第5层网络的输出和原始位置编码连接后, 作为第6层的输入
    skips = [4]
    # 创建nerf的MLP网络，并放入device
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    # 获取模型参数并转为list
    grad_vars = list(model.parameters())

    # 如果N_importance>0, 创建精网络，类似上面model
    # paper中说的两个网络，一个coarse(model)，一个fine(model_fine)
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # 运行网络的查询函数，实际调用的run_network
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # 创建Adam优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0 # 训练起始id
    basedir = args.basedir # log路径
    expname = args.expname # 实验名称

    ##########################

    # 加载之前训练时保存的checkpoints，用于断点处继续训练
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        # 加载checkpoint
        ckpt = torch.load(ckpt_path)
        # 读取global_step更新start
        start = ckpt['global_step']
        # 加载优化参数
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # 加载MLP模型
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################
    # 将所需数据组织为字典
    # 训练时的渲染参数
    render_kwargs_train = {
        'network_query_fn' : network_query_fn, # 网络查询函数
        'perturb' : args.perturb, # 分层随机采样
        'N_importance' : args.N_importance, # 精网络的采样点数(或者叫重要性采样)
        'network_fine' : model_fine,  # 精网络
        'N_samples' : args.N_samples, # 粗网络采样点数
        'network_fn' : model, # MLP模型
        'use_viewdirs' : args.use_viewdirs, # 使用视角方向
        'white_bkgd' : args.white_bkgd, # 白色背景(针对blender合成数据的)
        'raw_noise_std' : args.raw_noise_std, # 噪声标准差，paper中为体密度加的高斯噪声，提升训练效果
    }

    # NDC坐标只用于llff
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp # 是否使用逆深度线性采样
    # 测试时的渲染参数
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False    # 不再进行分层采样
    render_kwargs_test['raw_noise_std'] = 0. # 不加噪声

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    # 配置文件，parser会读取该文件中的参数，然后赋值给下面的参数
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    # 实验名字
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    # log路径，保存训练日志和网络参数
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    # 输入数据路径
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # --- 训练配置参数 ---
    # 粗MLP网络深度，默认：8
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    # 粗MLP网络宽度，默认：256
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    # 精MLP网络深度，默认：8
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    # 精MLP网络宽度，默认：256
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    # 随机采样射线的batch size
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    # 学习率
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    # 学习率衰减参数
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    # 并行处理的射线数量
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    # 网络中并行处理的射线采样点的数量
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # 不进行批量化，即一次只在一张图像中随机采样射线, 默认：false
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    # 不重载保存的网络权重，默认：false
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    # 为粗MLP网络加载特定权重npy文件
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # --- 渲染配置参数 ---
    # 粗网络每个射线上采样点数量，默认：64
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    # 精网络每个射线上采样点数量，默认：0
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    # 用于分层随机采样，设为0: 均匀采样，设为1: 分层随机采样
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 使用视角方向参与训练，组成5D输入，默认：false
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    # 编码参数，设为0: 默认编码，设为1: 无编码
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    # 用于3D位置编码，multires=log2(max_freq)，即2的指数
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    # 用于2D视角方向编码，multires_views=log2(max_freq)，即2的指数
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    # 噪声标准差，用于sigma_a(体密度)的正则化，提高训练稳定性
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    # 加载已有权重，只进行渲染，不进行优化，默认：false
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    # 渲染测试集，默认：false
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    # 降采样加速渲染
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # 其他训练配置参数
    # 图像中心区域训练步数
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    # 图像中心区域所占比例
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # --- 数据配置参数 ---
    # 数据类型：llff / blender / deepvoxels
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    # 从测试/验证集中仅用1/N个图像去测试
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # 物体形状：用于deepvoxels数据
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    # 白色背景：用于blender数据集，blender合成物体的背景是白色，默认：false
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    # 加载blender数据时降采样为一半，默认：false
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff配置参数
    # llff图像降采样因子 
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    # 不使用NDC坐标（适用于非正对拍摄场景），默认：false
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    # 视差线性采样（即使用逆深度，而不是深度），默认：false
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    # 数据是周围360度场景图片时，设为True，默认：false
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    # 每N张图像取1张作为llff测试集，paper中为8
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # --- 日志保存配置参数 ---
    # 控制台打印频率
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    # tensorboard图像日志频率
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    # 模型权重保存频率
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    # 测试集保存频率
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    # 渲染视频保存频率
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():
    '''
        训练主函数, 实现整个nerf pipeline
    '''
    # 读取参数（包括：配置文件，数据路径，log路径，实验名称，训练参数，渲染参数， 数据集参数等）
    parser = config_parser()
    args = parser.parse_args()

    K = None # 内参
    # 加载LLFF数据
    if args.dataset_type == 'llff':
        # images: 加载的图像，shape: [N_imgs, H, W, C], 像素值已归一化(pixel/255.)
        # poses: 加载的相机poses，shape: [N_imgs, 3, 5], [R, t, hwf]->3x5
        # bds: bounds，加载的边界，用于计算近和远平面距离
        # render_poses: 待渲染的新视角
        # i_test: 测试集ID，指定测试集的图像和pose
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        # hwf: image height, image width, camera focal length
        hwf = poses[0,:3,-1]
        # 去掉hwf后的3x4pose
        poses = poses[:,:3,:4] # poses: [N_imgs, 3, 4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        # 将i_test转为list
        if not isinstance(i_test, list):
            i_test = [i_test]
        # llffhold：每llffhold个图像选1个作为测试图像，paper中为8
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            # 更新后的测试集ID
            i_test = np.arange(images.shape[0])[::args.llffhold]
        # i_val：验证集ID
        i_val = i_test
        # i_train：训练集ID
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        # 如果不使用NDC坐标，near和far值由bounds确定
        # 否则，near=0,far=1
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    # 加载blender数据
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # 确保H,W为整数
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    # 得到内参矩阵K
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    # 如果渲染测试集，将render_poses设置为i_test对应的相机pose
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # 创建log文件夹并拷贝config文件
    basedir = args.basedir
    expname = args.expname
    # 创建log文件夹：basedir/expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    # 创建参数文件：basedir/expname/args.txt
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        # 读取args中的所有参数并排序，然后写入文件
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # 拷贝config文件至 basedir/expname/config.txt
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # 根据args创建nerf模型以及优化器等
    # render_kwargs_train：训练用的渲染参数，包括MLP网络，网络查询函数，单射线采样点数等
    # render_kwargs_test：测试用的渲染参数，类似render_kwargs_train
    # start：训练迭代起始ID，无ckpt时为0，有ckpt时为已经训练过的step数，方便断点训练
    # grad_vars：MLP模型参数
    # optimizer： Adam优化器
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    # 训练迭代step
    global_step = start

    # 近/远平面边界
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    # 更新近/远平面
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # 将render_poses放到device中(GPU/CPU)
    render_poses = torch.Tensor(render_poses).to(device)

    # 仅执行渲染（使用已有模型）
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            # 如果render_test是true, 使用测试数据，并将images作为groundtruth进行对比
            # 否则渲染更平滑的原始render_poses
            if args.render_test:
                images = images[i_test]
            else:
                images = None

            # 测试结果保存路径
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            # 根据render_poses, hwf和K进行模型渲染，得到一系列图像rgbs
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            # 将渲染结果保存至mp4
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # 如果no_batching是false，即使用批量化训练，需要准备批量化的射线，即random ray batching
    N_rand = args.N_rand # ray的batch size
    use_batching = not args.no_batching
    if use_batching: # 如果使用批量化
        print('get rays')
        # 获取N_imgs张图像的全部射线(每个像素对应的ray), 得到ray, shape: [N_imgs, ro+rd, H, W, 3], ro+rd=2
        # ro: 相机原点(Ox,Oy,Oz); rd: 未归一化方向向量(dx, dy, dz)
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0)
        print('done, concats')
        # 将图像进行连接得到rays_rgb, shape: [N_imgs, ro+rd+rgb, H, W, 3], ro+rd+rgb=3
        # rgb: 归一化后的rgb, 即pixel/255.
        rays_rgb = np.concatenate([rays, images[:,None]], 1)
        # 将H,W和ro+rd+rgb进行转置，shape: [N_imgs, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4])
        # 根据i_train挑选出训练数据, rays_rgb.shape: [N_train, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        # 将rays_rgb reshape为 [N_train*H*W, ro+rd+rgb, 3],便于随机选batch
        rays_rgb = np.reshape(rays_rgb, [-1,3,3])
        # 转类型，确保为float32
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        # 随机打乱rays_rgb
        np.random.shuffle(rays_rgb)

        print('done')
        # batch索引
        i_batch = 0

    # 如果使用批量化训练，则将图像放入device
    if use_batching:
        # images: [N_imgs, H, W, C]
        images = torch.Tensor(images).to(device)
    # 将poses放入device
    poses = torch.Tensor(poses).to(device)
    # 如果使用批量化训练，则将rays_rgb放入device
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    # 总迭代次数：200k+1
    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    # 迭代训练开始
    for i in trange(start, N_iters):
        # 计时开始
        time0 = time.time()

        # 随机采样ray batch
        if use_batching:
            # 取一个大小为N_rand的ray batch，shape: [N_rand, ro+rd+rgb, 3]
            batch = rays_rgb[i_batch:i_batch+N_rand]
            # 转置一下batch, shape: [ro+rd+rgb, N_rand, 3]
            batch = torch.transpose(batch, 0, 1)
            # batch[0]为ro, batch[1]为rd, batch[2]为rgb
            batch_rays, target_s = batch[:2], batch[2]

            # 更新i_batch
            i_batch += N_rand
            # 如果i_batch>=rays_rgb总数，即遍历了整个训练images的所有rays
            # 则重新打乱rays_rgb, 并将i_batch置0, 便于进行下轮训练
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else: # 从单张图像中随机选择rays
            # 从训练ID中随机选择一个image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            # 将该图像放入device
            target = torch.Tensor(target).to(device)
            # 获取该图像对应的pose
            pose = poses[img_i, :3,:4]
            
            # 如果N_rand非none，则在单张图像中随机采样N_rand个rays
            if N_rand is not None:
                # 计算单张图像所有像素对应的射线，rays_o和rays_d的shape：[H, W, 3]
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))
                
                if i < args.precrop_iters:
                    # 执行precrop_iters次计算，进行图像中心裁剪区域训练
                    dH = int(H//2 * args.precrop_frac) # 裁剪区域高度一半
                    dW = int(W//2 * args.precrop_frac) # 裁剪区域宽度一半
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1) # 裁剪区域组成的meshgrid [2*dH, 2*dW， 2]
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    # 不裁剪，使用所有像素
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                # 将coords reshape为 [H*W, 2]
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                # 随机选择
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
