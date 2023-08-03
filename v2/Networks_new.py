import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

from superpoint import SuperPoint


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            src0 = desc0
            delta0 = layer(desc0, src0)
            desc0 = (desc0 + delta0)
        return desc0


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue_Network(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        self.pool_layer = nn.AdaptiveMaxPool2d((self.config['descriptor_dim']))

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(str(path)))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))

    def forward(self, data, superpoint):
        # Extract SuperPoint (keypoints, scores, descriptors)
        if 'keypoints' not in data:
            pred = superpoint({'image': data['image']})

        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0 = data['descriptors'] # shape:bs*256*num_points
        kpts0 = data['keypoints'] # shape:bs*256*num_points*2

        if kpts0.shape[1] == 0:  # no keypoints
            shape0 = kpts0.shape[:-1]
            return None

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores'])

        # Multi-layer Transformer network.
        desc0 = self.gnn(desc0)

        # Final MLP projection.
        mdesc0 = self.final_proj(desc0)

        mdesc0 = torch.permute(mdesc0, (0, 2, 1))

        mdesc0 = torch.unsqueeze(mdesc0, 1)

        mdesc0 = self.pool_layer(mdesc0)

        bs = mdesc0.shape[0]
        mdesc0 = mdesc0.view(bs,-1)

        return mdesc0

class Resnet50_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet50_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=True) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.pool_method =  nn.AdaptiveMaxPool2d(1)


    def forward(self, input, bb_box = None):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        return F.normalize(x)


class VGG_Network(nn.Module):
    def __init__(self, hp):
        super(VGG_Network, self).__init__()
        self.backbone = backbone_.vgg16(pretrained=True).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box = None):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        return F.normalize(x)

class InceptionV3_Network(nn.Module):
    def __init__(self, hp):
        super(InceptionV3_Network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default


    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        backbone_tensor = self.Mixed_7c(x)
        feature = self.pool_method(backbone_tensor).view(-1, 2048)
        return F.normalize(feature)
