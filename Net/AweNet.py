import torch
from torch import nn
from Net.basic import *
from Net.defineViT import ViT
from Net.metaformer3D import poolformerv2_3d_s12
import torch.nn.functional as F
from efficientnet_pytorch_3d import EfficientNet3D
import itertools
class Conv1d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv1d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm1d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

class CascadedGroupAttention1D(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=128, kernels=None):
        super().__init__()
        if kernels is None:
            kernels = [5] * num_heads
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv1d_BN(dim // num_heads, self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv1d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(nn.ReLU(), Conv1d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        attention_biases = torch.zeros(num_heads, resolution)
        self.attention_biases = nn.Parameter(attention_biases)
        self.register_buffer('attention_bias_idxs', torch.arange(resolution).unsqueeze(0).expand(resolution, -1))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B, L, C)
        print("Input:", x.shape)
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, L)
        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out = []
        feat = feats_in[0]
        ab = self.attention_biases[:, self.attention_bias_idxs]

        for i, qkv in enumerate(self.qkvs):
            if i > 0:
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = (q.transpose(1, 2) @ k) * self.scale + (ab[i] if self.training else self.ab[i])
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(1, 2)).view(B, self.d, L)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, dim=1))
        x = x.permute(0, 2, 1)
        print("Output:", x.shape)
        return x

class CascadedGroupCrossAttention1D(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=128, kernels=None):
        super().__init__()
        if kernels is None:
            kernels = [5] * num_heads
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5
        qkvs_x = []
        dws_x = []
        qkvs_y = []
        dws_y = []
        for i in range(num_heads):
            qkvs_x.append(Conv1d_BN(dim // num_heads, self.key_dim * 2 + self.d, resolution=resolution))
            dws_x.append(Conv1d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs_x = nn.ModuleList(qkvs_x)
        self.dws_x = nn.ModuleList(dws_x)
        self.proj_x = nn.Sequential(nn.ReLU(), Conv1d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        for i in range(num_heads):
            qkvs_y.append(Conv1d_BN(dim // num_heads, self.key_dim * 2 + self.d, resolution=resolution))
            dws_y.append(Conv1d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs_y = nn.ModuleList(qkvs_y)
        self.dws_y = nn.ModuleList(dws_y)
        self.proj_y = nn.Sequential(nn.ReLU(), Conv1d_BN(self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        attention_biases = torch.zeros(num_heads, resolution)
        self.attention_biases = nn.Parameter(attention_biases)
        self.register_buffer('attention_bias_idxs', torch.arange(resolution).unsqueeze(0).expand(resolution, -1))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def compute_attention(self, q, k, v, idx, i):
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(1, 2) @ k) * self.scale
        ab = self.attention_biases[:, idx] if self.training else self.ab
        attn += ab[i]
        attn = attn.softmax(dim=-1)
        return v @ attn.transpose(1, 2)
    def forward(self, x, y):  # x (B, L, C)
        assert x.shape == y.shape, "x, y shapes must be the same"
        print("Input:", x.shape)
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, L)
        y = y.permute(0, 2, 1)
        feats_in_x = x.chunk(self.num_heads, dim=1)
        feats_out_x = []
        feat_x = feats_in_x[0]
        feats_in_y = y.chunk(self.num_heads, dim=1)
        feats_out_y = []
        feat_y = feats_in_y[0]
        for i in range(self.num_heads):
            if i > 0:
                feat_x = feat_x + feats_in_x[i]
                feat_y = feat_y + feats_in_y[i]
            feat_x = self.qkvs_x[i](feat_x)
            feat_y = self.qkvs_y[i](feat_y)
            q_x, k_x, v_x = feat_x.split([self.key_dim, self.key_dim, self.d], dim=1)
            q_y, k_y, v_y = feat_y.split([self.key_dim, self.key_dim, self.d], dim=1)
            q_x = self.dws_x[i](q_x)
            q_y = self.dws_y[i](q_y)
            feat_x = self.compute_attention(q_y, k_x, v_x, self.attention_bias_idxs, i).view(B, self.d, L)
            feat_y = self.compute_attention(q_x, k_y, v_y, self.attention_bias_idxs, i).view(B, self.d, L)
            feats_out_x.append(feat_x)
            feats_out_y.append(feat_y)
        result_x = self.proj_x(torch.cat(feats_out_x, dim=1))
        result_y = self.proj_y(torch.cat(feats_out_y, dim=1))
        result_x = result_x.permute(0, 2, 1)
        result_y = result_y.permute(0, 2, 1)
        return result_x, result_y

class AweSomeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AweSomeNet, self).__init__()
        self.name = 'AweSomeNet'
        self.MriExtraction = get_no_pretrained_vision_encoder(n_classes=128)
        self.PetExtraction = get_no_pretrained_vision_encoder(n_classes=128)
        self.Table = TransformerEncoder(output_dim=128)

        self.fusion = CascadedGroupCrossAttention1D(dim=256, key_dim=16, num_heads=4, resolution=1, attn_ratio=4)

        # self.mamba1 = SelfMamba(256, 256, hidden_dropout_prob=0.2, d_state=64)
        # self.mamba2 = SelfMamba(256, 256, hidden_dropout_prob=0.2, d_state=64)
        # self.mamba3 = SelfMamba(256, 256, hidden_dropout_prob=0.2, d_state=64)

        # self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        # self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        # self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)

        # self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=num_classes)
        self.classify_head = MlpKan(init_features=256 * 2, classes=num_classes)

    def forward(self, mri, pet, cli):
        """
        Mri: [8, 1, 96, 128, 96]
        Pet: [8, 1, 96, 128, 96]
        Clinicla: [8, 9]
        """
        mri_feature = self.MriExtraction(mri)  # [8, 256]
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.PetExtraction(pet)  # [8, 256]
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)  # [8, 256]

        mri_feature = torch.unsqueeze(mri_feature, dim=1)
        pet_feature = torch.unsqueeze(pet_feature, dim=1)
        cli_feature = torch.unsqueeze(cli_feature, dim=1)

        # mri_feature = self.SA1(mri_feature)
        # pet_feature = self.SA2(pet_feature)
        # cli_feature = self.SA3(cli_feature)
        mri_cli_feature = torch.cat((mri_feature, cli_feature), dim=-1)
        pet_cli_feature = torch.cat((pet_feature, cli_feature), dim=-1)

        # mri_feature.shape torch.Size([8, 1, 256])
        # pet_feature.shape torch.Size([8, 1, 256])
        # cli_feature.shape torch.Size([8, 1, 256])

        result_mri_cli, result_pet_cli = self.fusion(mri_cli_feature, pet_cli_feature)
        result = torch.cat((result_mri_cli, result_pet_cli), dim=-1)

        output = self.classify_head(result)
        return output



if __name__ == '__main__':
    model = AweSomeNet()
    x = torch.randn(8, 1, 96, 128, 96)
    y = torch.randn(8, 1, 96, 128, 96)
    z = torch.randn(8, 9)
    output = model(x, y, z)
    print(output.shape)
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(x, y, z))
    flops, params = clever_format([flops, params], "%.3f")
    print(f'flops:{flops}, params:{params}')
    torch.save(model.state_dict(), '../AweSomeNet.pth')

    # model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_classes=128)
    # from thop import profile, clever_format
    # flops, params = profile(model, inputs=(x,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f'flops:{flops}, params:{params}')
    # torch.save(model.state_dict(), 'AweSomeNet.pth')


