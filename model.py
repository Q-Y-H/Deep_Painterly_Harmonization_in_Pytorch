import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
import gc

import utils

vgg19_layers = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
]

style_layers_1st_pass = ["relu3_1", "relu4_1", "relu5_1"]
content_layers_1st_pass = ["relu4_1"]
histogram_layers_1st_pass = []


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class TVLoss(nn.Module):
    def __init__(self, weight):
        super(TVLoss, self).__init__()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff))) * self.weight
        return input


class ContentLoss(nn.Module):
    def __init__(self, target, mask, weight):
        super(ContentLoss, self).__init__()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0
        self.target = target.detach() * self.mask

    def forward(self, input):
        self.loss = F.mse_loss(input * self.mask, self.target) / self.mask.sum() / self.target.shape[
            1] * input.nelement() * self.weight
        return input

    def content_hook(self, module, grad_input, grad_output):
        return tuple([grad_input[0] * self.mask])


def gram(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    # return G.div(a * b * c * d)
    return G


class StyleLoss(nn.Module):
    def __init__(self, target, mask, weight):
        super(StyleLoss, self).__init__()
        self.match_G = target.detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.input_G = gram(input * self.mask)
        self.loss = F.mse_loss(self.input_G, self.match_G) * self.weight / (self.mask.sum() * self.match_G.shape[1])
        return input

    def style_hook(self, module, grad_input, grad_output):
        return tuple([grad_input[0] * self.mask])


class StyleLoss2(nn.Module):
    def __init__(self, target, mask, weight, match=None):
        super(StyleLoss2, self).__init__()
        self.target = target.detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0
        self.match = match

    def forward(self, input):
        self.input_G = gram(input * self.mask)
        self.loss = F.mse_loss(self.input_G, self.target) * self.weight / (self.mask.sum() * input.shape[1])
        return input

    def style_hook(self, module, grad_input, grad_output):
        return tuple([grad_input[0] * self.mask])


class HistogramLoss(nn.Module):
    def __init__(self, weight, mask_rough, mask_tight, bins):
        super(HistogramLoss, self).__init__()
        self.weight = weight
        self.mask_rough = mask_rough
        self.mask_tight = mask_tight
        self.bins = bins
        self.match = None

        self.style_his = None
        self.count = 0

    def compute_histogram(self):
        assert (self.match is not None)
        match_masked = self.match * self.mask_tight

        self.style_his = torch.stack(
            [torch.histc(match_masked[:, c, :, :], self.bins) for c in range(match_masked.shape[1])], dim=0)

        del match_masked, self.match
        gc.collect()
        self.match = None

    def select_idx(self, hist, idx):
        return hist.view(-1)[idx.view(-1)].view(hist.shape[0], -1)

    def remap_histogram(self, input):
        # Only Use the masked region & reshape to (channel, N)
        input = (input * self.mask_tight).reshape((input.shape[1], -1))
        C, N = input.shape

        # Sort feature map & remember corresponding index for each channel
        sort_fm, sort_idx = input.sort(1)
        channel_min, channel_max = input.min(1)[0].unsqueeze(1), input.max(1)[0].unsqueeze(1)

        step = (channel_max - channel_min) / self.bins
        rng = torch.arange(1, N + 1).unsqueeze(0)  # .to(self.device)

        # Since style histogran not nessary have same number of N, scale it
        style_his = self.style_his * N / self.style_his.sum(1).unsqueeze(1)  # torch.Size([channel, 256])
        style_his_cdf = style_his.cumsum(1)  # torch.Size([channel, 256])
        del style_his, self.style_his
        style_his_cdf_prev = torch.cat([torch.zeros(C, 1), style_his_cdf[:, :-1]], 1)  # torch.Size([channel, 256])

        # Find Corresponding
        idx = (style_his_cdf.unsqueeze(1) - rng.unsqueeze(2) < 0).sum(2).long()  # index need long tensor
        ratio = (rng - self.select_idx(style_his_cdf_prev, idx)) / (1e-8 + self.select_idx(style_his_cdf, idx))
        del style_his_cdf_prev
        del style_his_cdf
        ratio = ratio.squeeze().clamp(0, 1)

        # Build Correspponding FM
        input_corr = channel_min + (ratio + idx) * step
        del ratio
        input_corr[:, -1] = channel_max[:, 0]
        _, remap = sort_idx.sort()
        input_corr = self.select_idx(input_corr, idx)

        return input_corr

    def forward(self, input):
        if self.count % 50 == 0:
            self.corr_input = self.remap_histogram(input)
        self.count += 1
        self.loss = F.mse_loss(self.corr_input, torch.mul(input, self.mask_tight).reshape((input.shape[1], -1)))
        self.loss = self.loss.to(self.device) * self.mask_tight.sum() * input.shape[1] / input.nelement() * self.weight
        return input

    def histogram_hook(self, module, grad_input, grad_output):
        return tuple([grad_input[0] * self.mask])


@utils.timefn
def get_model_and_losses(cnn, normalization_mean, normalization_std, mask_rough, mask_tight,
                         style_img=None, content_img=None, interim_img=None,
                         content_layers=None, style_layers=None, histogram_layers=None,
                         args=None):
    if style_layers is None:
        style_layers = style_layers_1st_pass
    if content_layers is None:
        content_layers = content_layers_1st_pass
    if histogram_layers is None:
        histogram_layers = histogram_layers_1st_pass

    normalization = Normalization(normalization_mean, normalization_std).to(args.device)
    model = nn.Sequential(normalization)

    # total variation loss
    tv_loss = None
    if args.tv_weight > 0:
        tv_loss = TVLoss(args.tv_weight)
        model.add_module("tv_loss", tv_loss)

    content_losses = []
    style_losses = []
    histogram_losses = []

    i, j = 1, 0
    name = ""
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            j += 1
            name = "conv{}_{}".format(i, j)
            sap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            mask_rough = sap(mask_rough)

        elif isinstance(layer, nn.ReLU):
            name = "relu{}_{}".format(i, j)
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            name = "pool{}".format(i)
            mask_rough = F.interpolate(mask_rough, scale_factor=(0.5, 0.5))
            print(f"{name}: Resize mask to {mask_rough.size()}")
            i += 1
            j = 0

        model.add_module(name, layer)

        if name in content_layers:
            print(f"{name}: Add content loss layer.")
            target = model(content_img).detach()
            content_loss = ContentLoss(target, mask_rough, args.content_weight)
            content_loss.register_backward_hook(content_loss.content_hook)
            model.add_module("contentLoss{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            print(f"{name}: Add style loss layer.")
            style_loss = None
            if args.is_pass == 1:
                content_feature_map = model(content_img).detach()
                style_feature_map = model(style_img).detach()

                match, mapping = patch_match(content_feature_map, style_feature_map, device=args.device)
                print(f"\tMatch size: {match.size()}")

                mask_c = mask_rough.clone()
                G = gram(match * mask_c)
                style_loss = StyleLoss(G, mask_c, args.style_weight)
                del content_feature_map, style_feature_map, match, mapping
            else:
                if name != style_layers[-1]:
                    style_feature_map = model(style_img).detach()
                    style_loss = StyleLoss2(style_feature_map, mask_rough.clone(), args.style_weight)

                else:
                    pre_feature_map = model(interim_img).detach()
                    style_feature_map = model(style_img).detach()

                    ref_corr, match = patch_match_ref(pre_feature_map, style_feature_map, device=args.device)
                    print(f"\tMatch size: {match.size()}")

                    G = gram(match * mask_rough)
                    del pre_feature_map, style_feature_map
                    style_loss = StyleLoss2(G, mask_rough, args.style_weight, match)

                    for l in style_losses:
                        match = upsample_corr(ref_corr, l.target)
                        G = gram(match * l.mask)
                        del l.target
                        l.target = G
                        l.match = match

            style_loss.register_backward_hook(style_loss.style_hook)
            model.add_module("styleLoss{}".format(i), style_loss)
            style_losses.append(style_loss)

        if name in histogram_layers:
            print(f"{name}: Add histogram loss layer.")
            histogram_loss = HistogramLoss(args.histogram_weight, mask_rough, mask_tight, bins=256)
            if name == histogram_layers[0]:
                histogram_loss.match = style_losses[0].match
            elif name == histogram_layers[1]:
                histogram_loss.match = style_losses[-1].match
            histogram_loss.compute_histogram()
            histogram_loss.register_backward_hook(histogram_loss.histogram_hook)
            model.add_module("histogramtLoss{}".format(i), histogram_loss)
            histogram_losses.append(histogram_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            model = model[:i + 1]
            break

    # Post process model
    del cnn

    model = model.to(args.device).eval()
    for param in model.parameters():
        param.requires_grad = False
    print(model)

    return model, tv_loss, content_losses, style_losses, histogram_losses


@utils.timefn
def patch_match(content, style, patch_size=3, stride=1, device=torch.device("cpu")):
    assert (content.shape == style.shape)
    N, C, H, W = content.shape
    padding = patch_size // 2

    content_pad = F.pad(content, [padding, ] * 4, mode="reflect")  # l, r, t, b
    style_pad = F.pad(style, [padding, ] * 4, mode="reflect")

    ones_kernel = torch.ones(1, C, patch_size, patch_size).to(device)
    style_pad_norm = F.conv2d(style_pad ** 2, ones_kernel, padding=0, stride=stride) ** 0.5

    # grid_x = torch.zeros(1, H, W).to(device)
    # grid_y = torch.zeros(1, H, W).to(device)
    match = style.clone()

    for i in tqdm(range(H // stride)):
        for j in range(W // stride):
            patch_kernel = content_pad[:, :, i:i + patch_size, j:j + patch_size].clone()
            patch_kernel_norm = (patch_kernel ** 2).sum() ** 0.5

            score_map = F.conv2d(style_pad, patch_kernel, padding=0, stride=stride)  # 1 * 1 * H * W
            score_map = score_map / (style_pad_norm * patch_kernel_norm + 1e-9)
            score_map = score_map[0, 0, :, :]
            max_idx = torch.argmax(score_map).item()
            match[:, :, i, j] = style[:, :, int(max_idx // score_map.shape[1]), int(max_idx % score_map.shape[1])]

    return match, N


def get_patch(map, h, w, patch_size):
    return map[:, :, h - 1:h - 1 + patch_size, w - 1:w - 1 + patch_size]


@utils.timefn
def patch_match_ref(interim, style, patch_size=3, stride=1, device=torch.device("cpu")):
    N, C, H, W = interim.size()
    padding = patch_size // 2
    style_pad = F.pad(style, [padding, ] * 4, mode='reflect')

    # nearest neighbor index for ref_layer: H_ref * W_ref, same as P_out in paper
    ref_corr = np.zeros((H, W))  # Output

    # Step 1: Find matches for the reference layer.
    match, mapping = patch_match(interim, style, device=device)

    # Step 2: Enforce spatial consistency.
    patch_H, patch_W = mapping.shape[:-1]
    offsets_x = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    offsets_y = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    for i in range(patch_H):
        for j in range(patch_W):
            # Initialize a set of candidate style patches.
            candidate_set = set()
            # For all adjacent patches, look up the style patch of the adjacent patch p + o
            # and apply the opposite offset −o.
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    # skip if out of bounds
                    if i + di < 0 or i + di >= patch_H or j + dj < 0 or j + dj >= patch_W:
                        continue

                    patch_idx = mapping[i + di, j + dj, :]  # index of neighbor patch in style feature map
                    patch_pos = (patch_idx[0] - di, patch_idx[1] - dj)

                    # skip if out of bounds
                    if patch_pos[0] < 0 \
                            or patch_pos[0] >= patch_H \
                            or patch_pos[1] < 0 \
                            or patch_pos[1] >= patch_W:
                        continue

                    candidate_set.add((patch_pos[0], patch_pos[1]))

            # Select the candidate the most similar to the style patches
            # associated to the neighbors of p.
            min_sum = np.inf
            for c_h, c_w in candidate_set:
                style_fm_ref_c = get_patch(style_pad, c_h, c_w, patch_size)
                sum = 0

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        # skip if out of bounds
                        if i + di < 0 or i + di >= patch_H or j + dj < 0 or j + dj >= patch_W:
                            continue

                        patch_idx = mapping[i + di, j + dj, :]
                        patch_pos = (patch_idx[0], patch_idx[1])

                        # skip if out of bounds
                        if patch_pos[0] < 0 \
                                or patch_pos[0] >= patch_H \
                                or patch_pos[1] < 0 \
                                or patch_pos[1] >= patch_W:
                            continue

                        # get patch from style_fm at (patch_pos[0], patch_pos[1])
                        style_fm_ref_p = get_patch(style_pad, patch_pos[0], patch_pos[1], patch_size)
                        sum += F.conv2d(style_fm_ref_c, style_fm_ref_p).item()

                if sum < min_sum:
                    min_sum = sum
                    ref_corr[i, j] = c_h * patch_W + c_w

    # Step 3: Create style_fm_matched based on ref_corr
    style_fm_matched = style.clone()
    for i in range(patch_H):
        for j in range(patch_W):
            # Find matched index in style fm
            matched_style_idx = (int(ref_corr[i, j]) // W, int(ref_corr[i, j]) % W)  # 1 * C * 3 * 3
            style_fm_matched[:, :, i, j] = style[:, :, matched_style_idx[0], matched_style_idx[1]]

    return ref_corr.astype(np.int), style_fm_matched


def upsample_corr(ref_corr, style_fm):
    curr_h, curr_w = style_fm.shape[:-1]
    curr_corr = np.zeros((curr_h, curr_w))
    ref_h, ref_w = ref_corr.shape

    h_ratio = curr_h / ref_h
    w_ratio = curr_w / ref_w

    style_fm_matched = style_fm.clone()

    for i in range(curr_h):
        for j in range(curr_w):
            ref_idx = [(i + 0.4999) // h_ratio, (j + 0.4999) // w_ratio]
            ref_idx[0] = int(max(min(ref_idx[0], ref_h - 1), 0))
            ref_idx[1] = int(max(min(ref_idx[1], ref_w - 1), 0))

            ref_mapping_idx = ref_corr[ref_idx[0], ref_idx[1]]
            ref_mapping_idx = (ref_mapping_idx // ref_w, ref_mapping_idx % ref_w)

            curr_mapping_idx = (int(i + (ref_mapping_idx[0] - ref_idx[0]) * h_ratio + 0.4999),
                                int(j + (ref_mapping_idx[1] - ref_idx[1]) * w_ratio + 0.4999))
            curr_corr[i, j] = curr_mapping_idx[0] * curr_w + curr_mapping_idx[1]

            #  1 * C * 3 * 3
            style_fm_matched[:, :, i, j] = style_fm[:, :, curr_mapping_idx[0], curr_mapping_idx[1]]

    return curr_corr, style_fm_matched


def histogram_match(input, target, patch, stride):
    n1, c1, h1, w1 = input.size()
    n2, c2, h2, w2 = target.size()
    input.resize_(h1 * w1 * h2 * w2)
    target.resize_(h2 * w2 * h2 * w2)
    conv = torch.tensor((), dtype=torch.float32)
    conv = conv.new_zeros((h1 * w1, h2 * w2))
    conv.resize_(h1 * w1 * h2 * w2)
    assert c1 == c2, 'input:c{} is not equal to target:c{}'.format(c1, c2)

    size1 = h1 * w1
    size2 = h2 * w2
    N = h1 * w1 * h2 * w2
    print('N is', N)

    for i in range(0, N):
        i1 = i / size2
        i2 = i % size2
        x1 = i1 % w1
        y1 = i1 / w1
        x2 = i2 % w2
        y2 = i2 / w2
        kernal_radius = int((patch - 1) / 2)

        conv_result = 0
        norm1 = 0
        norm2 = 0
        dy = -kernal_radius
        dx = -kernal_radius
        while dy <= kernal_radius:
            while dx <= kernal_radius:
                xx1 = x1 + dx
                yy1 = y1 + dy
                xx2 = x2 + dx
                yy2 = y2 + dy
                if 0 <= xx1 < w1 and 0 <= yy1 < h1 and 0 <= xx2 < w2 and 0 <= yy2 < h2:
                    _i1 = yy1 * w1 + xx1
                    _i2 = yy2 * w2 + xx2
                    for c in range(0, c1):
                        term1 = input[int(c * size1 + _i1)]
                        term2 = target[int(c * size2 + _i2)]
                        conv_result += term1 * term2
                        norm1 += term1 * term1
                        norm2 += term2 * term2
                dx += stride
            dy += stride
        norm1 = math.sqrt(norm1)
        norm2 = math.sqrt(norm2)
        conv[i] = conv_result / (norm1 * norm2 + 1e-9)

    match = torch.tensor((), dtype=torch.float32)
    match = match.new_zeros(input.size())

    correspondence = torch.tensor((), dtype=torch.int16)
    correspondence.new_zeros((h1, w1, 2))
    correspondence.resize_(h1 * w1 * 2)

    for id1 in range(0, size1):
        conv_max = -1e20
        for y2 in range(0, h2):
            for x2 in range(0, w2):
                id2 = y2 * w2 + x2
                id = id1 * size2 + id2
                conv_result = conv[id1]

                if conv_result > conv_max:
                    conv_max = conv_result
                    correspondence[id1 * 2 + 0] = x2
                    correspondence[id1 * 2 + 1] = y2

                    for c in range(0, c1):
                        match[c * size1 + id1] = target[c * size2 + id2]

    match.resize_((n1, c1, h1, w1))

    return match, correspondence
