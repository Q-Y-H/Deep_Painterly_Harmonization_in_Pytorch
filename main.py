import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy

import utils
from model import get_model_and_losses
from config import process_args


def get_optimizer(input_img, lr=1, type="lbfgs"):
    if type == "adam":
        optimizer = optim.Adam([input_img.requires_grad_()], lr=lr)
    else:
        optimizer = optim.LBFGS([input_img.requires_grad_()], lr=lr)
    return optimizer


@utils.timefn
def run_painterly_transfer(
        model,
        input_img,
        style_img,
        mask_tight,
        mask_rough,
        style_losses,
        content_losses,
        histogram_losses,
        tv_loss,
        args=None,
):
    model = model.to(args.device).eval()
    for param in model.parameters():
        param.requires_grad = False

    img = input_img.clone()
    img = nn.Parameter(img)

    # input_img = nn.Parameter(input_img)
    optimizer = get_optimizer(img, lr=args.lr, type=args.optim)
    loss_history = {"content": [], "style": [], "tv": [], "histogram": [], "total": []}
    torch.cuda.empty_cache()

    def closure():
        img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(img)

        content_score = 0
        style_score = 0
        tv_score = 0
        histogram_score = 0

        for cl in content_losses:
            content_score += cl.loss
        for sl in style_losses:
            style_score += sl.loss
        if tv_loss is not None:
            tv_score = tv_loss.loss
        for hl in histogram_losses:
            histogram_score += hl.loss
        loss = style_score + content_score + tv_score + histogram_score

        loss_history["content"].append(content_score.item())
        loss_history["style"].append(style_score.item())
        if tv_loss is not None:
            loss_history["tv"].append(tv_score.item())
        loss_history["total"].append(loss.item())
        if len(histogram_losses) != 0:
            loss_history["histogram"].append((histogram_score.item()))
        # loss.backward()
        loss.backward(retain_graph=True)
        img.grad = img.grad * mask_rough

        if args.new_iter:
            run[0] += 1
        return loss

    print("==== {} ====".format("Optimization"))
    run = [0]
    # for iter in tqdm(range(0, num_steps, 1)):
    # for n range(0, args.epochs, 1):
    while run[0] < args.epochs:
        optimizer.step(closure)
        if not args.new_iter:
            run[0] += 1
        utils.print_log_period(run[0], loss_history, period=args.log_interval)
        utils.save_image_period(run[0], img, style_img, mask_tight, args.output, args.save_interval, args.train_tag)

    img.data.clamp_(0, 1)
    return img, loss_history


def main(img_idx=0, img_path="data"):
    # ---- Setup ----
    args = process_args(img_idx, img_path)
    args.train_tag = f"_{args.optim}_s{args.style_weight}_c{args.content_weight}_lr{args.lr}"

    print("\n==== {} ====".format("Device Configuration"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    args.device = device
    args.dtype = dtype
    print("Device", device)
    print("dtype", dtype)

    # ---- Load data ----
    print("\n==== {} ====".format("Loading images"))
    # 1 * 3 * W * H, 0-255, normalize to 0-1
    style_img = (utils.image_loader(args.style_image).type(dtype).to(device))
    content_img = (utils.image_loader(args.content_image).type(dtype).to(device))
    interim_img = (utils.image_loader(args.interim_image).type(dtype).to(device))
    # 1 * 1 * W * H, 0-1
    mask_tight = (utils.image_loader(args.mask, "L").type(dtype).to(device))
    mask_rough = (utils.image_loader(args.dilated_mask, "L").type(dtype).to(device))
    # 1 * 1 * W * H, 0/1
    mask_tight[mask_tight != 0] = 1
    mask_rough[mask_rough != 0] = 1
    # tmask_image = tmask_image.filter(ImageFilter.GaussianBlur())
    print("Content image shape", content_img.shape)
    print("Style image shape", style_img.shape)
    print("Mask shape", mask_rough.shape)
    print("Tight Mask shape", mask_tight.shape)

    # ---- Import vgg model ----
    print("\n==== {} ====".format("Importing vgg19 model"))
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn = copy.deepcopy(cnn)
    for param in cnn.parameters():
        param.requires_grad = False
    assert cnn.training is False

    # ---- Build Style Transfer Model ----
    print("\n==== {} ====".format("Building Deep Painterly Harmonization model"))
    # input_img = torch.randn(content_img.data.size(), device=device)
    if args.is_pass == 1:
        style_layers = ["relu3_1", "relu4_1", "relu5_1"]
        content_layers = ["relu4_1"]
        model, tv_loss, content_losses, style_losses, histogram_losses = get_model_and_losses(
            cnn,
            cnn_normalization_mean,
            cnn_normalization_std,
            mask_rough,
            mask_tight,
            style_img=style_img,
            content_img=content_img,
            interim_img=interim_img,
            style_layers=style_layers,
            content_layers=content_layers,
            args=args
        )
    elif args.is_pass == 2:
        style_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"]
        content_layers = ["relu4_1"]
        histogram_layers = ["relu1_1", "relu4_1"]
        model, tv_loss, content_losses, style_losses, histogram_losses = get_model_and_losses(
            cnn,
            cnn_normalization_mean,
            cnn_normalization_std,
            mask_rough,
            mask_tight,
            style_img=style_img,
            content_img=content_img,
            interim_img=interim_img,
            style_layers=style_layers,
            content_layers=content_layers,
            histogram_layers=histogram_layers,
            args=args,
        )
    else:
        print("Wrong input for the option --is-pass. Exit.")
        return

    # ---- Run model ----
    print("\n==== {} ====".format("Start training."))
    output_tensor, loss_history = run_painterly_transfer(
        model,
        content_img,
        style_img,
        mask_tight,
        mask_rough,
        style_losses,
        content_losses,
        histogram_losses,
        tv_loss,
        args=args,
    )

    # ---- Plot lost history and save final result----
    print("\n==== {} ====".format("Plot loss history figure."))
    utils.history_plot(loss_history, args.train_tag, args.output)

    print(f"\n==== Save the final output image to {args.output} ====")
    output_masked = utils.mask_crop(output_tensor, style_img, mask_tight)
    utils.save_image(output_masked, args.output)

    return


if __name__ == "__main__":
    main()
