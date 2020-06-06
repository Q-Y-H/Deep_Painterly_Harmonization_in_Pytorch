import argparse


def process_args(img_idx=10, img_path="data"):
    # ---- Data ----
    parser = argparse.ArgumentParser(description="A PyTorch implementation of Deep Painterly Harmonization.")
    parser.add_argument("--is-pass", choices=["1", "2"], default="1")
    parser.add_argument(
        "--content-image",
        default=f"{img_path}/{img_idx}_naive.jpg",
        help="path to the content image",
    )
    parser.add_argument(
        "--style-image",
        default=f"{img_path}/{img_idx}_target.jpg",
        help="path to the style image",
    )
    parser.add_argument(
        "--interim-image",
        help="path to the interim image",
        default=f"{img_path}/{img_idx}_naive.jpg"
    )
    parser.add_argument(
        "--mask",
        default=f"{img_path}/{img_idx}_c_mask.jpg",
        help="path to the mask image",
    )
    parser.add_argument(
        "--dilated-mask",
        default=f"{img_path}/{img_idx}_c_mask_dilated.jpg",
        help="path to the dilated/loss mask image",
    )

    # ---- Training config ----
    parser.add_argument("--style-weight", type=float, default=1e2)
    parser.add_argument("--content-weight", type=float, default=5e0)
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--histogram-weight", type=float, default=1e2)
    parser.add_argument("--lr", type=float, default=1e0)
    parser.add_argument("--epochs", type=int, default=1000, metavar="N")
    parser.add_argument("--optim", choices=["adam", "lbfgs"], default="lbfgs")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)

    # ---- Others ----
    # parser.add_argument("--cuda", action="store_true", help="Enable using GPU in training.")
    parser.add_argument("--output", help="path to the output image", default=f"output/{img_idx}_1st_pass.jpg")
    parser.add_argument("--new-iter", action="store_true")

    args = parser.parse_args()
    args.is_pass = int(args.is_pass)
    return args
