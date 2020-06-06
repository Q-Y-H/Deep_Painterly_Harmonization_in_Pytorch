import os
import time
from functools import wraps
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        time_elapsed = time.time() - t1
        print("@timefn: {} took {:.0f}m {:.0f}s".format(fn.__name__, time_elapsed // 60, time_elapsed % 60))
        return result

    return measure_time


loader = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    ])
unloader = transforms.Compose([transforms.ToPILImage()])


def PIL2tensor(image):
    image = loader(image).unsqueeze(0)
    return image


def tensor2PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def image_loader(image_name, flag="RGB"):
    image = Image.open(image_name).convert(flag)  # H * W
    return PIL2tensor(image)  # -1 * -1 * W * H


def mask_crop(img, style, mask):
    assert (img.shape == style.shape)
    mask_c = mask.clone().detach().expand_as(img)
    return img * mask_c + style * (1.0 - mask_c)


def print_log_period(iter, history, period=50):
    if iter % period != 0:
        return

    log_title = f"epoch [{iter}]: "
    log_body = "Total loss: {:3f}, Content loss: {:3f}, Style loss: {:3f}".format(
        history["total"][-1],
        history["content"][-1],
        history["style"][-1])
    if len(history["tv"]) != 0:
        log_body += ", TV loss: {:3f}".format(history["tv"][-1])

    print(log_title + log_body)


def save_image_period(iter, input_img, style_img, mask, path, period=50, train_tag=""):
    if iter % period != 0:
        return

    input_img.data.clamp_(0, 1)
    new_img = mask_crop(input_img, style_img, mask)

    save_path, save_name = os.path.split(path)
    file_name, file_ext = os.path.splitext(save_name)
    file_name += f"{train_tag}_e{iter}"
    save_name = file_name + file_ext
    save_image(new_img, save_path, save_name)


def save_image(tensor, save_path, save_name=None):
    if save_name is None:
        save_path, save_name = os.path.split(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_path = os.path.join(save_path, save_name)
    print(f"Save image to {img_path}")
    img = tensor2PIL(tensor)
    img.save(img_path)


def history_plot(history, train_tag, out_path):
    img_idx = os.path.split(out_path)[1].split("_")[0]
    for k, v in history.items():
        x = list(range(len(v)))
        plt.plot(x, v, label=k)
    plt.title("Loss History")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(f"{img_idx}_loss_history{train_tag}.jpg")
