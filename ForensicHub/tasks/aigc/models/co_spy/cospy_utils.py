import os
import cv2
import torch
import pickle
import random
import numpy as np
from io import BytesIO
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF
from scipy.ndimage.filters import gaussian_filter

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Set random seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Load dataset
def recursively_read(rootdir, must_contain, exts=["png", "PNG", "jpg", "JPG", "jpeg", "JPEG"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


# Data augmentation techniques
def data_augment(img, aug_config):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random.random() < aug_config["blur_prob"]:
        sig = sample_continuous(aug_config["blur_sig"])
        gaussian_blur(img, sig)

    if random.random() < aug_config["jpg_prob"]:
        method = sample_discrete(aug_config["jpg_method"])
        qual = sample_discrete(aug_config["jpg_qual"])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


# Data augmentation techniques
def tensor_data_augment(images, aug_config):
    device = images.device
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = np.uint8(images * 255.)
    outputs = []
    for img in images:
        if random.random() < aug_config["blur_prob"]:
            sig = sample_continuous(aug_config["blur_sig"])
            gaussian_blur(img, sig)

        if random.random() < aug_config["jpg_prob"]:
            method = sample_discrete(aug_config["jpg_method"])
            qual = sample_discrete(aug_config["jpg_qual"])
            img = jpeg_from_key(img, qual, method)
        outputs.append(img)
    outputs = np.stack(outputs)
    outputs = torch.from_numpy(outputs).to(device).permute(0, 3, 1, 2).float() / 255.
    return outputs


# Sample continuous or discrete values
def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


# Gaussian blur
def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


# JPEG compression
def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


def png_to_jpeg(img, quality=95):
    # Convert the PNG image to JPEG
    # Input: PIL image
    # Output: PIL image
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)
    img = np.array(Image.open(out))
    # Load from memory before ByteIO closes
    out.close()
    img = Image.fromarray(img)
    return img


def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)


# Custom resize function
def custom_resize(img, rz_interp, loadSize):
    rz_dict = {'bilinear': Image.BILINEAR,
                'bicubic': Image.BICUBIC,
                'lanczos': Image.LANCZOS,
                'nearest': Image.NEAREST}
    interp = sample_discrete(rz_interp)
    return TF.resize(img, loadSize, interpolation=rz_dict[interp])


def weights2cpu(weights):
    for key in weights:
        weights[key] = weights[key].cpu()
    return weights
