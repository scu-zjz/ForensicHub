import cv2
import torch
from PIL import Image
import numpy as np
import tempfile
import os
import io
from torch.nn import functional as F


def import_jpegio():
    try:
        import jpegio
        return jpegio
    except ImportError:
        raise ImportError("Please install jpegio first: pip install jpegio")


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def convert_to_temp_jpeg(tensor):
    # 将tensor转换为numpy数组
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    # 确保数据在0-255范围内
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)

    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_path = temp_file.name

    # 保存为JPEG
    cv2.imwrite(temp_path, tensor)

    return temp_path


def read_jpeg_from_memory(image_data):
    # 将图像数据转换为字节流
    if isinstance(image_data, bytes):
        image_bytes = image_data
    else:
        image_bytes = image_data.tobytes()

    # 使用jpegio读取内存中的JPEG数据
    jpegio = import_jpegio()
    jpeg_obj = jpegio.read_from_memory(image_bytes)

    return jpeg_obj


class EdgeMaskGenerator(torch.nn.Module):
    """generate the 'edge bar' for a 0-1 mask Groundtruth of a image
    Algorithm is based on 'Morphological Dilation and Difference Reduction'
    
    Which implemented with fixed-weight Convolution layer with weight matrix looks like a cross,
    for example, if kernel size is 3, the weight matrix is:
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]

    """

    def __init__(self, kernel_size=3) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def _dilate(self, image, kernel_size=3):
        """Doings dilation on the image

        Args:
            image (_type_): 0-1 tensor in shape (B, C, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert image.shape[2] > kernel_size and image.shape[3] > kernel_size, "Image must be larger than kernel size"

        kernel = torch.zeros((1, 1, kernel_size, kernel_size))
        kernel[0, 0, kernel_size // 2: kernel_size // 2 + 1, :] = 1
        kernel[0, 0, :, kernel_size // 2: kernel_size // 2 + 1] = 1
        kernel = kernel.float()
        # print(kernel)
        res = F.conv2d(image, kernel.view([1, 1, kernel_size, kernel_size]), stride=1, padding=kernel_size // 2)
        return (res > 0) * 1.0

    def _find_edge(self, image, kernel_size=3, return_all=False):
        """Find 0-1 edges of the image

        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        image = torch.tensor(image).float()
        shape = image.shape

        if len(shape) == 2:
            image = image.reshape([1, 1, shape[0], shape[1]])
        if len(shape) == 3:
            image = image.reshape([1, shape[0], shape[1], shape[2]])
        assert image.shape[1] == 1, "Image must be single channel"

        img = self._dilate(image, kernel_size=kernel_size)

        erosion = self._dilate(1 - image, kernel_size=kernel_size)

        diff = -torch.abs(erosion - img) + 1
        diff = (diff > 0) * 1.0
        # res = dilate(diff)
        diff = diff.numpy()
        if return_all:
            return diff, img, erosion
        else:
            return diff

    def forward(self, x, return_all=False):
        """
        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        return self._find_edge(x, self.kernel_size, return_all=return_all)
