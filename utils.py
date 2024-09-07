"""
Author: Gao Yu
Company: Bosch Research / Asia Pacific
Date: 2024-08-03
Description: util functions for stegaformer
"""

import os
from glob import glob
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data

class VGGLoss(nn.Module):
    """
    Part of pre-trained VGG16.
    See for instance https://arxiv.org/abs/1603.08155

    block_no：how many blocks do we need; layer_within_block：which layer within the block do we need
    """

    def __init__(self, block_no: int, layer_within_block: int, use_batch_norm_vgg: bool):
        super(VGGLoss, self).__init__()
        if use_batch_norm_vgg:
            vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        else:
            vgg16 = torchvision.models.vgg16(pretrained=True)
        curr_block = 1
        curr_layer = 1
        layers = []
        for layer in vgg16.features.children():
            layers.append(layer)
            if curr_block == block_no and curr_layer == layer_within_block:
                break
            if isinstance(layer, nn.MaxPool2d):
                curr_block += 1
                curr_layer = 1
            else:
                curr_layer += 1
        self.vgg_loss = nn.Sequential(*layers)

    def forward(self, img):
        return self.vgg_loss(img)

def infinite_sampler(n: int):
    """
    Yields an infinite sequence of random indices from 0 to n-1.

    Args:
        n (int): The number of samples.

    Yields:
        int: Random index from 0 to n-1.
    """
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.Sampler):
    """
    A data sampler that wraps around another sampler to yield an infinite sequence of samples.
    
    Args:
        data_source (Dataset): The dataset to sample from.
    """
    
    def __init__(self, data_source: data.Dataset):
        self.num_samples = len(data_source)

    def __iter__(self):
        """
        Returns an iterator that yields an infinite sequence of samples.
        
        Returns:
            Iterator[int]: An iterator over random indices.
        """
        return iter(infinite_sampler(self.num_samples))

    def __len__(self) -> int:
        """
        Returns a large number to simulate an infinite length.
        
        Returns:
            int: A large integer value.
        """
        return 2 ** 31
        

def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image to YUV color space.

    Args:
        image (torch.Tensor): Input image tensor with shape (*, 3, H, W).

    Returns:
        torch.Tensor: Image tensor in YUV color space with shape (*, 3, H, W).

    Raises:
        TypeError: If the input is not a torch.Tensor.
        ValueError: If the input tensor does not have at least 3 dimensions or the third dimension is not 3.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    yuv_image: torch.Tensor = torch.stack([y, u, v], dim=-3)

    return yuv_image

def get_message_accuracy(
    msg: torch.Tensor,
    deco_msg: torch.Tensor,
    msg_num: int
) -> float:
    """
    Calculate the bit accuracy of decoded messages.

    Args:
        msg (torch.Tensor): The original message tensor.
        deco_msg (torch.Tensor): The decoded message tensor.
        msg_num (int): The number of the message segments.

    Returns:
        float: The bit accuracy of the decoded messages.

    Raises:
        TypeError: If the inputs are not torch.Tensors or msg_num is not an int.
        ValueError: If the msg_num is less than 1.
    """
    if not isinstance(msg, torch.Tensor) or not isinstance(deco_msg, torch.Tensor):
        raise TypeError("Inputs msg and deco_msg must be torch.Tensors.")
    
    if not isinstance(msg_num, int):
        raise TypeError("Input msg_num must be an int.")
    
    if msg_num < 1:
        raise ValueError("msg_num must be at least 1.")
    
    if 'cuda' in str(deco_msg.device):
        deco_msg = deco_msg.cpu()
        msg = msg.cpu()

    if msg_num == 1:
        deco_msg = torch.round(deco_msg)
        correct_pred = torch.sum((deco_msg - msg) == 0, dim=1)
        bit_acc = torch.sum(correct_pred).numpy() / deco_msg.numel()
    else:
        bit_acc = 0.0
        for i in range(msg_num):
            cur_deco_msg = torch.round(deco_msg[:, i, :])
            correct_pred = torch.sum((cur_deco_msg - msg[:, i, :]) == 0, dim=1)
            bit_acc += torch.sum(correct_pred).numpy() / cur_deco_msg.numel()
        bit_acc /= msg_num

    return bit_acc

class MIMData(Dataset):
    """
    A custom dataset class for handling images and secret messages.

    Args:
        data_path (str): Path to the dataset.
        num_message (int, optional): Number of messages. Defaults to 16.
        message_size (int, optional): Size of each message. Defaults to 64*64.
        image_size (tuple, optional): Size of the images. Defaults to (256, 256).
        dataset (str, optional): Dataset type ('coco' or 'div2k'). Defaults to 'coco'.
        roi (str, optional): Region of interest ('fit' or 'crop'). Defaults to 'fit'.
        msg_r (int, optional): Message range. Defaults to 1.
        img_norm (bool, optional): Whether to normalize images. Defaults to False.
    """
    def __init__(
        self,
        data_path: str,
        num_message: int = 16,
        message_size: int = 64*64,
        image_size: tuple = (256, 256),
        dataset: str = 'coco',
        roi: str = 'fit',
        msg_r: int = 1,
        img_norm: bool = False
    ):
        self.data_path = data_path
        self.m_size = message_size
        self.m_num = num_message
        self.im_size = image_size
        self.roi = roi
        self.msg_range = msg_r
        self.image_norm = img_norm
        
        assert dataset in ['coco', 'div2k'], "Invalid DataSet. only support ['coco', 'div2k']."
        assert roi in ['fit', 'crop'], "Invalid Roi Selection. only support ['fit', 'crop']."
        
        if dataset == 'div2k':
            self.files_list = glob(os.path.join(self.data_path, '*.png'))
        elif dataset == 'coco':
            self.files_list = glob(os.path.join(self.data_path, '*.jpg'))
            
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx: int) -> tuple:
        img_cover_path = self.files_list[idx]

        # Load and process the cover image
        img = Image.open(img_cover_path).convert('RGB')
        if self.roi == 'fit':
            img_cover = ImageOps.fit(img, self.im_size)
        elif self.roi == 'crop':
            width, height = img.size   # Get dimensions
            left = (width - self.im_size[0]) / 2
            top = (height - self.im_size[1]) / 2
            right = (width + self.im_size[0]) / 2
            bottom = (height + self.im_size[1]) / 2
            # Crop the center of the image
            img_cover = img.crop((left, top, right, bottom))

        if self.image_norm:
            img_cover = np.array(img_cover, dtype=np.float32) / 255.0
        else:
            img_cover = np.array(img_cover, dtype=np.float32)

        img_cover = self.to_tensor(img_cover)
        
        # Generate secret messages
        messages = np.zeros((self.m_num, self.m_size))
        for n in range(self.m_num):
            message = np.random.randint(low=0, high=self.msg_range + 1, size=self.m_size)
            if self.image_norm:
                messages[n, :] = message / (self.msg_range + 1)
            else:
                messages[n, :] = message
            
        messages = torch.from_numpy(messages).float()
        
        return img_cover, messages

    def __len__(self) -> int:
        return len(self.files_list)

if __name__ == '__main__':
    path = os.path.join('/media/SSD2/stega_all/stega_data/','div2k','train')
    # message r (range) should be 1, 3, 7 for binary, 1, 2, 3-bit per message unit
    msg_range = 3
    dataset = MIMData(data_path=path, num_message=64*64, message_size=16, image_size=(256, 256),
                        dataset='div2k', msg_r=msg_range, img_norm=False)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    data_iterator = iter(data_loader)
    img, msg1 = next(data_iterator)

    # check message shape and value
    print(msg1.shape)
    print(msg1)

    img, msg2 = next(data_iterator)
    # check the message accuracy
    acc = get_message_accuracy(msg1, msg1, msg_num=64*64)
    print(f"Two identity message acc = {acc}")
    acc = get_message_accuracy(msg1, msg2, msg_num=64*64)
    print(f"Random guess of two message in range{msg_range}, acc = {acc}")