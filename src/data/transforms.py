"""
Data preprocessing and augmentation pipelines for Person Re-Identification.
Includes Random Erasing, Horizontal Flip, Normalization, and Evaluation transforms.
"""

import math
import random
import torch
import torchvision.transforms as T
from PIL import Image

class RandomErasing:
    """
    Random Erasing data augmentation for person Re-ID (Zhong et al.).
    Simulates occlusions by randomly zeroing/erasing rectangular regions.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


def build_reid_transforms(height=256, width=128, is_train=True, use_random_erasing=True):
    """
    Returns standard Person Re-ID torchvision transforms.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        transform_list = [
            T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10, padding_mode="edge"),
            T.RandomCrop((height, width)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        if use_random_erasing:
            transform_list.append(RandomErasing(probability=0.5, mean=mean))
    else:
        transform_list = [
            T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

    return T.Compose(transform_list)
