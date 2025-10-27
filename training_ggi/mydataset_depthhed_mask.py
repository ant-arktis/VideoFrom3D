import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gc
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import cv2

# import decord  # isort:skip
import os
from os.path import join
from einops import rearrange
from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop, ToTensor, ToPILImage
import random
from glob import glob
from PIL import Image
import json
from controlnet_aux import HEDdetector, MidasDetector
from skimage.transform import resize

import sys

sys.path.append('.')

# decord.bridge.set_bridge("torch")

logger = get_logger(__name__)


class CollateFunction:

    def __init__(self, weight_dtype: torch.dtype) -> None:
        self.weight_dtype = weight_dtype

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data]

        videos = [torch.from_numpy(x["video"]) for x in data]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        out = {
            "videos": videos,
            "prompts": prompts,
        }

        conds = [torch.from_numpy(x["cond"]) for x in data]
        conds = torch.stack(conds).to(dtype=self.weight_dtype, non_blocking=True)
        out['conds'] = conds

        masks = [torch.from_numpy(x["mask"]) for x in data]
        masks = torch.stack(masks).to(dtype=self.weight_dtype, non_blocking=True)
        out['masks'] = masks

        return out


# Inspired by the VideoMAE repository.
def normalize_input(
        item,
        mean=[0.5, 0.5, 0.5],  # Imagenet [0.485, 0.456, 0.406]
        std=[0.5, 0.5, 0.5],  # Imagenet [0.229, 0.224, 0.225]
        use_simple_norm=False):
    if item.dtype == torch.uint8 and not use_simple_norm:
        item = rearrange(item, 'f c h w -> f h w c')

        item = item.float() / 255.0
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        out = rearrange((item - mean) / std, 'f h w c -> f c h w')

        return out
    else:

        item = rearrange(item, 'f c h w -> f h w c')
        return rearrange(item / 127.5 - 1.0, 'f h w c -> f c h w')


class MultiviewDataset(Dataset):

    def __init__(
        self,
        path_root,
        path_seed='seeds.txt',
        width: int = 480,
        height: int = 720,
        n_sample_frames: int = 16,
        tokenizer=None,
        valid_only=False,
        frame_intervals=[1],
        device='cuda',
        mask_thres_up=[0.4, 0.5],
        mask_thres_down=[0.1, 0.2],
    ):
        self.path_root = path_root
        self.path_seed = path_seed
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.valid_only = valid_only
        self.frame_intervals = frame_intervals
        self.device = device
        self.mask_thres_up = mask_thres_up
        self.mask_thres_down = mask_thres_down

        with open(join(self.path_root, 'captions.json')) as f:
            self.captions = json.load(f)

        with open(join(self.path_root, self.path_seed), 'r') as f:
            self.seeds = f.read().splitlines()

        self.model_hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        self.model_hed.netNetwork.requires_grad_(False)
        self.model_depth = MidasDetector.from_pretrained("lllyasviel/Annotators")
        self.model_depth.model.requires_grad_(False)

    @staticmethod
    def __getname__():
        return 'folder'

    def __len__(self) -> int:
        return len(self.seeds)

    def _select_target(self, path):
        paths = sorted(glob(join(path, 'images*/*.*')))
        try:
            assert len(paths) >= self.n_sample_frames
        except Exception as e:
            print(path)
            raise e

        interval = random.sample(self.frame_intervals, 1)[0]
        paths = [Path(paths[i * interval]) for i in range(self.n_sample_frames)]
        return paths

    def _load_prompt(self, path_root):
        id = path_root.split('/')[-1]
        prompt = self.captions[id]
        return prompt

    def _load_image(self, paths):

        ims = [Image.open(path) for path in paths]
        xs = torch.stack([ToTensor()(im) for im in ims])

        # ADAPTIVE RESIZE --------------------------------------------------#
        *_, h, w = xs.shape
        coef = 1.1
        h_c, w_c = self.height, self.width
        r1 = w_c / w
        r2 = h_c / h
        r = r1 if r1 > r2 else r2
        h_resize = int(r * h * coef)
        w_resize = int(r * w * coef)
        # ------------------------------------------------------------------#

        len_min = min(self.height, self.width)
        tr = Compose([
            Resize((h_resize, w_resize)),
            RandomCrop((self.height, self.width)),
        ])
        xs = tr(xs)
        ims = [ToPILImage()(x) for x in xs]

        xs = []
        for x in ims:
            x = np.array(x)
            x = rearrange(x, 'h w c -> c h w')
            x = x.astype('float32')
            xs.append(x)
        xs = np.stack(xs)
        return xs

    def _load_cond(self, images):
        *_, h, w = images.shape
        conds = []
        masks = []

        self.model_depth.to(self.device)
        self.model_hed.to(self.device)

        for x in images:
            x = rearrange(x, 'c h w -> h w c').astype('uint8')
            img = Image.fromarray(x)

            img = self.model_depth(img)
            img = img.resize((w, h))

            # hack ----------------------------------------------------------------------------- #
            depth = np.array(img) / 255

            thres_up = random.sample(self.mask_thres_up, 1)[0]
            thres_down = random.sample(self.mask_thres_down, 1)[0]
            mask = ((thres_up > depth) & (depth > thres_down)).astype('uint8')
            # ---------------------------------------------------------------------------------- #

            img = self.model_hed(img)
            img = img.resize((w, h))

            x = np.array(img)
            conds.append(x)
            masks.append(mask)

        conds = np.stack(conds)
        conds = rearrange(conds, 'f h w c -> f c h w')

        masks = np.stack(masks)
        masks = rearrange(masks, 'f h w c -> f c h w')

        if random.random() < 0.1:
            masks = np.ones_like(masks)

        self.model_depth.to('cpu')
        self.model_hed.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

        return conds, masks

    def __getitem__(self, index):

        path_root = join(self.path_root, self.seeds[index])

        paths = self._select_target(path_root)
        images = self._load_image(paths)
        prompt = self._load_prompt(path_root)
        out = {
            "video": normalize_input(images),
            "prompt": prompt,
        }

        cond, mask = self._load_cond(images)
        out['cond'] = normalize_input(cond)
        out['mask'] = mask

        return out
