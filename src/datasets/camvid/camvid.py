from math import ceil
from pathlib import Path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import _T_co

from datasets.ds_path import datasets_path
from .meta import classes, split_dirs, class2rgb
import albumentations as A

camvid_path = datasets_path.joinpath("camvid")


class CamVid(Dataset):
    """
    classes: class name, a string
    label: unique integer represent the class.
    mask: rgb image, all colors are the rgb represent of classes
    target: image of labels
    """

    rgb2class = {v: k for k, v in class2rgb.items()}
    class2label = {c: i for i, c in enumerate(classes)}
    label2class = {i: c for i, c in enumerate(classes)}
    num_classes = len(classes)
    
    def __init__(
        self,
        root: Path = camvid_path,
        split: Literal["train", "test", "val"] = "train",
        load_size: tuple[int, int] | None = (512, 512),
        augmentation: A.Compose | None = None,
    ):
        """

        Args:
            root:
            split:
            load_size: Image will be resized by this value after loaded intermediately.
            augmentation: augmentation image after loaded and resized.
        """
        self.root = root
        self.dirs = split_dirs[split]
        self.data_dir = self.root.joinpath(self.dirs[0])
        self.labels_dir = self.root.joinpath(self.dirs[1])

        self.data_files: list[str] = []
        self.label_files: list[str] = []

        for file in self.data_dir.iterdir():
            if file.is_file():
                self.data_files.append(file.as_posix())
                fn, ext = file.name.split(".")
                self.label_files.append(
                    self.labels_dir.joinpath(f"{fn}_L.{ext}").as_posix()
                )  # see images folder for details

        self.augmentation = augmentation
        self.load_size = load_size

    @classmethod
    def mask2target(cls, mask: np.ndarray) -> np.ndarray:
        assert mask.shape[2] == 3

        target = np.zeros(mask.shape[:2], dtype=np.uint8)
        for rgb, c in cls.rgb2class.items():
            matches = np.all(mask == rgb, axis=-1)
            target[matches] = cls.class2label[c]
        return target

    @classmethod
    def target2mask(cls, target: np.ndarray) -> np.ndarray:
        target = target.squeeze()
        mask = np.zeros(target.shape + (3,), dtype=np.uint8)
        for rgb, c in cls.rgb2class.items():
            label = cls.class2label[c]
            matches = target == label
            mask[matches] = rgb
        return mask

    def load_image(self, file: str):
        image = cv2.imread(file)
        if self.load_size:
            image = cv2.resize(image, self.load_size)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> _T_co:
        image = self.load_image(self.data_files[index])
        mask = self.load_image(self.label_files[index])
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]
        
        return image, self.mask2target(mask).astype(np.long)

    def __len__(self):
        return len(self.data_files)

    def show_pairs(
        self,
        n=25,
        n_per_row: int = 2,
        figsize=(20, 10),
        *,
        images_pairs: list[tuple[np.ndarray, np.ndarray]] |None = None
    ):
        """
        
        Args:
            n:
            n_per_row:
            figsize:
            images_pairs: a pair of (image, target) # Note: receive target, not mask.

        Returns:

        """
        images_pair = images_pairs or []
        num_images = min(n, len(self))
        total_images = num_images+len(images_pair)
        nrows = ceil(total_images/ n_per_row)
        fig, axes = plt.subplots(nrows, n_per_row, figsize=figsize)
        axes = axes.flatten()

        def img_pair():
            for r, t in images_pair:
                assert isinstance(r,np.ndarray) and isinstance(t,np.ndarray)
                m = self.target2mask(t)
                assert r.shape==m.shape
                yield r,m
            for i in range(num_images):
                r, target = self[i]
                m = self.target2mask(target)
                yield r, m
        
        for i, (ri, mi) in enumerate(img_pair()):
            raw_np = (
                np.array(ri)
                if not isinstance(ri, np.ndarray)
                else ri
            )
            mask_np = (
                np.array(mi)
                if not isinstance(mi, np.ndarray)
                else mi
            )
            concatenated_image = np.hstack((raw_np, mask_np))

            ax = axes[i]
            ax.imshow(concatenated_image)
            ax.set_title(f"Size {raw_np.shape}")
            ax.axis("off")  # Hide axes ticks and labels

        # Turn of the redundant
        for j in range(total_images, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            "Dataset Visualization: Raw Image (Left) & Mask (Right)", fontsize=16
        )

        plt.tight_layout()
        plt.show()
