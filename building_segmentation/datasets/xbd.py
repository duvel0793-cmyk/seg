from pathlib import Path
from typing import Literal, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from building_segmentation.datasets.transforms import eval_transform, train_transform


Split = Literal["train", "val"]
DatasetSample = tuple[torch.Tensor, torch.Tensor, str]


def read_data_list(list_path: str | Path) -> list[str]:
    with open(list_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_image(path: str | Path) -> np.ndarray:
    image = np.asarray(imageio.imread(path), dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return image


def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(imageio.imread(path), dtype=np.int64)


class XBDBuildingDataset(Dataset[DatasetSample]):
    def __init__(
        self,
        dataset_path: str | Path,
        data_list: Sequence[str],
        crop_size: int = 256,
        split: Split = "train",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.data_list = list(data_list)
        self.crop_size = crop_size
        self.split = split

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> DatasetSample:
        basename = self.data_list[index]
        image = load_image(self.dataset_path / "images" / f"{basename}_pre_disaster.png")
        mask = load_mask(self.dataset_path / "targets" / f"{basename}_pre_disaster_target.png")

        if self.split == "train":
            image, mask = train_transform(image, mask, self.crop_size)
        else:
            image, mask = eval_transform(image, mask)

        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            basename,
        )


def make_dataloader(
    dataset_path: str | Path,
    data_list: Sequence[str],
    crop_size: int,
    batch_size: int,
    split: Split,
    num_workers: int = 4,
    shuffle: bool | None = None,
) -> DataLoader[DatasetSample]:
    dataset = XBDBuildingDataset(dataset_path=dataset_path, data_list=data_list, crop_size=crop_size, split=split)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=split == "train",
    )
