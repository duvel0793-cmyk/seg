from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from torch.utils.data import DataLoader, Dataset

import changedetection.datasets.imutils as imutils


def img_loader(path):
    image = np.asarray(imageio.imread(path), dtype=np.float32)
    if image.ndim == 2:
        return image
    if image.shape[-1] == 4:
        return image[:, :, :3]
    return image


class DamageAssessmentDataset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, split="train", max_iters=None):
        self.dataset_path = Path(dataset_path)
        self.data_list = list(data_list)
        self.crop_size = crop_size
        self.split = split

        if max_iters is not None:
            repeat = int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = (self.data_list * repeat)[:max_iters]

    def __len__(self):
        return len(self.data_list)

    def _transform(self, pre_img, post_img, loc_label, clf_label):
        augment = self.split == "train"
        if augment:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(
                pre_img,
                post_img,
                loc_label,
                clf_label,
                self.crop_size,
            )
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img = imutils.affin(pre_img, post_img, translate=0)

        pre_img = np.transpose(imutils.normalize_img(pre_img), (2, 0, 1))
        post_img = np.transpose(imutils.normalize_img(post_img), (2, 0, 1))
        return pre_img, post_img, np.asarray(loc_label), np.asarray(clf_label)

    def __getitem__(self, index):
        basename = self.data_list[index]
        pre_img = img_loader(self.dataset_path / "images" / f"{basename}_pre_disaster.png")
        post_img = img_loader(self.dataset_path / "images" / f"{basename}_post_disaster.png")
        loc_label = img_loader(self.dataset_path / "targets" / f"{basename}_pre_disaster_target.png")
        clf_label = img_loader(self.dataset_path / "targets" / f"{basename}_post_disaster_target.png")

        pre_img, post_img, loc_label, clf_label = self._transform(pre_img, post_img, loc_label, clf_label)
        if self.split == "train":
            clf_label[clf_label == 0] = 255

        return pre_img, post_img, loc_label, clf_label, basename


DamageAssessmentDatset = DamageAssessmentDataset


def read_data_list(list_path):
    with open(list_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def make_data_loader(dataset_path, data_list, crop_size, batch_size, split, max_iters=None, shuffle=None, num_workers=8):
    dataset = DamageAssessmentDataset(dataset_path, data_list, crop_size, split=split, max_iters=max_iters)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=split == "train",
    )
