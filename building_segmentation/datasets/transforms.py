import random

import numpy as np


IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]


def normalize_image(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    image = np.asarray(image, dtype=np.float32)
    output = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        output[..., channel] = (image[..., channel] - mean[channel]) / std[channel]
    return output


def pad_to_crop_size(image, mask, crop_size, mean_rgb=IMAGENET_MEAN, ignore_index=255):
    height, width = mask.shape
    padded_height = max(crop_size, height)
    padded_width = max(crop_size, width)

    padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    padded_mask = np.full((padded_height, padded_width), ignore_index, dtype=np.float32)

    padded_image[:, :, 0] = mean_rgb[0]
    padded_image[:, :, 1] = mean_rgb[1]
    padded_image[:, :, 2] = mean_rgb[2]

    offset_h = int(np.random.randint(padded_height - height + 1))
    offset_w = int(np.random.randint(padded_width - width + 1))

    padded_image[offset_h:offset_h + height, offset_w:offset_w + width] = image
    padded_mask[offset_h:offset_h + height, offset_w:offset_w + width] = mask
    return padded_image, padded_mask


def random_crop(image, mask, crop_size, mean_rgb=IMAGENET_MEAN, ignore_index=255, cat_max_ratio=0.75):
    image, mask = pad_to_crop_size(image, mask, crop_size, mean_rgb=mean_rgb, ignore_index=ignore_index)
    padded_height, padded_width = mask.shape

    crop_h_start = 0
    crop_w_start = 0
    for _ in range(10):
        crop_h_start = random.randrange(0, padded_height - crop_size + 1)
        crop_w_start = random.randrange(0, padded_width - crop_size + 1)
        crop = mask[crop_h_start:crop_h_start + crop_size, crop_w_start:crop_w_start + crop_size]
        values, counts = np.unique(crop, return_counts=True)
        counts = counts[values != ignore_index]
        if counts.size > 1 and np.max(counts) / np.sum(counts) < cat_max_ratio:
            break

    crop_h_end = crop_h_start + crop_size
    crop_w_end = crop_w_start + crop_size
    return (
        image[crop_h_start:crop_h_end, crop_w_start:crop_w_end],
        mask[crop_h_start:crop_h_end, crop_w_start:crop_w_end],
    )


def random_fliplr(image, mask):
    if random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask


def random_flipud(image, mask):
    if random.random() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask


def random_rot90(image, mask):
    k = random.randrange(4)
    if k:
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()
    return image, mask


def train_transform(image, mask, crop_size):
    image, mask = random_crop(image, mask, crop_size)
    image, mask = random_fliplr(image, mask)
    image, mask = random_flipud(image, mask)
    image, mask = random_rot90(image, mask)
    image = np.transpose(normalize_image(image), (2, 0, 1))
    return image.astype(np.float32), np.asarray(mask, dtype=np.int64)


def eval_transform(image, mask):
    image = np.transpose(normalize_image(image), (2, 0, 1))
    return image.astype(np.float32), np.asarray(mask, dtype=np.int64)

