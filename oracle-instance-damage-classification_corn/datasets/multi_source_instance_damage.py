from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from datasets.bright_instance_damage import BrightOpticalInstanceDamageDataset
from datasets.xbd_oracle_instance_damage import XBDOracleInstanceDamageDataset


class MultiSourceInstanceDataset(Dataset):
    def __init__(
        self,
        datasets_by_source: dict[str, Dataset],
        source_ratio: dict[str, int],
        split_name: str,
    ) -> None:
        super().__init__()
        self.datasets_by_source = {name: dataset for name, dataset in datasets_by_source.items() if len(dataset) > 0}
        if not self.datasets_by_source:
            raise ValueError("No non-empty datasets are available for MultiSourceInstanceDataset.")

        self.source_ratio = {
            source_name: max(1, int(source_ratio.get(source_name, 1)))
            for source_name in self.datasets_by_source.keys()
        }
        self.split_name = split_name
        self.index_map: list[tuple[str, int]] = []
        self.samples: list[dict[str, Any]] = []
        self.class_counts = [0 for _ in range(4)]
        self.source_counts = {source_name: 0 for source_name in self.datasets_by_source.keys()}

        base_unit = max(
            float(len(dataset)) / float(self.source_ratio[source_name])
            for source_name, dataset in self.datasets_by_source.items()
        )
        for source_name, dataset in self.datasets_by_source.items():
            target_count = max(1, int(math.ceil(base_unit * float(self.source_ratio[source_name]))))
            for virtual_index in range(target_count):
                sample_index = virtual_index % len(dataset)
                self.index_map.append((source_name, sample_index))
                self.source_counts[source_name] += 1
                sample_meta = copy.deepcopy(getattr(dataset, "samples", [{}])[sample_index])
                sample_meta["source_name"] = source_name
                self.samples.append(sample_meta)
                label = int(sample_meta["label"])
                if 0 <= label < len(self.class_counts):
                    self.class_counts[label] += 1

        cache_paths = []
        for source_name, dataset in self.datasets_by_source.items():
            cache_path = getattr(dataset, "cache_path", None)
            if cache_path:
                cache_paths.append(f"{source_name}:{cache_path}")
        self.cache_path = " | ".join(cache_paths)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_name, sample_index = self.index_map[index]
        sample = dict(self.datasets_by_source[source_name][sample_index])
        meta = dict(sample.get("meta", {}))
        meta["source_name"] = source_name
        meta["virtual_sample_index"] = int(index)
        meta["source_sample_index"] = int(sample_index)
        sample["meta"] = meta
        sample["source_name"] = source_name
        sample["sample_index"] = int(index)
        return sample


def _build_xbd_dataset(config: dict[str, Any], split_name: str, is_train: bool) -> XBDOracleInstanceDamageDataset:
    list_path = config["data"]["train_list"] if is_train else config["data"]["val_list"]
    return XBDOracleInstanceDamageDataset(
        config=config,
        split_name=split_name,
        list_path=list_path,
        is_train=is_train,
    )


def _build_bright_dataset(config: dict[str, Any], split_name: str, is_train: bool) -> BrightOpticalInstanceDamageDataset:
    return BrightOpticalInstanceDamageDataset(
        config=config,
        split_name=split_name,
        is_train=is_train,
    )


def build_instance_dataset(
    config: dict[str, Any],
    *,
    split_name: str,
    is_train: bool,
) -> Dataset:
    source_mode = str(config["data"]["train_source"] if is_train else config["data"]["eval_source"])
    if source_mode == "xbd_only":
        return _build_xbd_dataset(config, split_name=split_name, is_train=is_train)
    if source_mode == "bright_only":
        return _build_bright_dataset(config, split_name=split_name, is_train=is_train)
    if source_mode == "xbd_bright_mixed":
        datasets = {
            "xbd": _build_xbd_dataset(config, split_name=split_name, is_train=is_train),
            "bright": _build_bright_dataset(config, split_name=split_name, is_train=is_train),
        }
        return MultiSourceInstanceDataset(
            datasets_by_source=datasets,
            source_ratio=dict(config["data"].get("source_ratio", {"xbd": 1, "bright": 1})),
            split_name=split_name,
        )
    raise ValueError(
        "Unsupported source mode '{}'. Expected one of ['xbd_only', 'bright_only', 'xbd_bright_mixed'].".format(
            source_mode
        )
    )
