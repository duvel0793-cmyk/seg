import numpy as np


class SegmentationMetrics:
    def __init__(self, num_classes=2, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def _generate_matrix(self, gt_mask, pred_mask):
        valid = (gt_mask >= 0) & (gt_mask < self.num_classes) & (gt_mask != self.ignore_index)
        labels = self.num_classes * gt_mask[valid].astype(np.int64) + pred_mask[valid].astype(np.int64)
        counts = np.bincount(labels, minlength=self.num_classes ** 2)
        return counts.reshape(self.num_classes, self.num_classes)

    def update(self, gt_mask, pred_mask):
        if gt_mask.shape != pred_mask.shape:
            raise ValueError(f"Shape mismatch: {gt_mask.shape} vs {pred_mask.shape}")
        self.confusion_matrix += self._generate_matrix(gt_mask, pred_mask)

    def pixel_accuracy(self):
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0
        return float(np.diag(self.confusion_matrix).sum() / total)

    def mean_accuracy(self):
        denom = self.confusion_matrix.sum(axis=1) + 1e-7
        acc = np.diag(self.confusion_matrix) / denom
        return float(np.nanmean(acc))

    def iou_per_class(self):
        denom = (
            self.confusion_matrix.sum(axis=1)
            + self.confusion_matrix.sum(axis=0)
            - np.diag(self.confusion_matrix)
            + 1e-7
        )
        return np.diag(self.confusion_matrix) / denom

    def miou(self):
        return float(np.nanmean(self.iou_per_class()))

    def foreground_iou(self):
        return float(self.iou_per_class()[1])

    def precision(self):
        tp = self.confusion_matrix[1, 1]
        fp = self.confusion_matrix[0, 1]
        return float(tp / (tp + fp + 1e-7))

    def recall(self):
        tp = self.confusion_matrix[1, 1]
        fn = self.confusion_matrix[1, 0]
        return float(tp / (tp + fn + 1e-7))

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return float(2 * precision * recall / (precision + recall + 1e-7))

    def summary(self):
        iou = self.iou_per_class()
        return {
            "pixel_accuracy": round(self.pixel_accuracy(), 6),
            "mean_accuracy": round(self.mean_accuracy(), 6),
            "miou": round(self.miou(), 6),
            "iou_background": round(float(iou[0]), 6),
            "iou_building": round(float(iou[1]), 6),
            "precision": round(self.precision(), 6),
            "recall": round(self.recall(), 6),
            "f1": round(self.f1(), 6),
        }

