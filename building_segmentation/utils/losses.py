import torch


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 3:
        probas = probas.unsqueeze(1)
    num_classes = probas.size(1)
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def lovasz_softmax_flat(probas, labels, classes="present"):
    if probas.numel() == 0:
        return probas.sum() * 0.0

    num_classes = probas.size(1)
    losses = []
    class_to_sum = list(range(num_classes)) if classes in ["all", "present"] else classes

    for class_index in class_to_sum:
        foreground = (labels == class_index).float()
        if classes == "present" and foreground.sum() == 0:
            continue
        errors = (foreground - probas[:, class_index]).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, lovasz_grad(foreground_sorted)))

    if not losses:
        return probas.sum() * 0.0
    return sum(losses) / len(losses)


def lovasz_softmax(probas, labels, classes="present", ignore=None):
    probas, labels = flatten_probas(probas, labels, ignore=ignore)
    return lovasz_softmax_flat(probas, labels, classes=classes)

