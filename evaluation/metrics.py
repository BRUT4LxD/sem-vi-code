from typing import List
import torch
from tqdm import tqdm
import numpy as np

from domain.attack_eval_score import AttackEvaluationScore
from domain.attack_result import AttackResult


@torch.no_grad()
def l1_distance(image1, image2):
    return torch.norm(image1 - image2, p=1)


@torch.no_grad()
def l2_distance(image1, image2):
    return torch.norm(image1 - image2, p=2)


@torch.no_grad()
def linf_distance(image1, image2):
    return torch.norm(image1 - image2, p=float('inf'))


@torch.no_grad()
def _accuracy(output, target):
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(target).sum().item()
    acc = correct / len(target)
    return acc


@torch.no_grad()
def _precision(output, target):
    pred = torch.argmax(output, dim=1)
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    x = (tp + fp) if (tp + fp) != 0 else 1
    prec = tp / x
    return prec


@torch.no_grad()
def _recall(output, target):
    pred = torch.argmax(output, dim=1)
    tp = (pred & target).sum().item()
    fn = (~pred & target).sum().item()
    x = (tp + fn) if (tp + fn) != 0 else 1
    rec = tp / x
    return rec


@torch.no_grad()
def _f1_score(output, target):
    prec = _precision(output, target)
    rec = _recall(output, target)
    x = (prec + rec) if (prec + rec) != 0 else 1
    f1 = 2 * (prec * rec) / x
    return f1


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    acc, prec, rec, f1 = 0, 0, 0, 0
    total = 0
    device = next(model.parameters()).device

    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        acc += _accuracy(outputs, labels)
        prec += _precision(outputs, labels)
        rec += _recall(outputs, labels)
        f1 += _f1_score(outputs, labels)
        total += 1
    acc /= total
    prec /= total
    rec /= total
    f1 /= total
    return acc, prec, rec, f1


@torch.no_grad()
def evaluate_attack(attack_results: List[AttackResult], num_classes: int) -> AttackEvaluationScore:
    actual = [result.actual for result in attack_results]
    predicted = [result.predicted for result in attack_results]

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    for a, p in zip(actual, predicted):
        conf_matrix[a][p] += 1

    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    # calculate precision, recall, and f1 score for each class
    precision = []
    recall = []
    f1_score = []
    for i in range(num_classes):
        tp = conf_matrix[i][i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp

        precision_i = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall_i = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f1_score_i = 2 * precision_i * recall_i / \
            (precision_i + recall_i) if (precision_i + recall_i) != 0 else 0.0

        precision.append(precision_i)
        recall.append(recall_i)
        f1_score.append(f1_score_i)

    acc = (accuracy * 100).round(2)
    prec = np.mean(np.array(precision) * 100).round(2)
    rec = np.mean(np.array(recall) * 100).round(2)
    f1 = np.mean(np.array(f1_score) * 100).round(2)

    return AttackEvaluationScore(acc, prec, rec, f1, conf_matrix)
