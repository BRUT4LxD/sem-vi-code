from typing import List
import torch
from tqdm import tqdm
import numpy as np
from domain.attack_distance_score import AttackDistanceScore

from domain.attack_eval_score import AttackEvaluationScore
from domain.attack_result import AttackResult
from domain.model_config import ModelConfig

class Metrics:

    @torch.no_grad()
    @staticmethod
    def _accuracy(output, target):
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target).sum().item()
        acc = correct / len(target)
        return acc

    @torch.no_grad()
    @staticmethod
    def _precision(output, target):
        pred = torch.argmax(output, dim=1)
        tp = (pred & target).sum().item()
        fp = (pred & ~target).sum().item()
        x = (tp + fp) if (tp + fp) != 0 else 1
        prec = tp / x
        return prec

    @torch.no_grad()
    @staticmethod
    def _recall(output, target):
        pred = torch.argmax(output, dim=1)
        tp = (pred & target).sum().item()
        fn = (~pred & target).sum().item()
        x = (tp + fn) if (tp + fn) != 0 else 1
        rec = tp / x
        return rec

    @torch.no_grad()
    @staticmethod
    def _f1_score(output, target):
        prec = Metrics._precision(output, target)
        rec = Metrics._recall(output, target)
        x = (prec + rec) if (prec + rec) != 0 else 1
        f1 = 2 * (prec * rec) / x
        return f1


    @torch.no_grad()
    @staticmethod
    def l1_distance(image1: torch.Tensor, image2: torch.Tensor):
        return torch.norm(image1 - image2, p=1)

    @torch.no_grad()
    @staticmethod
    def l2_distance(image1: torch.Tensor, image2: torch.Tensor):
        return torch.norm(image1 - image2, p=2)

    @torch.no_grad()
    @staticmethod
    def linf_distance(image1: torch.Tensor, image2: torch.Tensor):
        return torch.norm(image1 - image2, p=float('inf'))

    @torch.no_grad()
    @staticmethod
    def calculate_attack_power(image1: torch.Tensor, image2: torch.Tensor):
        diff = image1 != image2
        pwr = diff.int().sum()
        return pwr

    @torch.no_grad()
    @staticmethod
    def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader):
        model.eval()
        acc, prec, rec, f1 = 0, 0, 0, 0
        total = 0
        device = next(model.parameters()).device

        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            acc += Metrics._accuracy(outputs, labels)
            prec += Metrics._precision(outputs, labels)
            rec += Metrics._recall(outputs, labels)
            f1 += Metrics._f1_score(outputs, labels)
            total += 1
        acc /= total
        prec /= total
        rec /= total
        f1 /= total
        return acc, prec, rec, f1

    @torch.no_grad()
    @staticmethod
    def calculate_attack_distance_score(attack_results: List[AttackResult]) -> AttackDistanceScore:
        l1 = 0.0
        l2 = 0.0
        lInf = 0.0
        power = 0.0
        n = len(attack_results)

        if n == 0:
            return AttackDistanceScore(l1, l2, lInf, power)

        for result in attack_results:
            l1 += Metrics.l1_distance(result.src_image, result.adv_image)
            l2 += Metrics.l2_distance(result.src_image, result.adv_image)
            lInf += Metrics.linf_distance(result.src_image, result.adv_image)
            power += Metrics.calculate_attack_power(result.src_image, result.adv_image)

        l1 = l1 / n
        l2 = l2 / n
        lInf = lInf / n
        power = power / n

        return AttackDistanceScore(l1.item(), l2.item(), lInf.item(), power.item())

    @torch.no_grad()
    @staticmethod
    def evaluate_attack(attack_results: List[AttackResult], num_classes: int) -> AttackEvaluationScore:
        if len(attack_results) == 0:
            return AttackEvaluationScore(0.0, 0.0, 0.0, 0.0, np.zeros((num_classes, num_classes), dtype=np.int32), AttackDistanceScore(0.0, 0.0, 0.0, 0.0))

        actual = [result.actual for result in attack_results]
        predicted = [result.predicted for result in attack_results]

        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        for a, p in zip(actual, predicted):
            conf_matrix[a][p] += 1

        accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

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

        acc = (accuracy * 100)
        prec = np.mean(np.array(precision) * 100)
        rec = np.mean(np.array(recall) * 100)
        f1 = np.mean(np.array(f1_score) * 100)
        distances = Metrics.calculate_attack_distance_score(attack_results)

        return AttackEvaluationScore(acc, prec, rec, f1, conf_matrix, distances)
