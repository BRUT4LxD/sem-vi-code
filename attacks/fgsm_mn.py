import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from attacks.attack_mn import Attack_MN
from domain.attack_result import AttackResult


class FGSM_MN(Attack_MN):

    def __init__(self, model: torch.nn.Module, eps=8/255):
        super().__init__("FGSM", model)
        self.eps = eps

    def forward(self, data_loader: torch.utils.data.DataLoader) -> List[AttackResult]:
        """
        Generates adversarial examples using the FGSM attack and evaluates the model on them.

        Parameters:
        test_loader (DataLoader): the data loader for the test set
        epsilon (float): the magnitude of the perturbation (0 <= epsilon <= 1)
        """
        adv_examples: AttackResult = []
        loss_fn = nn.CrossEntropyLoss()

        print(f'Running FGSM attack for epsilon: {self.eps}')
        for images, labels in tqdm(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            images.requires_grad = True

            if self.targeted:
                target_labels = self.get_target_label(images, labels)

            init_pred = self.model(images)
            _, init_predicted = torch.max(init_pred, 1)

            # If the initial prediction is wrong, continue
            if init_predicted != labels:
                continue

            if self.targeted:
                loss = -loss_fn(init_pred, target_labels)
            else:
                loss = loss_fn(init_pred, labels)

            self.model.zero_grad()
            loss.backward()

            perturbed_images = images + self.eps * images.grad.sign()
            perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()
            outputs = self.get_logits(perturbed_images)

            self.add_attack_result(
                outputs, labels, adv_examples, perturbed_images, init_predicted)

        return adv_examples
