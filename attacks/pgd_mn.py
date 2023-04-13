from typing import List
from attacks.attack_mn import Attack_MN, AttackResult
import torch
import torch.nn as nn
from tqdm import tqdm


class PGD_MN(Attack_MN):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)
    """

    def __init__(self, model: torch.nn.Module, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, data_loader: torch.utils.data.DataLoader) -> List[AttackResult]:

        adv_examples: AttackResult = []
        loss_fn = nn.CrossEntropyLoss()

        print(f'Running PGD attack for epsilon: {self.eps}')
        for images, labels in tqdm(data_loader):

            images = images.detach().to(self.device)
            labels = labels.detach().to(self.device)

            init_pred = self.model(images)
            _, init_predicted = torch.max(init_pred, 1)

            # If the initial prediction is wrong, continue
            if init_predicted != labels:
                continue

            if self.targeted:
                target_labels = self.get_target_label(images, labels)

            adv_images = images.clone().detach()

            if self.random_start:
                adv_images = adv_images + \
                    torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.get_logits(adv_images)

                if self.targeted:
                    cost = -loss_fn(outputs, target_labels)
                else:
                    cost = loss_fn(outputs, labels)

                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                adv_images = adv_images + self.alpha*grad.sign()
                delta = torch.clamp(adv_images - images,
                                    min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            outputs = self.get_logits(adv_images)
            self.add_attack_result(
                outputs, labels, adv_examples, adv_images, init_predicted)

        return adv_examples
