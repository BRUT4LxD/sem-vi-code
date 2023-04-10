import torch
import torch.nn as nn
from attacks.attack import Attack, AttackResult
from tqdm import tqdm
from typing import List


class FGSM(Attack):

    def __init__(self, model: torch.nn.Module, eps=8/255):
        super().__init__("FGSM", model)
        self.eps = eps

    def attack(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> List[AttackResult]:
        """
        Generates adversarial examples using the FGSM attack and evaluates the model on them.

        Parameters:
        model (nn.Module): the pre-trained PyTorch model to be attacked
        test_loader (DataLoader): the data loader for the test set
        epsilon (float): the magnitude of the perturbation (0 <= epsilon <= 1)
        """
        adv_examples: AttackResult = []
        loss_fn = nn.CrossEntropyLoss()

        print(f'Running FGSM attack for epsilon: {self.eps}')
        for images, labels in tqdm(data_loader):

            images, labels = images.to(self.device), labels.to(self.device)
            images.requires_grad = True
            init_pred = model(images)
            _, init_predicted = torch.max(init_pred, 1)

            # If the initial prediction is wrong, continue
            if init_predicted != labels:
                continue

            loss = loss_fn(init_pred, labels)
            model.zero_grad()
            loss.backward()

            perturbed_images = images + self.eps * images.grad.sign()
            perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()
            outputs = model(perturbed_images)
            _, predicted = torch.max(outputs, 1)

            match_vector = predicted == labels
            for i in range(len(labels)):
                adv_ex = None
                if match_vector[i] == False and len(adv_examples) < self.num_of_adv_examples:
                    adv_ex = perturbed_images[i].squeeze(
                    ).detach().cpu().numpy()

                res = AttackResult(init_predicted[i], predicted[i], adv_ex)
                adv_examples.append(res)

        return adv_examples
