from attacks.attack import Attack
from attacks.simple_attacks import SimpleAttacks
from attacks.transferability import Transferability
import torch
from torch.utils.data.dataloader import DataLoader
from typing import List
from domain.attack.multiattack_result import MultiattackResult

class SystemUnderAttack:

    @staticmethod
    def simple_attack(attack: Attack, test_loader: DataLoader, device='cuda', iterations: int = 1):
        SimpleAttacks.single_attack(attack, test_loader, device, iterations)

    @staticmethod
    def attack_images(
            attack: Attack,
            model_name: str,
            data_loader: DataLoader,
            images_to_attack=20,
            save_results=False,
            save_base_path="./data/attacked_imagenette"):

        return SimpleAttacks.attack_images(attack, model_name, data_loader, images_to_attack, save_results, save_base_path)

    @staticmethod
    def multiattack(attacks: List[Attack], test_loader: DataLoader, device='cuda', print_results=True, iterations=10, save_results=False) -> MultiattackResult:
        return SimpleAttacks.multiattack(attacks, test_loader, device, print_results, iterations, save_results)

    @staticmethod
    def transferability_attack(
            attacked_model: torch.nn.Module,
            trans_models: List['torch.nn.Module'],
            attacks: List['Attack'],
            data_loader: DataLoader,
            iterations=10,
            save_results=False,
            print_results=True,
            device='cuda') -> dict:

        return Transferability.transferability_attack(attacked_model, trans_models, attacks, data_loader, iterations, save_results, print_results, device)