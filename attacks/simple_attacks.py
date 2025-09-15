from collections import defaultdict
import datetime
import os
import time
from attacks.attack import Attack
from attacks.attack_factory import AttackFactory
from config.imagenet_classes import ImageNetClasses
from config.imagenette_classes import ImageNetteClasses
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from data_eng.dataset_loader import DatasetLoader, DatasetType
from data_eng.transforms import imagenette_transformer
from domain.attack.attack_distance_score import AttackDistanceScore
from domain.attack.attack_eval_score import AttackEvaluationScore
from domain.attack.attack_result import AttackResult, AttackedImageResult
from evaluation.metrics import Metrics
from evaluation.visualization import attack_visualization
from shared.model_utils import ModelUtils
from typing import List
from domain.attack.attack_eval_score import AttackEvaluationScore
from domain.attack.attack_result import AttackResult
from domain.attack.multiattack_result import MultiattackResult

from shared.model_utils import ModelUtils
from config.imagenet_models import ImageNetModels



class SimpleAttacks:

    @staticmethod
    def single_attack(attack: Attack, test_loader: DataLoader, device='cuda', iterations: int = 1, num_classes: int = None, visualize: bool = False):
        model: torch.nn.Module = attack.model

        if num_classes is None:
            last_layer = list(model.children())[-1]
            last_layer_outputs = list(last_layer.parameters())[-1]
            num_classes = len(last_layer_outputs)

        it = 0
        for images, labels in test_loader:
            if it >= iterations:
                break
            images = images.to(device)
            labels = labels.to(device)
            start = time.time()
            adv_images = attack(images, labels)
            attack_res = AttackResult.create_from_adv_image(
                model, adv_images, images, labels, attack.model_name, attack.attack)
            end = time.time()
            ev = Metrics.evaluate_attack_score(attack_res, num_classes)
            print('{}: samples: {}, {} ({} ms)'.format(attack.attack, labels.shape[0], ev,
                int((end-start)*1000)))
            
            if visualize:
                attack_visualization(
                    source_image=images[0],
                    attacked_image=adv_images[0],
                    model_name=attack.model_name,
                    attack_name=attack.attack,
                    true_label=labels[0].item(),
                    predicted_label=attack_res[0].predicted,
                    distance_score=ev.distance_score
                )
            
            it += 1

            del adv_images, attack_res

    @staticmethod
    def get_attacked_imagenette_images(
        attack_name: str,
        model_name: str,
        num_of_images=20,
        batch_size=4,
        use_test_set=True,
        model: torch.nn.Module = None) -> List[AttackedImageResult]:
        """
        Returns attacked imagenette images.
        :param attack_name: attack name
        :param model_name: model name
        :param num_of_images: number of images to attack
        :param batch_size: batch size
        :param use_test_set: if True, test set is used, otherwise train set is used
        :param model: model
        :return: attacked imagenette images
        """

        imagenette_to_imagenet_index_map = ImageNetteClasses.get_imagenette_to_imagenet_map_by_index()
        imagenet_to_imagenette_index_map = ImageNetClasses.get_imagenet_to_imagenette_index_map()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model if model is not None else ImageNetModels.get_model(model_name)
        model.to(device)
        model.eval()
        attack = AttackFactory.get_attack(attack_name, model)
        train_loader, test_loader = DatasetLoader.get_dataset_by_type(dataset_type=DatasetType.IMAGENETTE, batch_size=batch_size)
        data_loader = test_loader if use_test_set else train_loader

        attack_results = []
        for images, labels in data_loader:
            if len(attack_results) >= num_of_images:
                break

            for original, mapped in imagenette_to_imagenet_index_map.items():
                mask = labels == original
                labels[mask] = mapped

            images, labels = images.to(device), labels.to(device)

            images, labels = ModelUtils.remove_missclassified(model, images, labels, device=device)

            if labels.numel() == 0:
                continue

            adv_images = attack(images, labels)
            outputs = attack.model(adv_images)
            _, predicted_labels = torch.max(outputs.data, 1)
            for i in range(len(adv_images)):
                if len(attack_results) >= num_of_images:
                    break

                if predicted_labels[i] == labels[i]:
                    continue

                imagenette_label_tensor = torch.tensor(imagenet_to_imagenette_index_map[labels[i].item()])
                img_detached, adv_img_detached = images[i].detach().clone(), adv_images[i].detach().clone()
                l1 = Metrics.l1_distance(img_detached, adv_img_detached)
                l2 = Metrics.l2_distance(img_detached, adv_img_detached)
                lInf = Metrics.linf_distance(img_detached, adv_img_detached)
                power = Metrics.attack_power_mse(img_detached, adv_img_detached)
                distance_score = AttackDistanceScore(l1.item(), l2.item(), lInf.item(), power.item())
                attack_results.append(AttackedImageResult(adv_img_detached, imagenette_label_tensor, distance_score))

        return attack_results

    @staticmethod
    def attack_images_imagenette(attack: Attack, data_loader: DataLoader, save_results: bool = False):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # map for storing the counts of successful attacks for each class
        successful_attacks = defaultdict(int)

        num_classes = len(ImageNetteClasses.get_classes())

        attack_results = []

        for images, labels in tqdm(data_loader):

            # 1. Remove missclassified images
            # 2. Attack the images
            # 3. Test the attacked images
            # 4. Collect the evaluation scores

            images, labels = ModelUtils.remove_missclassified_imagenette(attack.model, images.to(device), labels.to(device))
            if labels.numel() == 0:
                continue

            adv_images = attack(images, labels)
            outputs = attack.model(adv_images.to(device))
            _, predicted_labels = torch.max(outputs.data, 1)
            for i in range(len(adv_images)):
                label = labels[i].item()
                predicted_label = predicted_labels[i].item()

                att_result = AttackResult(
                    actual=label,
                    predicted=predicted_label,
                    adv_image=adv_images[i],
                    src_image=images[i],
                    model_name=attack.model_name,
                    attack_name=attack.attack)

                attack_results.append(att_result)
        
        # 5. Calculate global evaluation scores
        ev = Metrics.evaluate_attack_score(attack_results, num_classes)

        if save_results:
            save_path = f'{attack.model_name}_{attack.attack}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            with open(save_path, 'w') as file:
                file.write(str(ev))
        
        return ev


    @staticmethod
    def multiattack(
        attacks: List[Attack],
        data_loader: DataLoader,
        device='cuda',
        print_results=True,
        iterations=10,
        save_folder_path=None,
        is_imagenette_model=False) -> MultiattackResult:
        """
        Run multiple attacks on the same model and dataset and return the results.
        :param attacks: List of attacks to run
        :param test_loader: DataLoader for the dataset
        :param device: Device to run the attacks on
        :param print_results: Whether to print the results
        :param iterations: Number of iterations to run each attack
        :param save_results: Whether to save the results
        :return: MultiattackResult
        """
        evaluation_scores: List['AttackEvaluationScore'] = []
        attack_results: List[List['AttackResult']] = []

        labels_mapper = ImageNetClasses.get_imagenet_to_imagenette_index_map() if is_imagenette_model else None

        pbar = tqdm(total=len(attacks) * iterations)
        it = -1
        for attack in attacks:
            it += 1
            attack_results.append([])
            torch.cuda.empty_cache()
            pbar.set_description(f'({attack.model_name}) {attack.attack:12s}')
            num_classes = ModelUtils.get_model_num_classes(attack.model)
            att_time = 0

            itx = 0
            try:
                for images, labels in data_loader:

                    pbar.update(1)
                    if itx >= iterations:
                        break
                    itx += 1
                    images, labels = images.to(device), labels.to(device)
                    images, labels = ModelUtils.remove_missclassified(
                        attack.model, images, labels, device, labels_mapper)

                    if labels.numel() == 0:
                        continue

                    start = time.time()
                    adv_images = attack(images, labels)
                    end = time.time()
                    att_time += end-start
                    attack_res = AttackResult.create_from_adv_image(attack.model, adv_images, images, labels, attack.model_name, attack.attack, labels_mapper)
                    attack_results[it].extend(attack_res)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(
                        f'WARNING: ran out of memory, skipping attack for model: {attack.model_name} and attack: {attack.attack}')
                    continue

            ev = Metrics.evaluate_attack_score(attack_results[it], num_classes)
            ev.set_after_attack(attack.attack, att_time, len(attack_results[it]))
            evaluation_scores.append(ev)

        if print_results:
            print("\n\n************* RESULTS *************\n\n")
            print(f'SAMPLES: {ev.n_samples}\n')
            for ev in evaluation_scores:
                print(ev)

        if save_folder_path is not None:
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            with open(f'{save_folder_path}/{attack.model_name}.txt', 'w') as file:
                for eval_score in evaluation_scores:
                    file.write(str(eval_score) + '\n')
        return MultiattackResult(evaluation_scores, attack_results)
