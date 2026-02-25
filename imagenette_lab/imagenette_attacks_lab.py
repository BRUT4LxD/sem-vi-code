from attacks.attack_names import AttackNames
from data_eng.dataset_loader import load_imagenette
from domain.model.model_names import ModelNames
from imagenette_lab.imagenette_direct_attacks import ImageNetteDirectAttacks


if __name__ == "__main__":
    model_names = [
        ModelNames().resnet18,
        ModelNames().vgg16,
        ModelNames().densenet121,
        ModelNames().mobilenet_v2,
        ModelNames().efficientnet_b0
    ]

    attack_names = AttackNames().all_attack_names
    _, test_loader = load_imagenette(batch_size=3, test_subset_size=1000)
    direct_attacks = ImageNetteDirectAttacks(device='auto')
    results = direct_attacks.run_attacks_on_models(
      model_names=model_names,
      attack_names=attack_names,
      data_loader=test_loader,
      results_folder=f"./results/imagenette_attacks_lab"
    )