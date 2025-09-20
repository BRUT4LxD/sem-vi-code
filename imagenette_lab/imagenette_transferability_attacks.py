
from attacks.attack_factory import AttackFactory
from data_eng.io import load_model_imagenette


def imagenette_transferability_model2model(model_names, attack_names, data_loader, images_per_attack=50):
  for model_name in model_names:
    model_path = f"models/imagenette/{model_name}_advanced.pt"
    model = load_model_imagenette(model_path, model_name)

    attacked_images_counts = {}
    for attack_name in attack_names:
      attack = AttackFactory.get_attack(attack_name, model)
      attacked_images_counts[attack_name] = 0

      for(images, labels) in data_loader:
        adv_images = attack(images, labels)

        # count only the images that are misclassified
        
        attacked_images_counts[attack_name] += len(adv_images)

