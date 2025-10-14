import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, Dataset
from attacks.attack_names import AttackNames
from data_eng.transforms import mnist_transformer, cifar_transformer, imagenette_transformer, no_transformer
from domain.model.model_names import ModelNames
from config.imagenette_classes import ImageNetteClasses
import torch
import random
import os

class DatasetType (object):
    CIFAR10 = 'cifar10'
    IMAGENETTE = 'imagenette'
    MNIST = 'mnist'

    @staticmethod
    def is_valid_data_set_type(data_set_type):
        return data_set_type in [DatasetType.CIFAR10, DatasetType.IMAGENETTE, DatasetType.MNIST]

class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

class DatasetLoader:
    @staticmethod
    def get_dataset_by_type(dataset_type: str, transform=None, batch_size=1, train_subset_size=-1, test_subset_size=-1, shuffle=True):
        if not DatasetType.is_valid_data_set_type(dataset_type):
            raise Exception('Invalid dataset type')

        if dataset_type == DatasetType.CIFAR10:
            return load_CIFAR10(transform, batch_size=batch_size, train_subset_size=train_subset_size, test_subset_size=test_subset_size)

        if dataset_type == DatasetType.MNIST:
            return load_MNIST(transform, batch_size=batch_size, train_subset_size=train_subset_size, test_subset_size=test_subset_size)

        if dataset_type == DatasetType.IMAGENETTE:
            return load_imagenette(transform, batch_size=batch_size, train_subset_size=train_subset_size, test_subset_size=test_subset_size, shuffle=shuffle)

    @staticmethod
    def get_attacked_imagenette_dataset(model_name: str, attack_name: str, batch_size=1, shuffle=True, train_base_path=None, test_base_path=None, pt_extension=False):
        if model_name not in ModelNames().all_model_names:
            raise Exception('Invalid model name')

        if attack_name not in AttackNames().all_attack_names:
            raise Exception('Invalid attack name')

        train_base_path = train_base_path if train_base_path is not None else f'./data/attacked_imagenette_train'
        test_base_path = test_base_path if test_base_path is not None else f'./data/attacked_imagenette_test'

        def tensor_transformer(tensor):
            return tensor

        # load attacked imagenette from folder `attacked_imagenette_train/{model_name}/{attack_name}`
        if pt_extension:
            train_data = datasets.DatasetFolder(root=f'{train_base_path}/{model_name}/{attack_name}', loader=torch.load, transform=tensor_transformer, extensions=('.pt',), target_transform=None)
            test_data = datasets.DatasetFolder(root=f'{test_base_path}/{model_name}/{attack_name}', loader=torch.load, transform=tensor_transformer, extensions=('.pt',), target_transform=None)
        else:
            train_data = datasets.ImageFolder(root=f'{train_base_path}/{model_name}/{attack_name}', transform=no_transformer())
            test_data = datasets.ImageFolder(root=f'{test_base_path}/{model_name}/{attack_name}', transform=no_transformer())

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

        return train_loader, test_loader


def _get_data_loaders(train_dataset, test_dataset, batch_size=1, train_subset_size=-1, test_subset_size=-1, shuffle=True, random_seed=42):
    train_loader = None
    test_loader = None

    if train_subset_size > 0:
        # Use random sampling instead of taking first N samples
        random.seed(random_seed)
        train_indices = random.sample(range(len(train_dataset)), min(train_subset_size, len(train_dataset)))
        subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle)

    if test_subset_size > 0:
        # Use random sampling instead of taking first N samples
        random.seed(random_seed)  # Use same seed for reproducibility
        test_indices = random.sample(range(len(test_dataset)), min(test_subset_size, len(test_dataset)))
        subset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    else:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


def load_MNIST(transform=None, path_to_data='./data', batch_size=1, train_subset_size=-1, test_subset_size=-1):
    trans = transform if transform is not None else mnist_transformer()

    train_dataset = datasets.MNIST(
        root=path_to_data, train=True, download=True, transform=trans)
    test_dataset = datasets.MNIST(
        root=path_to_data, train=False, download=True, transform=trans)

    return _get_data_loaders(train_dataset, test_dataset, batch_size, train_subset_size, test_subset_size)


def load_CIFAR10(transform=None, path_to_data='./data', batch_size=1, train_subset_size=-1, test_subset_size=-1):
    trans = transform if transform is not None else cifar_transformer()

    train_dataset = datasets.CIFAR10(
        root=path_to_data, train=True, download=True, transform=trans)
    test_dataset = datasets.CIFAR10(
        root=path_to_data, train=False, download=True, transform=trans)

    return _get_data_loaders(train_dataset, test_dataset, batch_size, train_subset_size, test_subset_size)


def load_imagenette(transform=None, path_to_data='./data/imagenette', batch_size=1, train_subset_size=-1, test_subset_size=-1, shuffle=True):
    trans = transform if transform is not None else imagenette_transformer()

    train_path = path_to_data + '/train'
    test_path = path_to_data + '/val'

    train_dataset = datasets.ImageFolder(root=train_path, transform=trans)
    test_dataset = datasets.ImageFolder(root=test_path, transform=trans)

    return _get_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
        shuffle=shuffle)

def load_empty_dataloader():
    return DataLoader(EmptyDataset(), batch_size=1)


def _load_adversarial_images_from_folder(folder_path: str):
    """
    Load all adversarial images from a folder structure.
    
    Args:
        folder_path: Path to folder containing model/attack/class structure
        
    Returns:
        List of PIL Images
    """
    from PIL import Image
    adversarial_images = []
    
    # Get all model folders
    model_folders = [d for d in os.listdir(folder_path) 
                    if os.path.isdir(os.path.join(folder_path, d))]
    
    print(f"   📊 Found {len(model_folders)} model folders")
    
    for model_folder in model_folders:
        model_path = os.path.join(folder_path, model_folder)
        
        # Get all attack folders for this model
        attack_folders = [d for d in os.listdir(model_path) 
                         if os.path.isdir(os.path.join(model_path, d))]
        
        print(f"      📊 {model_folder}: {len(attack_folders)} attacks")
        
        for attack_folder in attack_folders:
            attack_path = os.path.join(model_path, attack_folder)
            
            # Get all class folders
            class_folders = [d for d in os.listdir(attack_path) 
                           if os.path.isdir(os.path.join(attack_path, d))]
            
            for class_folder in class_folders:
                class_path = os.path.join(attack_path, class_folder)
                
                # Get all adversarial images (skip src_ images)
                image_files = [f for f in os.listdir(class_path) 
                             if f.endswith('.png') and not f.startswith('src_')]
                
                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    try:
                        # Load image
                        image = Image.open(image_path).convert('RGB')
                        adversarial_images.append(image)
                    except Exception as e:
                        print(f"⚠️ Error loading {image_path}: {e}")
                        continue
    
    return adversarial_images


def load_attacked_imagenette(
    attacked_images_folder: str = "data/attacks/imagenette_models",
    clean_train_folder: str = "./data/imagenette/train",
    clean_test_folder: str = "./data/imagenette/val",
    batch_size: int = 32,
    shuffle: bool = True,
    transform=None
):
    """
    Load attacked ImageNette images from all models and attacks for binary classification.
    
    Creates balanced datasets:
    - Train: adversarial images (label=1) from train folder + 2x clean images (label=0) from train folder
    - Test: adversarial images (label=1) from test folder + equal clean images (label=0) from val folder
    
    Folder structure expected:
        attacked_images_folder/
            train/
                model_name/
                    attack_name/
                        class_label/
                            timestamp.png (adversarial)
            test/
                model_name/
                    attack_name/
                        class_label/
                            timestamp.png (adversarial)
    
    Args:
        attacked_images_folder: Root folder containing attacked images with train/test subfolders
        clean_train_folder: Folder containing clean ImageNette training images
        clean_test_folder: Folder containing clean ImageNette validation images
        batch_size: Batch size for dataloaders
        shuffle: Whether to shuffle the data
        transform: Image transformation to apply (default: imagenette_transformer)
    
    Returns:
        Tuple of (train_loader, test_loader)
        - Labels: 0 = clean, 1 = adversarial
    """
    from PIL import Image
    import numpy as np
    
    print("📁 Loading attacked ImageNette dataset for binary classification...")
    print(f"   Attacked images folder: {attacked_images_folder}")
    print(f"   Clean train images folder: {clean_train_folder}")
    print(f"   Clean test images folder: {clean_test_folder}")
    
    trans = transform if transform is not None else imagenette_transformer()
    
    # Storage for train and test data
    train_adversarial = []
    test_adversarial = []
    
    # Collect adversarial images from train and test folders
    if not os.path.exists(attacked_images_folder):
        raise FileNotFoundError(f"Attacked images folder not found: {attacked_images_folder}")
    
    # Load adversarial images from train folder
    train_folder = os.path.join(attacked_images_folder, "train")
    if os.path.exists(train_folder):
        print(f"🔍 Loading adversarial images from train folder...")
        train_adversarial = _load_adversarial_images_from_folder(train_folder)
        print(f"✅ Loaded {len(train_adversarial)} training adversarial images")
    else:
        print(f"⚠️ Train folder not found: {train_folder}")
    
    # Load adversarial images from test folder
    test_folder = os.path.join(attacked_images_folder, "test")
    if os.path.exists(test_folder):
        print(f"🔍 Loading adversarial images from test folder...")
        test_adversarial = _load_adversarial_images_from_folder(test_folder)
        print(f"✅ Loaded {len(test_adversarial)} test adversarial images")
    else:
        print(f"⚠️ Test folder not found: {test_folder}")
    
    if len(train_adversarial) == 0 and len(test_adversarial) == 0:
        raise FileNotFoundError(f"No adversarial images found in {attacked_images_folder}/train or {attacked_images_folder}/test")
    
    # Load clean images from separate train and test folders
    print("📁 Loading clean ImageNette images...")
    
    # Load clean training images
    clean_train_dataset = datasets.ImageFolder(root=clean_train_folder, transform=None)
    # Load clean test images  
    clean_test_dataset = datasets.ImageFolder(root=clean_test_folder, transform=None)
    
    print(f"📊 Available clean images: {len(clean_train_dataset)} train, {len(clean_test_dataset)} test")
    
    # IMPORTANT: For test set, match adversarial images to available clean validation images for 50:50 split
    # Limit test adversarial images to the number of available clean validation images
    if len(test_adversarial) > len(clean_test_dataset):
        print(f"⚠️ Limiting test adversarial images from {len(test_adversarial)} to {len(clean_test_dataset)} to match clean validation images")
        # Randomly sample to match clean validation set size
        import random
        random.seed(42)
        sampled_indices = random.sample(range(len(test_adversarial)), len(clean_test_dataset))
        test_adversarial = [test_adversarial[i] for i in sampled_indices]
    
    # Calculate how many clean images we need
    num_clean_train = len(train_adversarial) * 2  # 2x adversarial for balance
    num_clean_test = len(test_adversarial)  # Equal to adversarial for test (guaranteed to be <= clean_test_dataset size)
    
    if len(clean_train_dataset) < num_clean_train:
        print(f"⚠️ Warning: Not enough clean training images. Need {num_clean_train}, have {len(clean_train_dataset)}")
        num_clean_train = len(clean_train_dataset)
    
    # Sample clean images
    import random
    random.seed(42)
    clean_train_indices = random.sample(range(len(clean_train_dataset)), num_clean_train)
    clean_test_indices = random.sample(range(len(clean_test_dataset)), num_clean_test)
    
    train_clean = [clean_train_dataset[i][0] for i in clean_train_indices]
    test_clean = [clean_test_dataset[i][0] for i in clean_test_indices]
    
    print(f"✅ Loaded {len(train_clean)} clean training images")
    print(f"✅ Loaded {len(test_clean)} clean test images")
    
    # Create datasets with labels
    # Label: 0 = clean, 1 = adversarial
    print("🔨 Creating binary classification datasets...")
    
    train_dataset = []
    for img in train_adversarial:
        if trans:
            img = trans(img)
        train_dataset.append((img, 1))  # Adversarial = 1
    
    for img in train_clean:
        if trans:
            img = trans(img)
        train_dataset.append((img, 0))  # Clean = 0
    
    test_dataset = []
    for img in test_adversarial:
        if trans:
            img = trans(img)
        test_dataset.append((img, 1))  # Adversarial = 1
    
    for img in test_clean:
        if trans:
            img = trans(img)
        test_dataset.append((img, 0))  # Clean = 0
    
    # Shuffle the datasets
    if shuffle:
        random.shuffle(train_dataset)
        random.shuffle(test_dataset)
    
    print(f"📊 Dataset Summary:")
    print(f"   Train: {len(train_dataset)} total ({len(train_adversarial)} adv + {len(train_clean)} clean)")
    print(f"   Test: {len(test_dataset)} total ({len(test_adversarial)} adv + {len(test_clean)} clean)")
    print(f"   Train ratio (clean:adv): {len(train_clean)}:{len(train_adversarial)} = {len(train_clean)/len(train_adversarial):.2f}x")
    print(f"   Test ratio (clean:adv): {len(test_clean)}:{len(test_adversarial)} = {len(test_clean)/len(test_adversarial) if len(test_adversarial) > 0 else 0:.2f}x (should be 1.0 for 50:50)")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Don't shuffle test
    
    print("✅ Binary classification dataloaders created successfully!")
    
    return train_loader, test_loader


def load_attacked_imagenette_for_adversarial_training(
    attacked_images_folder: str = "data/attacks/imagenette_models",
    clean_images_folder: str = "./data/imagenette/train",
    batch_size: int = 32,
    shuffle: bool = True,
    transform=None,
    train_dataset: bool = True
):
    """
    Load attacked and clean ImageNette images for adversarial training.
    
    This creates a mixed dataset of adversarial and clean images for robust model training.
    Unlike noise detection (binary classification), this maintains original ImageNette class labels (0-9).
    
    Folder structure expected:
        attacked_images_folder/
            train/ or test/
                model_name/
                    attack_name/
                        class_label/
                            timestamp.png (adversarial)
    
    Args:
        attacked_images_folder: Root folder containing attacked images with train/test subfolders
        clean_images_folder: Folder containing clean ImageNette images  
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle the data
        transform: Image transformation to apply (default: imagenette_transformer)
        train_dataset: If True, load from train folder; if False, load from test folder
        
    Returns:
        DataLoader with mixed adversarial and clean images (ImageNette class labels 0-9 preserved)
    """
    from PIL import Image
    
    dataset_type = "train" if train_dataset else "test"
    print(f"📁 Loading attacked ImageNette for adversarial training ({dataset_type})...")
    print(f"   Attacked images folder: {attacked_images_folder}/{dataset_type}")
    print(f"   Clean images folder: {clean_images_folder}")
    
    trans = transform if transform is not None else imagenette_transformer()
    
    # Load adversarial images with labels from folder structure
    attacked_folder = os.path.join(attacked_images_folder, dataset_type)
    if not os.path.exists(attacked_folder):
        raise FileNotFoundError(f"Attacked images folder not found: {attacked_folder}")
    
    print(f"🔍 Loading adversarial images with class labels...")
    # Use ImageFolder to automatically get labels from directory structure
    # Note: We need a modified loader that gets labels from the innermost class folder
    adversarial_dataset = _load_adversarial_with_labels(attacked_folder, trans)
    
    print(f"✅ Loaded {len(adversarial_dataset)} adversarial images")
    
    # Load clean images
    print(f"📁 Loading clean ImageNette images...")
    clean_dataset = datasets.ImageFolder(root=clean_images_folder, transform=trans)
    
    # Sample clean images to match adversarial count (1:1 ratio for balanced training)
    num_clean = len(adversarial_dataset)
    
    if len(clean_dataset) < num_clean:
        print(f"⚠️ Warning: Not enough clean images. Need {num_clean}, have {len(clean_dataset)}")
        num_clean = len(clean_dataset)
    
    import random
    random.seed(42)
    clean_indices = random.sample(range(len(clean_dataset)), num_clean)
    
    print(f"✅ Sampled {num_clean} clean images")
    
    # Create combined dataset
    from torch.utils.data import ConcatDataset, Subset
    clean_subset = Subset(clean_dataset, clean_indices)
    combined_dataset = ConcatDataset([adversarial_dataset, clean_subset])
    
    print(f"📊 Dataset Summary:")
    print(f"   Total: {len(combined_dataset)} images ({len(adversarial_dataset)} adv + {num_clean} clean)")
    print(f"   Ratio: 1:1 (adversarial:clean)")
    print(f"   Labels: ImageNette classes (0-9)")
    
    # Create dataloader
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)
    
    print("✅ Adversarial training dataloader created successfully!")
    
    return dataloader


def _load_adversarial_with_labels(folder_path: str, transform=None):
    """
    Load adversarial images with their class labels from folder structure.
    
    Args:
        folder_path: Path to folder containing model/attack/class structure
        transform: Transform to apply to images
        
    Returns:
        Dataset with (image, label) pairs
    """
    from PIL import Image
    from torch.utils.data import Dataset
    
    class AdversarialDataset(Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples
            self.transform = transform
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    samples = []
    
    # Get all model folders
    model_folders = [d for d in os.listdir(folder_path) 
                    if os.path.isdir(os.path.join(folder_path, d))]
    
    for model_folder in model_folders:
        model_path = os.path.join(folder_path, model_folder)
        
        # Get all attack folders for this model
        attack_folders = [d for d in os.listdir(model_path) 
                         if os.path.isdir(os.path.join(model_path, d))]
        
        for attack_folder in attack_folders:
            attack_path = os.path.join(model_path, attack_folder)
            
            # Get all class folders (these are the class labels)
            class_folders = [d for d in os.listdir(attack_path) 
                           if os.path.isdir(os.path.join(attack_path, d))]
            
            for class_folder in class_folders:
                class_path = os.path.join(attack_path, class_folder)
                class_label = int(class_folder)  # Folder name is the class label
                
                # Get all adversarial images (skip src_ images)
                image_files = [f for f in os.listdir(class_path) 
                             if f.endswith('.png') and not f.startswith('src_')]
                
                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    samples.append((image_path, class_label))
    
    return AdversarialDataset(samples, transform)