import torch
from config.imagenet_models import ImageNetModels
from data_eng.dataset_loader import DatasetLoader, DatasetType
from data_eng.io import load_model
from data_eng.pretrained_model_downloader import PretrainedModelDownloader
from domain.model.model_names import ModelNames
from evaluation.metrics import Metrics
from training.transfer.setup_pretraining import SetupPretraining


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = ModelNames().resnet18
raw_resnet_model = ImageNetModels().get_model(model_name)
print("Raw model loaded")
adapted_resnet = SetupPretraining.setup_imagenette(raw_resnet_model)
print("Model adapted")
trained_resnet = load_model(adapted_resnet, f"./models/imagenette/{model_name}.pt")
trained_resnet.to(device)
print("Model loaded")
_, test_loader = DatasetLoader.get_dataset_by_type(dataset_type=DatasetType.IMAGENETTE, batch_size=100, test_subset_size=1000)
print("Test loader loaded")
acc, prec, rec, f1_score = Metrics.evaluate_model_torchmetrics(trained_resnet, test_loader, 10)
print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1_score}")