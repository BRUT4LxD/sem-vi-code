import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# Load a pre-trained model
model = torch.load('mnist_model.pt')

# Define the adversarial attack function
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Evaluate the model on the original test set
def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy on original test set: {:.2f}%'.format(accuracy))

# Generate adversarial examples and evaluate the model on them
def attack(model, test_loader, epsilon):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data
        perturbed_images = fgsm_attack(images, epsilon, data_grad)
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Epsilon: {:.2f}\tAccuracy on adversarial test set: {:.2f}%'.format(epsilon, accuracy))
