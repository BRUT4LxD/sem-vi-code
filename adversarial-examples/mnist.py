import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MNIST_PRETRAINED_PATH = './models/cnn-mnist.pth'
model = torch.load(MNIST_PRETRAINED_PATH)
batch_size = 1
epsilons = [0, .05, .1, .15, .2, .25, .3]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

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

def attack(model, test_loader, epsilon):
    """
    Generates adversarial examples using the FGSM attack and evaluates the model on them.
    
    Parameters:
    model (nn.Module): the pre-trained PyTorch model to be attacked
    test_loader (DataLoader): the data loader for the test set
    epsilon (float): the magnitude of the perturbation (0 <= epsilon <= 1)
    """
    correct = 0
    total = 0
    adv_examples = []
    loader_cnt = 0
    criterion = nn.CrossEntropyLoss()
    
    print(f'Running attack for epsilon: {epsilon}')
    for images, labels in test_loader:
        if loader_cnt % 1000 == 0: 
            print(f'attacked: {loader_cnt}')
        loader_cnt += 1
        
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        init_pred = model(images)
        _, init_predicted = torch.max(init_pred, 1)
        
        if(init_predicted != labels):
            continue
        
        loss = criterion(init_pred, labels)
        model.zero_grad()
        loss.backward()
        
        data_grad = images.grad
        perturbed_images = fgsm_attack(images, epsilon, data_grad)
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        
        match_vector = predicted == labels
        total += labels.size(0)
        correct += match_vector.sum().item()
        for i in range(batch_size):
            if match_vector[i] == False and len(adv_examples) < 5:
                adv_ex = perturbed_images[i].squeeze().detach().cpu().numpy()
                adv_examples.append((init_predicted[i], predicted[i], adv_ex))
        
    
    # Compute the accuracy of the model on the adversarial test set
    accuracy = 100 * correct / total
    print('Epsilon: {:.4f}\tAccuracy on adversarial test set: {:.2f}%'.format(epsilon, accuracy))
    
    return adv_examples, accuracy

def run_attack(model, test_loader):
    accuracies = []
    examples = []
    for eps in epsilons:
        adv_ex, acc = attack(model, test_loader, eps)
        accuracies.append(acc)
        examples.append(adv_ex)
    return examples, accuracies

def plot_accuracy(accuracies):
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 100, step=10))
    plt.xticks(np.arange(0, .35, step=.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

# Plot several examples of adversarial samples at each epsilon
def plot_adversarial_examples(examples, epsilons):
    cnt = 0
    plt.figure(figsize=(7,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
    
model = torch.load(MNIST_PRETRAINED_PATH)    

# attack(model, test_loader, 0.3)

adversarial_examples, accuracies = run_attack(model, test_loader)
  
plot_accuracy(accuracies)

plot_adversarial_examples(adversarial_examples, epsilons)
