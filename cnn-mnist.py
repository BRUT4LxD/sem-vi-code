import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

batch_size = 4
num_epochs = 5
learning_rate = 0.001
image_size = 28
MODEL_SAVE_PATH = './models/cnn-mnist.pth'
LOAD_PRETRAINED = False

transform = transforms.Compose(
  [
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
  ]
)

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# cifar10 classes
mnist_classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'); 

# implement conv net 
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16*4*4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
    
  
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16*4*4)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

model = torch.load(MODEL_SAVE_PATH) if LOAD_PRETRAINED else ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(model, criterion, optimizer, train_loader):
  n_total_steps = len(train_loader)
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      images = images.to(device)
      labels = labels.to(device)
      
      outputs = model(images)
      
      loss = criterion(outputs, labels)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      if (i + 1) % 2000 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
      
  print('Finished training')

def save(model):
  torch.save(model, MODEL_SAVE_PATH)

def validation(model, test_loader):
  with torch.no_grad():
    n_correct = 0
    n_samples = 0 
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader: 
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      
      _, predictions = torch.max(outputs, 1)
      n_samples += labels.size(0)
      n_correct += (predictions == labels).sum().item()
      
      for i in range(batch_size):
        label = labels[i]
        pred = predictions[i]
        if (label == pred):
          n_class_correct[label] += 1
        n_class_samples[label] += 1     
      
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}%')
    
    for i in range(10):
      acc = 100.0 * n_class_correct[i] / n_class_samples[i]
      print(f'accuracy of {mnist_classes[i]}: {acc}%')
    
def visualize(test_model):
  batch_cnt = 0
  with torch.no_grad():
    for images, labels in test_loader:
      if batch_cnt > 5:
        break
      batch_cnt += 1
      images = images.to(device)
      labels = labels.to(device)
      outputs = test_model(images)
      _, predictions = torch.max(outputs, 1)
      print('predictions: ', ' '.join('%5s' % mnist_classes[predictions[j]] for j in range(batch_size)))
      
      images = images.cpu().numpy()
      images = np.transpose(images, (0, 2, 3, 1))
      for i in range(batch_size):
        plt.figure()
        plt.imshow(images[i])
        plt.title(f'predicted: {mnist_classes[predictions[i]]}, label: {mnist_classes[labels[i]]}')
        plt.show()
    
saved_model = torch.load(MODEL_SAVE_PATH)
   
visualize(saved_model)

