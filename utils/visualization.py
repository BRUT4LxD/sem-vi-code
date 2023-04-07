import torch
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def simple_visualize(test_model, test_loader, batch_size, classes, device='gpu'):
    batch_cnt = 0
    for images, labels in test_loader:
        if batch_cnt > 1:
            break
        batch_cnt += 1
        images = images.to(device)
        labels = labels.to(device)
        outputs = test_model(images)
        _, predictions = torch.max(outputs, 1)
        print('predictions: ', ' '.join('%5s' %
                classes[predictions[j]] for j in range(batch_size)))

        images = images.cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))
        for i in range(batch_size):
            plt.figure()
            plt.imshow(images[i])
            plt.title(
                f'predicted: {classes[predictions[i]]}, label: {classes[labels[i]]}')
            plt.show()
