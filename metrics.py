import torch;

# l1 distance between two images
def l1_distance(image1, image2):
  return torch.norm(image1 - image2, p=1)

# l2 distance between two images
def l2_distance(image1, image2):
  return torch.norm(image1 - image2, p=2)

# linf distance between two images
def linf_distance(image1, image2):
  return torch.norm(image1 - image2, p=float('inf'))  
