from utils.dataset_loader import load_imagenette
from torchvision import transforms
import pathlib

# list all directories in the current directory
dirs = [x.name for x in pathlib.Path(
    './data/imagenette/train').iterdir() if x.is_dir()]

print(dirs)
