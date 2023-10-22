from training.mobilenet_v2_imagenette import train_all_mobilenet
from .vgg_imagenette import train_all_vgg
from .efficient_net_imagenette import train_all_efficient_net
from .resnet_imagenette import train_all_resnet
from .densenet_imagenette import train_all_densenet

def train_all_archs_for_imagenette(epochs=50):
    # train_all_vgg(epochs)
    train_all_densenet(epochs)
    train_all_efficient_net(epochs)
    train_all_resnet(epochs)
    train_all_mobilenet(epochs)
