class ModelNames:
  resnet18 = 'resnet18'
  resnet50 = 'resnet50'
  resnet101 = 'resnet101'
  resnet152 = 'resnet152'

  densenet121 = 'densenet121'
  densenet161 = 'densenet161'
  densenet169 = 'densenet169'
  densenet201 = 'densenet201'

  vgg11 = 'vgg11'
  vgg13 = 'vgg13'
  vgg16 = 'vgg16'
  vgg19 = 'vgg19'

  mobilenet_v2 = 'mobilenet_v2'
  mobilenet_v3_small = 'mobilenet_v3_small'
  mobilenet_v3_large = 'mobilenet_v3_large'

  efficientnet_b0 = 'efficientnet_b0'
  efficientnet_b1 = 'efficientnet_b1'
  efficientnet_b2 = 'efficientnet_b2'
  efficientnet_b3 = 'efficientnet_b3'
  efficientnet_b4 = 'efficientnet_b4'
  efficientnet_b5 = 'efficientnet_b5'
  efficientnet_b6 = 'efficientnet_b6'
  efficientnet_b7 = 'efficientnet_b7'

  def __init__(self):

    self.resnet_model_names = [
      ModelNames.resnet18,
      ModelNames.resnet50,
      ModelNames.resnet101,
      ModelNames.resnet152,
    ]

    self.densenet_model_names = [
      ModelNames.densenet121,
      ModelNames.densenet161,
      ModelNames.densenet169,
      ModelNames.densenet201
    ]

    self.efficientnet_model_names = [
      ModelNames.efficientnet_b0,
      ModelNames.efficientnet_b1,
      ModelNames.efficientnet_b2,
      ModelNames.efficientnet_b3,
      ModelNames.efficientnet_b4,
      ModelNames.efficientnet_b5,
      ModelNames.efficientnet_b6,
      ModelNames.efficientnet_b7,
    ]

    self.mobilenet_model_names = [
      ModelNames.mobilenet_v2,
      ModelNames.mobilenet_v3_small,
      ModelNames.mobilenet_v3_large,
    ]

    self.vgg_model_names = [
      ModelNames.vgg11,
      ModelNames.vgg13,
      ModelNames.vgg16,
      ModelNames.vgg19,
    ]

    self.all_model_names = self.resnet_model_names + self.densenet_model_names + self.efficientnet_model_names + self.mobilenet_model_names + self.vgg_model_names

