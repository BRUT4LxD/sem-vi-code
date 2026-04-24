class ModelNames:
  resnet18 = 'resnet18'
  resnet34 = 'resnet34'
  resnet50 = 'resnet50'
  resnet101 = 'resnet101'
  resnet152 = 'resnet152'

  densenet121 = 'densenet121'
  densenet161 = 'densenet161'
  densenet169 = 'densenet169'
  densenet201 = 'densenet201'

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
  efficientnet_v2_s = 'efficientnet_v2_s'
  efficientnet_v2_m = 'efficientnet_v2_m'
  efficientnet_v2_l = 'efficientnet_v2_l'

  inception_v3 = 'inception_v3'
  maxvit_t = 'maxvit_t'

  swin_t = 'swin_t'
  swin_s = 'swin_s'
  swin_b = 'swin_b'
  swin_v2_t = 'swin_v2_t'
  swin_v2_s = 'swin_v2_s'
  swin_v2_b = 'swin_v2_b'

  vit_b_16 = 'vit_b_16'
  vit_b_32 = 'vit_b_32'
  vit_l_16 = 'vit_l_16'
  vit_l_32 = 'vit_l_32'

  def __init__(self):

    self.resnet_model_names = [
      ModelNames.resnet18,
      ModelNames.resnet34,
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
      ModelNames.efficientnet_v2_s,
      ModelNames.efficientnet_v2_m,
      ModelNames.efficientnet_v2_l,
    ]

    self.mobilenet_model_names = [
      ModelNames.mobilenet_v2,
      ModelNames.mobilenet_v3_small,
      ModelNames.mobilenet_v3_large,
    ]

    self.vit_model_names = [ModelNames.vit_b_16, ModelNames.vit_b_32, ModelNames.vit_l_16, ModelNames.vit_l_32]
    self.swin_model_names = [
      ModelNames.swin_t, ModelNames.swin_s, ModelNames.swin_b,
      ModelNames.swin_v2_t, ModelNames.swin_v2_s, ModelNames.swin_v2_b,
    ]

    self.all_model_names = self.resnet_model_names + self.densenet_model_names + self.efficientnet_model_names + self.mobilenet_model_names

