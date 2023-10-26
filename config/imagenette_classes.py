class ImageNetteClasses:
  
  @staticmethod
  def get_classes() -> list:
    return ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

  @staticmethod
  def get_id_to_classes_map() -> dict:
    return {
    'n01440764': 'tench, Tinca tinca',
    'n02102040': 'English springer, English springer spaniel',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw, chainsaw',
    'n03028079': 'church, church building',
    'n03394916': 'French horn, horn',
    'n03417042': 'garbage truck, dustcart',
    'n03425413': 'gas pump, gasoline pump, petrol pump, island dispenser',
    'n03445777': 'golf ball',
    'n03888257': 'parachute, chute'
    }

  @staticmethod
  def get_imagenette_to_imagenet_map_by_index() -> dict:
    return {
      0: 0,
      1: 217,
      2: 482,
      3: 491,
      4: 497,
      5: 566,
      6: 569,
      7: 571,
      8: 574,
      9: 701
    }