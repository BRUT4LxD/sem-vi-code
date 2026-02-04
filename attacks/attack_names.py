
class AttackNames():

  # White-box attacks

  APGD = "APGD"
  APGDT = "APGDT"
  BIM = "BIM"
  CW = "CW"
  DeepFool = "DeepFool"
  DIFGSM = "DIFGSM"
  EADEN = "EADEN"
  EADL1 = "EADL1"
  EOTPGD = "EOTPGD"
  FAB = "FAB"
  FFGSM = "FFGSM"
  FGSM = "FGSM"
  GN = "GN"
  Jitter = "Jitter"
  JSMA = "JSMA"
  MIFGSM = "MIFGSM"
  NIFGSM = "NIFGSM"
  OnePixel = "OnePixel"
  PGD = "PGD"
  PGDL2 = "PGDL2"
  PGDRS = "PGDRS"
  PGDRSL2 = "PGDRSL2"
  RFGSM = "RFGSM"
  SINIFGSM = "SINIFGSM"
  SparseFool = "SparseFool"
  SPSA = "SPSA"
  TIFGSM = "TIFGSM"
  TPGD = "TPGD"
  UPGD = "UPGD"
  VMIFGSM = "VMIFGSM"
  VNIFGSM = "VNIFGSM"

  # Black-box attacks

  Pixle = "Pixle"
  Square = "Square"

  def __init__(self) -> None:
    self.white_box_attacks = [
        self.APGD,
        self.APGDT,
        self.BIM,
        self.CW,
        self.DeepFool,
        self.DIFGSM,
       
        # output is not normalized after the attack
        self.EADEN,
        
        # output is not normalized after the attack
        self.EADL1,
       
        self.EOTPGD,
        self.FAB,
        self.FFGSM,
        self.FGSM,
        self.GN,
        self.Jitter,
       
        #  CUDA out of memory
        # self.JSMA,
       
        self.MIFGSM,
        self.NIFGSM,
        self.OnePixel,
        self.PGD,
        self.PGDL2,
        self.PGDRS,
        self.PGDRSL2,
        self.RFGSM,
        self.SINIFGSM,
        #  CUDA out of memory
        # self.SparseFool,
        self.SPSA,
        self.TIFGSM,
        self.TPGD,
        self.UPGD,
        self.VMIFGSM,
        self.VNIFGSM,
    ]

    self.black_box_attacks = [
        self.Pixle,
        self.Square,
    ]

    self.all_attack_names = self.white_box_attacks + self.black_box_attacks