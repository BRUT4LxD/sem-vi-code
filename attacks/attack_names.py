
class AttackNames():

  # White-box attacks

  APGD = "APGD"
  APGDT = "APGDT"
  BIM = "BIM"
  CW = "CW"
  DeepFool = "DeepFool"
  DIFGSM = "DI-FGSM"
  EADEN = "EADEN"
  EADL1 = "EADL1"
  EOTPGD = "EOT-PGD"
  FAB = "FAB"
  FFGSM = "FFGSM"
  FGSM = "FGSM"
  GN = "GN"
  Jitter = "Jitter"
  JSMA = "JSMA"
  MIFGSM = "MI-FGSM"
  NIFGSM = "NI-FGSM"
  PGD = "PGD"
  PGDL2 = "PGDL2"
  PGDRS = "PGDRS"
  PGDRSL2 = "PGDRSL2"
  RFGSM = "RFGSM"
  SINIFGSM = "SIN-IFGSM"
  SparseFool = "SparseFool"
  SPSA = "SPSA"
  TIFGSM = "TI-FGSM"
  TPGD = "TPGD"
  UPGD = "UPGD"
  VMIFGSM = "VMI-FGSM"
  VNIFGSM = "VNI-FGSM"

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
        self.EADEN,
        self.EADL1,
        self.EOTPGD,
        self.FAB,
        self.FFGSM,
        self.FGSM,
        self.GN,
        self.Jitter,
        self.JSMA,
        self.MIFGSM,
        self.NIFGSM,
        self.PGD,
        self.PGDL2,
        self.PGDRS,
        self.PGDRSL2,
        self.RFGSM,
        self.SINIFGSM,
        self.SparseFool,
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

    self.all_attacks = self.white_box_attacks + self.black_box_attacks