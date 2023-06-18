from .apgd import APGD
from .apgdt import APGDT
from .bim import BIM
from .cw import CW
from .deepfool import DeepFool
from .difgsm import DIFGSM
from .eaden import EADEN
from .eadl1 import EADL1
from .eotpgd import EOTPGD
from .fab import FAB
from .ffgsm import FFGSM
from .fgsm import FGSM
from .gn import GN
from .jitter import Jitter
from .jsma import JSMA
from .mifgsm import MIFGSM
from .nifgsm import NIFGSM
from .onepixel import OnePixel
from .pgd import PGD
from .pgdl2 import PGDL2
from .pgdrs import PGDRS
from .pgdrsl2 import PGDRSL2
from .rfgsm import RFGSM
from .sinifgsm import SINIFGSM
from .sparsefool import SparseFool
from .spsa import SPSA
from .tifgsm import TIFGSM
from .tpgd import TPGD
from .upgd import UPGD
from .vmifgsm import VMIFGSM
from .vnifgsm import VNIFGSM

# heavy attacks were commented out
def get_all_white_box_attack(model):
    return [
        APGD(model),
        APGDT(model),
        BIM(model),
        CW(model),
        DeepFool(model),
        DIFGSM(model),
        EADEN(model),
        EADL1(model),
        EOTPGD(model),
        FAB(model),
        FFGSM(model),
        FGSM(model),
        GN(model),
        Jitter(model),
        # JSMA(model),
        MIFGSM(model),
        NIFGSM(model),
        OnePixel(model),
        PGD(model),
        PGDL2(model),
        PGDRS(model),
        PGDRSL2(model),
        RFGSM(model),
        SINIFGSM(model),
        # SparseFool(model),
        SPSA(model),
        TIFGSM(model),
        TPGD(model),
        UPGD(model),
        VMIFGSM(model),
    ]
