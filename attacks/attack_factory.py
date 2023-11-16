import torch
from attacks.attack import Attack
from attacks.attack_names import AttackNames

class AttackFactory:

  @staticmethod
  def get_attack(attack_name: str, model: torch.nn.Module) -> Attack:
    attack_names = AttackNames().all_attack_names
    if attack_name not in attack_names:
      raise ValueError(f"Attack name {attack_name} not found in {attack_names}")

    if attack_name == AttackNames.APGD:
      from attacks.white_box.apgd import APGD
      return APGD(model)

    if attack_name == AttackNames.APGDT:
      from attacks.white_box.apgdt import APGDT
      return APGDT(model)

    if attack_name == AttackNames.BIM:
      from attacks.white_box.bim import BIM
      return BIM(model)

    if attack_name == AttackNames.CW:
      from attacks.white_box.cw import CW
      return CW(model)

    if attack_name == AttackNames.DeepFool:
      from attacks.white_box.deepfool import DeepFool
      return DeepFool(model)

    if attack_name == AttackNames.DIFGSM:
      from attacks.white_box.difgsm import DIFGSM
      return DIFGSM(model)

    if attack_name == AttackNames.EADEN:
      from attacks.white_box.eaden import EADEN
      return EADEN(model)

    if attack_name == AttackNames.EADL1:
      from attacks.white_box.eadl1 import EADL1
      return EADL1(model)

    if attack_name == AttackNames.EOTPGD:
      from attacks.white_box.eotpgd import EOTPGD
      return EOTPGD(model)

    if attack_name == AttackNames.FAB:
      from attacks.white_box.fab import FAB
      return FAB(model)

    if attack_name == AttackNames.FFGSM:
      from attacks.white_box.ffgsm import FFGSM
      return FFGSM(model)

    if attack_name == AttackNames.FGSM:
      from attacks.white_box.fgsm import FGSM
      return FGSM(model)

    if attack_name == AttackNames.GN:
      from attacks.white_box.gn import GN
      return GN(model)

    if attack_name == AttackNames.Jitter:
      from attacks.white_box.jitter import Jitter
      return Jitter(model)

    if attack_name == AttackNames.JSMA:
      from attacks.white_box.jsma import JSMA
      return JSMA(model)

    if attack_name == AttackNames.MIFGSM:
      from attacks.white_box.mifgsm import MIFGSM
      return MIFGSM(model)

    if attack_name == AttackNames.NIFGSM:
      from attacks.white_box.nifgsm import NIFGSM
      return NIFGSM(model)

    if attack_name == AttackNames.PGD:
      from attacks.white_box.pgd import PGD
      return PGD(model)

    if attack_name == AttackNames.PGDL2:
      from attacks.white_box.pgdl2 import PGDL2
      return PGDL2(model)

    if attack_name == AttackNames.PGDRS:
      from attacks.white_box.pgdrs import PGDRS
      return PGDRS(model)

    if attack_name == AttackNames.PGDRSL2:
      from attacks.white_box.pgdrsl2 import PGDRSL2
      return PGDRSL2(model)

    if attack_name == AttackNames.RFGSM:
      from attacks.white_box.rfgsm import RFGSM
      return RFGSM(model)

    if attack_name == AttackNames.SINIFGSM:
      from attacks.white_box.sinifgsm import SINIFGSM
      return SINIFGSM(model)

    if attack_name == AttackNames.SparseFool:
      from attacks.white_box.sparsefool import SparseFool
      return SparseFool(model)

    if attack_name == AttackNames.SPSA:
      from attacks.white_box.spsa import SPSA
      return SPSA(model)

    if attack_name == AttackNames.TIFGSM:
      from attacks.white_box.tifgsm import TIFGSM
      return TIFGSM(model)

    if attack_name == AttackNames.TPGD:
      from attacks.white_box.tpgd import TPGD
      return TPGD(model)

    if attack_name == AttackNames.UPGD:
      from attacks.white_box.upgd import UPGD
      return UPGD(model)

    if attack_name == AttackNames.VMIFGSM:
      from attacks.white_box.vmifgsm import VMIFGSM
      return VMIFGSM(model)

    if attack_name == AttackNames.VNIFGSM:
      from attacks.white_box.vnifgsm import VNIFGSM
      return VNIFGSM(model)

    if attack_name == AttackNames.Pixle:
      from attacks.black_box.pixle import Pixle
      return Pixle(model)

    if attack_name == AttackNames.Square:
      from attacks.black_box.square import Square
      return Square(model)

  @staticmethod
  def get_all_whitebox_attacks(model: torch.nn.Module):
    attacks = []
    for attack_name in AttackNames().white_box_attacks:
      attacks.append(AttackFactory.get_attack(attack_name, model))
    return attacks

  @staticmethod
  def get_all_blackbox_attacks(model: torch.nn.Module):
    attacks = []
    for attack_name in AttackNames().black_box_attacks:
      attacks.append(AttackFactory.get_attack(attack_name, model))
    return attacks

  @staticmethod
  def get_all_attacks(model: torch.nn.Module):
    attacks = []
    for attack_name in AttackNames().all_attack_names:
      attacks.append(AttackFactory.get_attack(attack_name, model))
    return attacks



