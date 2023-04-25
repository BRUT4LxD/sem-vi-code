from typing import List
import torch

from domain.attack_eval_score import AttackEvaluationScore
from domain.attack_result import AttackResult


class MultiattackResult():
  def __init__(self,
               eval_scores: List['AttackEvaluationScore'],
               attack_results: List[List['AttackResult']],
               adv_images: torch.Tensor):
    self.eval_scores = eval_scores
    self.attack_results = attack_results
    self.adv_images = adv_images