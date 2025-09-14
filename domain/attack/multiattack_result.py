from typing import List

from domain.attack.attack_eval_score import AttackEvaluationScore
from domain.attack.attack_result import AttackResult

class MultiattackResult():
  def __init__(self,
               eval_scores: List['AttackEvaluationScore'],
               attack_results: List[List['AttackResult']]):
    self.eval_scores = eval_scores
    self.attack_results = attack_results