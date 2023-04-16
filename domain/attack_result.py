class AttackResult():
    def __init__(self, actual, predicted, adv_example, targeted=False):
        self.actual = actual
        self.predicted = predicted
        self.adv = adv_example
        self.targeted = targeted

    def __str__(self) -> str:
        return f"Actual: {self.actual}, Predicted: {self.predicted}"