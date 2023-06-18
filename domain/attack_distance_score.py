class AttackDistanceScore():
    def __init__(self, l1: float, l2: float, lInf: float, power: float):
        self.l1 = round(l1 * 256, 2)
        self.l2 = round(l2 * 256, 2)
        self.lInf = round(lInf * 256, 2)
        self.power = round(power, 2)

    def __str__(self) -> str:
        l1 = f'l1: {self.l1}'
        l2 = f'l2: {self.l2}'
        lInf = f'lInf: {self.lInf}'
        p = f'pwr: {self.power}'
        return f"{l1:14s}{l2:14s}{lInf:14s}{p:17s}"
