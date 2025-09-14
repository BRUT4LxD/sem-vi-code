class AttackDistanceScore():
    def __init__(self, l0_pixels: float, l1: float, l2: float, linf: float, power_mse: float):
        self.l0_pixels = l0_pixels
        self.l1 = l1
        self.l2 = l2
        self.linf = linf
        self.power_mse = power_mse

    def __str__(self) -> str:
        l0 = f'l0_pix: {self.l0_pixels:.0f}'
        l1 = f'l1: {self.l1:.4f}'
        l2 = f'l2: {self.l2:.4f}'
        lInf = f'lInf: {self.linf:.4f}'
        p = f'pwr_mse: {self.power_mse:.2e}'
        return f"{l0:12s}{l1:14s}{l2:14s}{lInf:14s}{p:17s}"
