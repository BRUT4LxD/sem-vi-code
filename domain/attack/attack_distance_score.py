from dataclasses import dataclass


@dataclass
class AttackDistanceScore:
    """
    Dataclass to store attack distance metrics.
    
    Attributes:
        l0_pixels: Number of modified pixels
        l1: L1 distance (Manhattan distance)
        l2: L2 distance (Euclidean distance)
        linf: L∞ distance (maximum absolute difference)
        power_mse: Mean squared perturbation per element
    """
    l0_pixels: float
    l1: float
    l2: float
    linf: float
    power_mse: float

    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the distance score.
        """
        return {
            'l0_pixels': self.l0_pixels,
            'l1': self.l1,
            'l2': self.l2,
            'linf': self.linf,
            'power_mse': self.power_mse
        }

    def __str__(self) -> str:
        l0 = f'l0_pix: {self.l0_pixels:.0f}'
        l1 = f'l1: {self.l1:.4f}'
        l2 = f'l2: {self.l2:.4f}'
        lInf = f'lInf: {self.linf:.4f}'
        p = f'pwr_mse: {self.power_mse:.2e}'
        return f"{l0:12s}{l1:14s}{l2:14s}{lInf:14s}{p:17s}"
