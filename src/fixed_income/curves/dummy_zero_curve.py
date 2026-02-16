from .zero_curve import ZeroCurve
import numpy as np

class DummyZeroCurve(ZeroCurve):
    """
    A dummy implementation of ZeroCurve that returns constant zero rates and discount factors.
    """

    def __init__(self, constant_rate: float = 0.05):
        self.constant_rate = constant_rate

    def get_zero_rate(self, t: float, comp: str = "continuous") -> float:
        """
        Returns a constant zero rate regardless of t.
        """
        return self.constant_rate

    def get_discount_factor(self, t: float) -> float:
        """
        Returns the discount factor based on the constant zero rate.
        """
        if t < 0:
            raise ValueError("Time t must be non-negative.")
        
        # Continuous compounding
        return np.exp(-self.constant_rate * t)
