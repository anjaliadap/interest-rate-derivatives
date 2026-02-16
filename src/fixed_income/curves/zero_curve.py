from abc import ABC, abstractmethod
import numpy as np

class ZeroCurve(ABC):
    """
    Abstract base class for zero-coupon yield cuirves. 
    """

    @abstractmethod
    def get_zero_rate(self, t: float, comp: str = "continuous") -> float:
        """
        Returns y(0, t) under a compounding convention.
        Presently, this just returns constant zero rates for convenience. It will be replaced later. 
        """
        raise NotImplementedError

    @abstractmethod
    def get_discount_factor(self, t: float, comp: str = "continuous") -> float:
        """
        Returns P(0, t): PV of $1 paid at time t. 
        """
        raise NotImplementedError
    
    """
    Definitons for values of compounding convention:
    continuous: P(0,t) = exp(-y(0,t) * t)
    annual: P(0,t) = 1 / (1 + y(0,t))^t
    simple: P(0,t) = 1 / (1 + y(0,t) * t) # Not required for NSS spline
    """
    
    def get_forward_rate(self, t1: float, t2: float, comp: str = "continuous") -> float:
        """
        Returns f(t1, t2) under a compounding convention.
        """
        z1 = self.get_zero_rate(t1, comp)
        z2 = self.get_zero_rate(t2, comp)

        # Continuous compounding 
        if comp == "continuous":
            fwd_rate = (z2 * t2 - z1 * t1) / (t2 - t1)
        # Discrete compounding
        elif comp == "annual":
            df1 = self.get_discount_factor(t1, comp)
            df2 = self.get_discount_factor(t2, comp)
            fwd_rate = (df1 / df2 - 1) / (t2 - t1)
        else:
            raise ValueError("Unsupported compounding convention.")

        return fwd_rate

    def get_instantaneous_forward_rate(self, t: float, eps: float = 1e-5, comp: str = "continuous") -> float:
        """
        Numerical approximation to f(0,t) = -d ln P(0,t) / dt
        """
        if t < eps:
            raise ValueError("t must be greater than eps for numerical differentiation.")
        
        p0 = self.get_discount_factor(t, comp)
        p1 = self.get_discount_factor(t - eps, comp)
        return -(np.log(p0) - np.log(p1)) / eps

"""
Abstract classes say:
Any object that claims to be a ZeroCurve must be able to: 
    1. return a zero rate
    2. return a discount factor

@abstractmethod forces subclasses to implement these methods.
"""