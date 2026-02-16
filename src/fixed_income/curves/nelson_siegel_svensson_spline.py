import numpy as np
from scipy.optimize import minimize
from fixed_income.curves.zero_curve import ZeroCurve

class NelsonSiegelSvenssonSpline(ZeroCurve):
    """
    Nelson-Siegel-Svensson spline model for yield curve fitting.

    Params for NSS: [beta0, beta1, beta2, beta3, tau1, tau2]
    Params for NS : [beta0, beta1, beta2, tau1]   (internally set beta3=0, tau2=1)
    """

    def __init__(self, params):
        self.params = np.array(params, dtype=float)

        if len(self.params) != 6:
            raise ValueError("params must be length 6: [beta0, beta1, beta2, beta3, tau1, tau2]")

    @staticmethod
    def nelson_siegel_svensson_yield(maturity, params):
        """
        Returns the zcb yield (continuous) given maturity (float or array).
        params must be full 6-vector: [b0,b1,b2,b3,t1,t2]
        """
        b0, b1, b2, b3, t1, t2 = params
        m = np.asarray(maturity, dtype=float)

        eps = 1e-12
        x1 = np.maximum(m / t1, eps)
        x2 = np.maximum(m / t2, eps)

        term1 = b0
        term2 = b1 * ((1 - np.exp(-x1)) / x1)
        term3 = b2 * (((1 - np.exp(-x1)) / x1) - np.exp(-x1))
        term4 = b3 * (((1 - np.exp(-x2)) / x2) - np.exp(-x2))
        return term1 + term2 + term3 + term4
    
    # ---------- Zero Curve interface methods ----------
    def get_zero_rate(self, t: float) -> float:
        return float(self.nelson_siegel_svensson_yield(t, self.params))
        
    def get_discount_factor(self, t: float) -> float:
        r = self.get_zero_rate(t)
        return float(np.exp(-r * t))

    
    @staticmethod
    def objective(params, bond_list, pv_market_list, dv01_list, settlement_date):
        """
        total_error += (1/dv01_i) * (pv_model_i - pv_mkt_i)^2
        """
        curve = NelsonSiegelSvenssonSpline(params=params)

        total = 0.0
        for b, pv_mkt, dv01 in zip(bond_list, pv_market_list, dv01_list):
            w = max(abs(float(dv01)), 1e-10)
            pv_model = b.pv_from_zero_curve(curve, settlement_date)
            total += (1.0 / w) * (pv_model - pv_mkt) ** 2
        return float(total)
    
    def fit(self, bond_list, pv_market_list, dv01_list, tau1=None, tau2=None, settlement_date=None, model="NSS",):
        if settlement_date is None:
            raise ValueError("settlement_date is required")

        if (tau1 is None) ^ (tau2 is None):
            raise ValueError("Provide both tau1 and tau2, or neither.")

        model = str(model).upper()
        if model not in ("NS", "NSS"):
            raise ValueError("model must be 'NS' or 'NSS'")

        # ----------------
        # bounds
        # ----------------
        beta_bounds = [(-0.10, 0.20), (-0.50, 0.50), (-0.50, 0.50)]
        b3_bounds = [(-0.50, 0.50)]

        # prevent overlap tau1 < 10 < tau2
        EPS = 1e-6
        tau1_bounds = (2.0, 10.0 - EPS)
        tau2_bounds = (10.0 + EPS, 18.0)

        options = {"maxiter": 500}

        # ----------------
        # NS: fit (b0,b1,b2,tau1)
        # ----------------
        if model == "NS":
            x0 = np.array([0.04, -0.02, 0.02, 4.0], dtype=float)
            bounds = beta_bounds + [tau1_bounds]

            def obj_ns(x):
                b0, b1, b2, t1 = x
                full = [b0, b1, b2, 0.0, t1, 1.0]  # beta3=0, tau2 unused
                return NelsonSiegelSvenssonSpline.objective(
                    full, bond_list, pv_market_list, dv01_list, settlement_date
                )

            res = minimize(obj_ns, x0=x0, bounds=bounds, method="L-BFGS-B", options=options)

            b0, b1, b2, t1 = res.x
            self.params = np.array([b0, b1, b2, 0.0, t1, 1.0], dtype=float)
            return self.params, res

        # ----------------
        # NSS with fixed taus: fit (b0,b1,b2,b3)
        # ----------------
        if tau1 is not None and tau2 is not None:
            x0 = np.array([0.04, -0.02, 0.02, 0.00], dtype=float)
            bounds = beta_bounds + b3_bounds

            def obj_fix(x):
                b0, b1, b2, b3 = x
                full = [b0, b1, b2, b3, float(tau1), float(tau2)]
                return NelsonSiegelSvenssonSpline.objective(
                    full, bond_list, pv_market_list, dv01_list, settlement_date
                )

            res = minimize(obj_fix, x0=x0, bounds=bounds, method="L-BFGS-B", options=options)

            b0, b1, b2, b3 = res.x
            self.params = np.array([b0, b1, b2, b3, float(tau1), float(tau2)], dtype=float)
            return self.params, res

        # ----------------
        # NSS free taus (Q4): fit (b0,b1,b2,b3,tau1,tau2)
        # ----------------
        x0 = np.array([0.04, -0.02, 0.02, 0.00, 4.0, 10.75], dtype=float)
        bounds = beta_bounds + b3_bounds + [tau1_bounds, tau2_bounds]

        def obj_nss(x):
            # x = [b0,b1,b2,b3,tau1,tau2]
            return NelsonSiegelSvenssonSpline.objective(
                x, bond_list, pv_market_list, dv01_list, settlement_date
            )

        res = minimize(obj_nss, x0=x0, bounds=bounds, method="L-BFGS-B", options=options)

        self.params = np.array(res.x, dtype=float)
        return self.params, res
