# fixed_income/calibration/calibrate_nelson_siegel_svensson.py

import numpy as np
import pandas as pd

from fixed_income.instruments.bond import FixedRateBond
from fixed_income.curves.nelson_siegel_svensson_spline import NelsonSiegelSvenssonSpline

class CalibrateNelsonSiegelSvensson:
    """
    Load bond info + price history, then calibrate NS / NSS curves day-by-day.

    Holds:
    - bond_info_path: path to bond static info CSV
    - price_info_path: path to bond price history CSV
    - spline configuration (model, fix_lambdas, tau1, tau2)

    Main entry:
        fit_spline(settlement_date, model=None, fix_lambdas=None, tau1=None, tau2=None)
        If None, uses configured defaults.

    Objective: sum_i (PV_model - PV_mkt)^2 / DV01_i
    """

    def __init__(self, bond_info_path: str, price_info_path: str):
        self.bond_info_path = bond_info_path
        self.price_info_path = price_info_path

        self.bond_list: list[FixedRateBond] = []
        self.price_history: pd.DataFrame = pd.DataFrame()

        # configuration defaults (can be changed via configure_spline)
        self.model: str = "NSS"
        self.fix_lambdas: bool = True
        self.tau1: float = 4.0
        self.tau2: float = 10.75

        self._load_data()

    def _load_data(self):
        # bond static data
        info = pd.read_csv(self.bond_info_path)
        info.columns = [c.strip().lower() for c in info.columns]
        for c in ["issue_date", "dated_date", "maturity_date"]:
            info[c] = pd.to_datetime(info[c], errors="raise").dt.date

        for _, r in info.iterrows():
            cusip = str(r["cusip"]).strip()
            coupon = float(r["coupon_rate"]) / 100.0
            self.bond_list.append(
                FixedRateBond(
                    ticker=cusip,
                    coupon_rate=coupon,
                    issue_date=r["issue_date"],
                    dated_date=r["dated_date"],
                    maturity_date=r["maturity_date"],
                    face_value=100.0,
                    coupon_frequency=2,
                )
            )

        # price history (long: cusip, date, price)
        px = pd.read_csv(self.price_info_path)
        px.columns = [c.strip().lower() for c in px.columns]
        px["cusip"] = px["cusip"].astype(str).str.strip()
        px["date"] = pd.to_datetime(px["date"], errors="raise").dt.date
        px["price"] = pd.to_numeric(px["price"], errors="coerce")
        px.loc[px["price"] == 0, "price"] = np.nan
        self.price_history = px.dropna(subset=["price"])[["cusip", "date", "price"]].copy()

    def configure_spline(self, model: str = "NSS", fix_lambdas: bool = True, tau1: float = 4.0, tau2: float = 10.75):
        """
        Sets the default configuration for spline fitting.
        These can be overridden in fit_spline().
        """
        self.model = model.upper()
        if self.model not in ("NS", "NSS"):
            raise ValueError("model must be 'NS' or 'NSS'")

        self.fix_lambdas = bool(fix_lambdas)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)

        if self.model == "NSS" and (self.tau1 <= 0 or self.tau2 <= 0):
            raise ValueError("tau1 and tau2 must be > 0")
        if self.model == "NS" and self.tau1 <= 0:
            raise ValueError("tau1 must be > 0")

    def _bonds_and_prices_for_date(self, settlement_date):
        px_today = self.price_history[self.price_history["date"] == settlement_date]
        if px_today.empty:
            raise ValueError(f"No prices found for settlement_date={settlement_date}")

        price_map = dict(zip(px_today["cusip"], px_today["price"]))

        bonds_today = [b for b in self.bond_list if b.ticker in price_map]
        prices_today = [float(price_map[b.ticker]) for b in bonds_today]

        return bonds_today, prices_today

    def _market_pv_and_dv01(self, bonds_today, clean_prices, settlement_date):
        """
        Build pv_market_list and dv01_list.

        IMPORTANT: pv_market is set to dirty price = clean + accrued_interest
        assuming b.pv_from_zero_curve returns a dirty PV.
        """
        pv_market_list = []
        dv01_list = []

        for b, clean in zip(bonds_today, clean_prices):
            pv_market_list.append(clean + b.accrued_interest(settlement_date))

            y = b.price_to_yield_bisection(clean, settlement_date)
            dv01 = abs(b.DV01(y, settlement_date))
            dv01_list.append(max(float(dv01), 1e-10))

        return pv_market_list, dv01_list

    def fit_spline(self, settlement_date, model=None, fix_lambdas=None, tau1=None, tau2=None):

        model = self.model if model is None else model
        fix_lambdas = self.fix_lambdas if fix_lambdas is None else fix_lambdas
        tau1 = self.tau1 if tau1 is None else tau1
        tau2 = self.tau2 if tau2 is None else tau2
        fix_lambdas = bool(fix_lambdas)

        model = str(model).upper()
        if model not in ("NS", "NSS"):
            raise ValueError("model must be 'NS' or 'NSS'")

        tau1 = float(tau1)
        tau2 = float(tau2)

        if model == "NSS" and (tau1 <= 0 or tau2 <= 0):
            raise ValueError("tau1 and tau2 must be > 0")
        if model == "NS" and tau1 <= 0:
            raise ValueError("tau1 must be > 0")

        bonds_today, clean_prices = self._bonds_and_prices_for_date(settlement_date)

        # minimum bonds required
        min_needed = 4 if model == "NS" else (4 if fix_lambdas else 6)
        if len(bonds_today) < min_needed:
            raise ValueError(f"Not enough bonds to fit {model} on {settlement_date}: {len(bonds_today)}")

        pv_market_list, dv01_list = self._market_pv_and_dv01(bonds_today, clean_prices, settlement_date)

        # instantiate spline with dummy params; fit() will overwrite params
        spline = NelsonSiegelSvenssonSpline(params=[0, 0, 0, 0, tau1, tau2])

        # NS Model
        if model == "NS":
            params, res = spline.fit(
                bond_list=bonds_today,
                pv_market_list=pv_market_list,
                dv01_list=dv01_list,
                settlement_date=settlement_date,
                model="NS",
            )
            return params, res

        # NSS Model
        if fix_lambdas:
            params, res = spline.fit(
                bond_list=bonds_today,
                pv_market_list=pv_market_list,
                dv01_list=dv01_list,
                settlement_date=settlement_date,
                model="NSS",
                tau1=tau1,
                tau2=tau2,
            )
        else:
            params, res = spline.fit(
                bond_list=bonds_today,
                pv_market_list=pv_market_list,
                dv01_list=dv01_list,
                settlement_date=settlement_date,
                model="NSS",
            )

        return params, res
    
    def run_history(self, model=None, fix_lambdas=None):
        """
        Fit on all dates in price_history.
        Returns DataFrame with params and objective value.
        """
        dates = sorted(self.price_history["date"].unique())

        rows = []
        for d in dates:
            try:
                params, res = self.fit_spline(d, model=model, fix_lambdas=fix_lambdas)
                rows.append(
                    {
                        "date": d,
                        "params": params,
                        "fun": float(getattr(res, "fun", np.nan)),
                        "success": bool(getattr(res, "success", True)),
                        "nit": int(getattr(res, "nit", -1)),
                        "nfev": int(getattr(res, "nfev", -1)),
                        "message": str(getattr(res, "message", "")),
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "date": d,
                        "params": None,
                        "fun": np.nan,
                        "success": False,
                        "nit": np.nan,
                        "nfev": np.nan,
                        "message": str(e),
                    }
                )

        return pd.DataFrame(rows)
