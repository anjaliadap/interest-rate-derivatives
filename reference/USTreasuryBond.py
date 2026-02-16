# USTreasuryBond.py
# A class to model U.S. Treasury Bonds with methods for pricing, yield calculation, accrued interest, DV01, and convexity.
# DISCLAIMER: THERE WILL BE BUGS IN THIS CODE.
# BUGS OF ALL KINDS ARE LIKELY PRESENT, INCLUDING LOGICAL, MATHEMATICAL, PROGRAMMING ERRORS AND STYLE ERRORS.
# BUGS MAY CAUSE INCORRECT RESULTS.
# THERE WILL BE NO SUPPORT FOR THIS CODE. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
# USE AT YOUR OWN RISK.
# FEEL FREE TO MODIFY AND IMPROVE THE CODE.
# BY USING THIS CODE, YOU AGREE TO THE TERMS OF THIS DISCLAIMER.
# THIS CODE IS PROVIDED FOR EDUCATIONAL PURPOSES ONLY.
# DO NOT USE THIS CODE FOR REAL TRADING OR INVESTMENT DECISIONS.
# THE AUTHOR IS NOT RESPONSIBLE FOR ANY LOSSES OR DAMAGES ARISING FROM THE USE OF THIS CODE.
# DO NOT DISTRIBUTE THIS CODE WITHOUT PERMISSION FROM THE AUTHOR.
# FOR PERSONAL USE ONLY
import datetime
from DateUtilities import add_months
from pandas.tseries.holiday import USFederalHolidayCalendar
from math import exp
class USTreasuryBond:
    def __init__(self, ticker, issue_date, dated_date, maturity_date, coupon_rate, face_value=100):
        self.ticker = ticker
        self.issue_date = issue_date
        self.dated_date = dated_date
        self.maturity_date = maturity_date
        self.coupon_rate = coupon_rate # Annual coupon rate as a decimal
        self.face_value = face_value
        self.coupon_periods = self._generate_coupon_period_dates(dated_date, maturity_date) # List of tuples (start_date, end_date) for each coupon period
        self.coupon_period_dates = [dated_date] + [end_date for (start_date, end_date) in self.coupon_periods] # List of coupon period dates from dated date to maturity date
    def _generate_coupon_period_dates(self, start_date, end_date):
        """
        Generate coupon payment dates between start_date and end_date, which are both coupon period dates
        Parameters:
        start_date (datetime.date): The date from which to start generating coupon dates.
        end_date (datetime.date): The maturity date of the bond.
        Returns:
        list: A list of coupon payment dates.
        """
        coupon_periods = []
        current_period_start_date  = start_date
        class no_calendar():
            def __init__(self):
                self.holidays = []
        cal_no = no_calendar()
        current_period_end_date = add_months(current_period_start_date, num_months=6, cal=cal_no, day_convention = "None", eom=True)
        current_period = (current_period_start_date, current_period_end_date)
        coupon_periods.append(current_period)
        while current_period_end_date < end_date:
            current_period_start_date = current_period_end_date
            current_period_end_date = add_months(current_period_start_date, num_months=6, cal=cal_no, day_convention = "None", eom=True)
            current_period = (current_period_start_date, current_period_end_date)
            coupon_periods.append(current_period)
        return coupon_periods

    def yield_to_pv(self, yield_rate, settlement_date): # payments scale by face value, yields and coupon rates are decimal.
        """
        Calculate the price of the USTreasuryBond given a yield rate and settlement date.
        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.
        Returns:
        float: The price of the bond.
        """
        # Implementation of bond pricing logic goes here
        pv = 0.0
        period_number = 0
        year_fraction = None
        for period_start, period_end in self.coupon_periods:
            if period_end > settlement_date:
                if year_fraction == None:
                    year_fraction = (period_end - settlement_date) / (period_end - period_start)
                coupon_payment = (self.coupon_rate / 2) * self.face_value
                pv += coupon_payment / ((1 + yield_rate / 2) ** (period_number + year_fraction))
                period_number += 1
        pv += self.face_value / ((1 + yield_rate / 2) ** (period_number -1 + year_fraction))
        return pv

    def accrued_interest(self, settlement_date):
        """
        Calculate the accrued interest of the USTreasuryBond as of the settlement date.
        Parameters:
        settlement_date (datetime.date): The date on which the bond is being priced.
        Returns:
        float: The accrued interest.
        """
        last_coupon_date = None
        next_coupon_date = None
        for period_start, period_end in self.coupon_periods:
            if period_end > settlement_date:
                next_coupon_date = period_end
                last_coupon_date = period_start
                break
        if last_coupon_date is None or next_coupon_date is None:
            return 0.0

        accrued_days = (settlement_date - last_coupon_date).days
        total_days = (next_coupon_date - last_coupon_date).days
        accrued_interest = (self.coupon_rate / 2) * self.face_value * (accrued_days / total_days)
        return accrued_interest

    def yield_to_price(self, yield_rate, settlement_date):
        """
        Calculate the quoted price of the USTreasuryBond, which excludes accrued interest.
        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.
        Returns:
        float: The quoted price of the bond, which excludes accrued interest.
        """
        pv = self.yield_to_pv(yield_rate, settlement_date)
        accrued_interest = self.accrued_interest(settlement_date)
        clean_price = pv - accrued_interest
        return clean_price

    def price_to_yield(self, market_price, settlement_date, tol=1e-6, max_iter=100):
        """
        Calculate the yield to maturity of the USTreasuryBond given a market price and settlement date.
        Parameters:
        market_price (float): The market price of the bond.
        settlement_date (datetime.date): The date on which the bond is being priced.
        tol (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations.
        Returns:
        float: The yield to maturity (as a decimal).
        """
        lower_bound = 0.0
        upper_bound = 1.0
        for _ in range(max_iter): # bisection method. Can be improved with newton-raphson or secant method or solvers from scipy
            mid_yield = (lower_bound + upper_bound) / 2
            price = self.yield_to_price(mid_yield, settlement_date)
            if abs(price - market_price) < tol:
                return mid_yield
            if price < market_price:
                upper_bound = mid_yield
            else:
                lower_bound = mid_yield
        raise ValueError("Yield to maturity calculation did not converge")
    def price_to_coupon(self, price, base_yield, settlement_date, tol=1e-6, max_iter=100): #
        """
        Calculate the coupon rate of the USTreasuryBond given a market price and settlement date.
        Parameters:
        market_price (float): The market price of the bond.
        settlement_date (datetime.date): The date on which the bond is being priced.
        tol (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations.
        Returns:
        float: The coupon rate (as a decimal).
        """
        test_bond = USTreasuryBond(self.ticker, self.issue_date, self.dated_date, self.maturity_date, self.coupon_rate, self.face_value)
        lower_bound = 0.0
        upper_bound = 0.5
        for _ in range(max_iter):
            mid_coupon = (lower_bound + upper_bound) / 2
            test_bond.coupon_rate = mid_coupon
            mid_price = test_bond.yield_to_price_quoted(base_yield, settlement_date)
            if abs(price - mid_price) < tol:
                return mid_coupon
            if price < mid_price: # if price is less than mid_price, need to lower coupon
                upper_bound = mid_coupon
            else:
                lower_bound = mid_coupon
        raise ValueError("Coupon rate calculation did not converge")

    def pv_from_zcb_yield_curve(self, zcb_yield_curve, settlement_date):
        """
        Calculate the price of the USTreasuryBond using a zero-coupon bond yield curve.
        Parameters:
        zcb_yield_curve (function): A function that takes time to maturity (in years) and returns the zero-coupon yield.
        settlement_date (datetime.date): The date on which the bond is being priced.
        Returns:
        float: The pv of the bond.
        IMPORTANTpv is defined as the sum of the present values of all future cash flows, discounted using the zero-coupon yield curve.
        """
        pass

    def CPN01(self, yield_rate, settlement_date, delta_yield=0.0001): #DV01 is positive, negative of the derivative of price wrt yield
        """
        Calculate the DV01 (Dollar Value of 01) of the USTreasuryBond.
        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.
        delta_yield (float): The change in yield to calculate DV01.
        Returns:
        float: The DV01 of the bond.
        """
        bond_cpn_up = USTreasuryBond("", self.issue_date, self.dated_date, self.maturity_date, self.coupon_rate + 0.0001, self.face_value)
        price_up = bond_cpn_up.price_quoted(yield_rate, settlement_date)
        bond_cpn_down = USTreasuryBond("", self.issue_date, self.dated_date, self.maturity_date, self.coupon_rate - 0.0001, self.face_value)
        price_down = bond_cpn_down.price_quoted(yield_rate, settlement_date)
        cpn01 = (price_up - price_down) / 2
        return cpn01
    def DV01(self, yield_rate, settlement_date, delta_yield=0.0001): #DV01 is positive, negative of the derivative of price wrt yield
        """
        Calculate the DV01 (Dollar Value of 01) of the USTreasuryBond.
        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.
        delta_yield (float): The change in yield to calculate DV01.
        Returns:
        float: The DV01 of the bond.
        """
        price_up = self.yield_to_price_quoted(yield_rate + delta_yield, settlement_date)
        price_down = self.yield_to_price_quoted(yield_rate - delta_yield, settlement_date)
        dv01 = (price_down - price_up) / 2
        return dv01

    def convexity(self, yield_rate, settlement_date, delta_yield=0.0001):
        """
        Calculate the convexity of the USTreasuryBond.
        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.
        delta_yield (float): The change in yield to calculate convexity.
        Returns:
        float: The convexity of the bond.
        """
        price_up = self.yield_to_price_quoted(yield_rate + delta_yield, settlement_date)
        price_down = self.yield_to_price_quoted(yield_rate - delta_yield, settlement_date)
        price_current = self.yield_to_price_quoted(yield_rate, settlement_date)
        convexity = (price_up + price_down - 2 * price_current)
        return convexity
def main():
    issue_date1 = datetime.date(2025,12,1)
    dated_date1 = datetime.date(2025,11,15)
    maturity_date1 = datetime.date(2045,11,15)
    coupon_rate1 = 0.03  # 3%
    bond1 = USTreasuryBond('912810RP',issue_date1, dated_date1, maturity_date1, coupon_rate1)
    issue_date2 = datetime.date(2015,11,15)
    dated_date2 = datetime.date(2015,11,15)
    maturity_date2 = datetime.date(2045,11,15)
    coupon_rate2 = 0.04625  # 3%
    bond2 = USTreasuryBond('912810RP',issue_date2, dated_date2, maturity_date2, coupon_rate2)
    settlement_date = datetime.date(2025, 12, 1)
    price1 = 77.4609375
    price2 = 98.9921875
    accrued_interest1 = bond1.accrued_interest(settlement_date)
    accrued_interest2 = bond2.accrued_interest(settlement_date)
    ytm1 = bond1.yield_to_maturity(price1, settlement_date)
    ytm2 = bond2.yield_to_maturity(price2, settlement_date)
    print(accrued_interest1)
    print(accrued_interest2)
    print(ytm1)
    print(ytm2)
    price1_from_yield = bond1.price_quoted(ytm1, settlement_date)
    price2_from_yield = bond2.price_quoted(ytm2, settlement_date)
    x=0
    dv01 = bond1.DV01(ytm1, settlement_date)
    convexity = bond1.convexity(ytm1, settlement_date)
    #print(f"Quoted Price: {price:.2f}")
    #print(f"Accrued Interest: {accrued_interest:.2f}")
    #print(f"Yield to Maturity: {ytm:.6f}")
    #print(f"DV01: {dv01:.6f}")
    #print(f"Convexity: {convexity:.6f}")
    if __name__ == "__main__":
        main()
