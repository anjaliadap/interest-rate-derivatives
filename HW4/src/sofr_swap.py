from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from date_utilities import next_business_day, day_adjust, add_years, add_months, year_fraction
from typing import List
import numpy as np


class SOFRSwap:
    """Price SOFR swaps using discount factor curves."""

    def __init__(self, current_date: datetime.date, maturity_str: str, fixed_rate: float, yield_curve, notional: float = 100.0):
    
        """
        """
        self.current_date:  datetime.date   = current_date
        self.maturity_str:  str             = maturity_str  
        self.settle_date:   datetime.date   = day_adjust(self.current_date + timedelta(days=2))# T+2 settlement
        self.start_date:    datetime.date   = self.settle_date # for now, assume starts at settlement
        self.end_date:      datetime.date   = self.calculate_end_date(self.start_date, maturity_str)
        self.notional:      float           = notional
        self.fixed_rate:    float           = fixed_rate
        self.day_count:     str             = "ACT/360" # assuming ACT/360 for both legs of SOFR swaps
        self.yield_curve = yield_curve #yield curve has both 'OIS' and 'SOFR' when retrieving rates or discount factors

    
        

        self.payment_dates, self.period_dates = self.generate_payment_dates() # period dates are tuples of (start_date, end_date) for each period, used for calculating forward rates for floating leg cashflows
        self.year_fractions: list[float] = self.calculate_year_fractions()

        self.payment_frequency: int = 12 # Annual coupon payments is standard for SOFR swaps

    def calc_valuations(self):
        self.fixed_leg_pv = 0.0
        self.floating_leg_pv = 0.0
        self.BPV = self.calc_BPV()
        self.par_rate = self.calc_par_swap_rate()

    def calculate_end_date(self, start_date: datetime.date, maturity: str) -> datetime.date:
        """Calculate end date based on maturity string (e.g., '5Y' for 5 years)."""
        num_years = int(maturity[:-1])
        end_date = add_years(start_date, num_years)
        return end_date
    
    def generate_payment_dates(self) -> List[datetime]:
        """Generate coupon payment dates."""
        pay_dates = []
        period_dates = []
        this_date = self.start_date
        #settlement_date = self.start_date
        years_to_add = 1
        this_date = add_years(self.start_date, years_to_add)
        period_dates.append((self.start_date, this_date))
        while this_date <= self.end_date:
            pay_dates.append(this_date)
            years_to_add += 1
            this_date = add_years(self.start_date, years_to_add)
        for payment_date in pay_dates[1:]:
            period_dates.append((period_dates[-1][1], payment_date))
        return pay_dates, period_dates
    
    def calculate_year_fractions(self) -> List[float]:
        year_fracs = [0.0] * len(self.period_dates)
        year_fracs[0] = year_fraction(start=self.period_dates[0][0], end=self.period_dates[0][1])
        for period_num in range(0,len(self.period_dates)):
            year_fracs[period_num] = year_fraction(start=self.period_dates[period_num][0], end=self.period_dates[period_num][1])
        return year_fracs
    
    def get_discount_factor(self, this_date: datetime) -> float:
        """Get discount factor for a given date."""
        df = self.yield_curve.discount_factor(this_date, curve_type='OIS') 
        return df 
    
    def price_fixed_leg(self) -> float:
        """Calculate PV of fixed leg."""
        pv = 0.0
        for i in range(0, len(self.period_dates)):
            accrual_period = self.year_fractions[i]
            cashflow = self.notional * self.fixed_rate * accrual_period
            df = self.yield_curve.discount_factor(self.period_dates[i][1], curve_type='OIS') # use OIS discount factors for fixed leg
            pv += cashflow * df
        self.fixed_leg_pv = pv
        return pv
    
    def price_floating_leg(self, forward_rates: dict = None) -> float:
        """Calculate PV of floating leg. forward_rates: {date: sofr_rate}"""
        pv = 0.0
        for i in range(0, len(self.period_dates)):
            accrual_period = self.year_fractions[i]
            rate = self.yield_curve.get_forward_rate(self.period_dates[i][0], self.period_dates[i][1], curve_type='SOFR') #, self.day_count)
            cashflow = self.notional * rate * accrual_period
            df = self.yield_curve.discount_factor(self.period_dates[i][1], curve_type='OIS') # use OIS discount factors for fixed leg
            pv += cashflow * df
        self.floating_leg_pv = pv
        return pv
    
    def swap_value(self, forward_rates: dict) -> float:
        """Calculate swap value (receiver perspective: long fixed, short floating)."""
        fixed_pv = self.price_fixed_leg()
        floating_pv = self.price_floating_leg(forward_rates)
        return floating_pv - fixed_pv
    
    def calc_BPV(self):
        bpv = self.fixed_leg_pv / self.fixed_rate
        self.BPV = bpv
        return bpv

    def calc_par_swap_rate(self):
       par_rate = self.floating_leg_pv / self.BPV
       self.par_rate = par_rate
       return par_rate
   
    def fixed_leg_cashflows_to_df(self):
        #print("Fixed Leg Cashflows:")
        import pandas as pd
        fixed_leg_cashflows_df = pd.DataFrame(columns=['Period Start', 'Payment Date', 'Cashflow', 'Disc Cashflow', 'YearFraction', 'Coupon', 'OIS DF'], index=self.payment_dates[1:]) # create an empty DataFrame to store payment dates and cashflows
        for i in range(0, len(self.period_dates)):
            accrual_period = self.year_fractions[i]
            cashflow = self.notional * self.fixed_rate * accrual_period
            df = self.get_discount_factor(self.payment_dates[i])
            #print(f"Payment Date: {self.payment_dates[i]}, Cashflow: {cashflow:.2f}, Discount Factor: {df:.6f}")
            fixed_leg_cashflows_df.loc[self.payment_dates[i]] = [self.period_dates[i][0], self.period_dates[i][1], cashflow, cashflow*df, accrual_period, self.fixed_rate, df]
        fixed_leg_cashflows_df.sort_index(inplace=True)
        self.fixed_leg_cashflows_df = pd.DataFrame(fixed_leg_cashflows_df)
        return self.fixed_leg_cashflows_df
    
    def display_fixed_leg_cashflows(self):
        print("Fixed Leg Cashflows:")
        print(self.fixed_leg_cashflows_df)
    
    def floating_leg_cashflows_to_df(self, forward_rates: dict = None):
        #print("Floating Leg Cashflows:")
        import pandas as pd
        floating_leg_cashflows_df = pd.DataFrame(columns=['Period Start', 'Payment Date', 'Cashflow', 'Disc Cashflow', 'YearFraction', 'Forward Rate', 'OIS DF', 'SOFR DF Start', 'SOFR DF End'], index=self.payment_dates[1:]) # create an empty DataFrame to store payment dates and cashflows
        for i in range(0, len(self.period_dates)):
            accrual_period = self.year_fractions[i]
            rate = self.yield_curve.get_forward_rate(self.period_dates[i][0], self.period_dates[i][1], curve_type='SOFR') #self.day_count)
            cashflow = self.notional * rate * accrual_period
            df = self.get_discount_factor(self.payment_dates[i])
            df_sofr_start = self.yield_curve.discount_factor(self.period_dates[i][0], curve_type='SOFR')    
            df_sofr_end = self.yield_curve.discount_factor(self.period_dates[i][1], curve_type='SOFR')
            #print(f"Payment Date: {self.payment_dates[i]}, Cashflow: {cashflow:.2f}")
            floating_leg_cashflows_df.loc[self.payment_dates[i]] = [self.period_dates[i][0], self.period_dates[i][1], cashflow, cashflow*df, accrual_period, rate, df, df_sofr_start, df_sofr_end]
        floating_leg_cashflows_df.sort_index(inplace=True)
        self.floating_leg_cashflows_df = pd.DataFrame(floating_leg_cashflows_df)
        return self.floating_leg_cashflows_df
    
    def display_floating_leg_cashflows(self):
        print("Floating Leg Cashflows:")
        print(self.floating_leg_cashflows_df)





