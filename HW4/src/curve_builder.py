# curve_builder.py
import pandas as pd
import numpy as np
from datetime import datetime
import date_utilities as du

class SOFRCurveBuilder:
    """Builds SOFR and OIS discount curves from futures data."""
    
    def __init__(self, curve_date: str, sigma: float = 0.007):
        """curve_date in "YYYY-MM-DD" format"""
        self.curve_date = pd.to_datetime(curve_date)
        self.sigma = sigma
        self.sofr_futures = None
        self.daily_curve = None
    
    def load_futures_data(self, filepath: str):
        """Load SOFR futures from CSV"""
        self.sofr_futures = pd.read_csv(filepath)
        return self
    
    def prepare_futures(self):
        """Calculate implied rates, dates from tickers, t1, t2 (Q1-2)."""
        
        df = self.sofr_futures.copy()
        
        # Extract CBOT month codes from ticker 
        df['cbot_mth'] = df['ticker'].str[3:4]
        df['cbot_yr'] = df['ticker'].str[4:6].astype(int) + 2020
        
        df['start_date'] = df.apply(
            lambda row: du.Convert_CBOT_Month_Code_to_IMM_Date(row['cbot_mth'], row['cbot_yr']),
            axis=1
        )
        df['end_date'] = df.apply(
            lambda row: du.next_IMM_date(row['start_date']),
            axis=1
        )
        
        # Convert to datetime
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        # Calculate t1 and t2
        df['t1'] = (df['start_date'] - self.curve_date).dt.days / 365.25
        df['t2'] = (df['end_date'] - self.curve_date).dt.days / 365.25
        
        self.sofr_futures = df
        return self
    
    def apply_convexity_adjustment(self):
        """Apply convexity adjustment to get forward rates."""
        df = self.sofr_futures.copy()
        
        adjustment = 0.5 * (self.sigma ** 2) * df['t1'] * df['t2']
        df['forward_rate'] = df['futures_rate'] - adjustment
        
        self.sofr_futures = df
        return self
        
    def bootstrap_discount_factors(self):
        """Bootstrap discount factors at IMM dates."""
        df = self.sofr_futures.copy()
        
        df['days'] = (df['end_date'] - df['start_date']).dt.days
        df['single_period_df'] = 1 / (1 + (df['forward_rate'] / 100) * (df['days'] / 360))
        
        stub_days = (df['start_date'].iloc[0] - self.curve_date).days  # Dec 1 → Dec 17
        stub_rate = df['forward_rate'].iloc[0] / 100                   # 3.705%
        stub_df = 1 / (1 + stub_rate * stub_days / 360)                # 0.99836
        
        # cumprod starting from stub_df instead of 1.0
        df['discount_factor'] = df['single_period_df'].cumprod() * stub_df
        
        df['log_discount_factor'] = np.log(df['discount_factor'])
        
        self.sofr_futures = df
        return self
    
    def interpolate_daily_curve(self):
        """Create daily curve via log-linear interpolation"""
        # Add curve_date row
        curve_row = pd.DataFrame({
            'date': [self.curve_date],
            'discount_factor': [1.0],
            'log_discount_factor': [0.0]
        })
        
        # add stub period anchor point
        stub_end = self.sofr_futures['start_date'].iloc[0]  # Dec 17
        stub_days = (stub_end - self.curve_date).days        # 16 days
        stub_rate = self.sofr_futures['forward_rate'].iloc[0] / 100  # 3.705%
        
        stub_df = 1.0 / (1 + stub_rate * stub_days / 360)
        stub_row = pd.DataFrame({
            'date': [stub_end],
            'discount_factor': [stub_df],
            'log_discount_factor': [np.log(stub_df)]
        })

        imm_points = self.sofr_futures[['end_date', 'discount_factor', 'log_discount_factor']].copy()
        imm_points = imm_points.rename(columns={'end_date': 'date'})
        
        all_points = pd.concat([curve_row, stub_row, imm_points], ignore_index=True) # Combine
        
        end_date = self.curve_date + pd.DateOffset(years=4, months=3) # Create daily date range
        daily_dates = pd.date_range(start=self.curve_date, end=end_date, freq='D')
        
        daily_df = pd.DataFrame({'date': daily_dates})
        
        # Merge and interpolate
        daily_df = daily_df.merge(all_points[['date', 'log_discount_factor']], on='date', how='left')
        daily_df['log_discount_factor'] = daily_df['log_discount_factor'].interpolate(method='linear')
        
        # Exponentiate
        daily_df['discount_factor'] = np.exp(daily_df['log_discount_factor'])
        
        self.daily_curve = daily_df
        return self
    
    def calculate_daily_rates(self):
        """Calculate daily SOFR rates from discount factors"""
        df = self.daily_curve.copy()
        
        df['next_day_df'] = df['discount_factor'].shift(-1) # Shift to get next day's DF
        df['daily_sofr_rate'] = (df['discount_factor'] / df['next_day_df'] - 1) * 360 * 100 # Calculate daily rate
        df['daily_sofr_rate'] = df['daily_sofr_rate'].ffill() # Forward fill last day
        
        self.daily_curve = df
        return self
    
    def create_ois_curve(self, spread_bps: float = 6.0):
        """Create OIS rates and discount factors"""
        df = self.daily_curve.copy()
        
        # OIS rates
        df['ois_rate'] = df['daily_sofr_rate'] - (spread_bps / 100)
        
        df['single_day_ois_df'] = 1 / (1 + (df['ois_rate'] / 100) / 360)
        df['ois_discount_factor'] = df['single_day_ois_df'].cumprod()
        
        self.daily_curve = df
        return self
    
    def build(self, futures_filepath: str = None):
        """
        Main method - builds complete curve.
        
        Parameters:
        -----------
        futures_filepath : str, optional
            Path to futures CSV. If None, assumes data already loaded.
        
        Returns:
        --------
        pd.DataFrame : Daily curve with SOFR and OIS
        """
        if futures_filepath:
            self.load_futures_data(futures_filepath)
        
        self.prepare_futures()
        self.apply_convexity_adjustment()
        self.bootstrap_discount_factors()
        self.interpolate_daily_curve()
        self.calculate_daily_rates()
        self.create_ois_curve()
        
        return self.daily_curve