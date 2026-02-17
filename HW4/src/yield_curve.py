# yield_curve.py
import pandas as pd

class YieldCurve:
    """Wrapper for daily curve to interface with SOFRSwap class."""
    
    def __init__(self, daily_curve_df):
        self.curve = daily_curve_df.copy()
        self.curve.set_index('date', inplace=True)
    
    def discount_factor(self, date, curve_type='OIS'):
        date = pd.to_datetime(date)
        
        if date in self.curve.index:
            if curve_type == 'OIS':
                return self.curve.loc[date, 'ois_discount_factor']
            else:
                return self.curve.loc[date, 'discount_factor']
        else:
            idx = self.curve.index.get_indexer([date], method='nearest')[0]
            if curve_type == 'OIS':
                return self.curve.iloc[idx]['ois_discount_factor']
            else:
                return self.curve.iloc[idx]['discount_factor']
    
    def get_forward_rate(self, start_date, end_date, curve_type='SOFR'):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        df_start = self.discount_factor(start_date, curve_type)
        df_end = self.discount_factor(end_date, curve_type)
        
        days = (end_date - start_date).days
        forward_rate = (df_start / df_end - 1) * (360 / days)
        
        return forward_rate