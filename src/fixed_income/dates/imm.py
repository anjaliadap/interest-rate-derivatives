import datetime as dt
from .constants import MONTH_MAPPING, IMM_MONTHS

"""
What are CBOT Month Codes?

CBOT month codes are single-letter codes used to represent the months of the year in futures 
contracts, which are traded on the Chicago Board of Trade (CBOT).

Each month is assigned a specific letter to simplify the identification of contract expiration months.
These codes are widely used in the financial industry for trading and referencing futures contracts.
"""

"""
Table of Contents:

1. get_third_Wednesday
2. get_next_IMM_Date
3. convert_CBOT_Month_Code_to_next_quarterly_IMM_Date
4. convert_CBOT_Month_Code_to_month_IMM_Date

"""

def get_third_Wednesday(year: int, month: int) -> dt.date:
    """
    This function returns the third Wednesday of a given month and year.
    
    Args:
        year (int): The year for which the third Wednesday is to be calculated.
        month (int): The month for which the third Wednesday is to be calculated.

    Returns:
        dt.date: The third Wednesday of the specified month and year.
    """

    # Get the first day of the month 
    first_date = dt.date(year, month, 1)
    
    # Get the first Wednesday of the month
    first_wednesday = first_date + dt.timedelta(days = (2 + 7 - first_date.weekday()) % 7)
    
    return first_wednesday + dt.timedelta(weeks=2)

# IMM Dates 

def get_next_IMM_Date(from_date: dt.date) -> dt.date: 
    """
    This function returns the next IMM date (the third Wednesday of March, June, September, or December)
    after the given date.
    
    Args:
        from_date (dt.date): The date from which to find the next IMM date.
    
    Returns:
        dt.date: The next IMM date after the given date.
    """

    # get list of IMM dates for the current year 
    imm_dates = [get_third_Wednesday(from_date.year, imm_month) for imm_month in IMM_MONTHS]

    # Check if given date is before any of the IMM dates in the current year
    for imm_date in imm_dates:
        if from_date <= imm_date:
            return imm_date
    
    # If not, return the first IMM date of the next year
    return get_third_Wednesday(from_date.year + 1, IMM_MONTHS[0])
    
def convert_CBOT_Month_Code_to_next_quarterly_IMM_Date(month_code: str, year: int) -> dt.date:
    """
    This function converts a CBOT month code and year to the next quarterly IMM date.
    
    Args:
        month_code (str): A single-letter CBOT month code.
        year (int): The year for which the IMM date is to be calculated.
    
    Returns:
        dt.date: The IMM date corresponding to the given month code and year.
    """
    if month_code not in MONTH_MAPPING:
        raise ValueError(f"Invalid CBOT month code: {month_code}")
    
    month = MONTH_MAPPING[month_code]
    
    return get_next_IMM_Date(dt.date(year, month, 1))

def convert_CBOT_Month_Code_to_month_IMM_Date(month_code: str, year: int) -> dt.date:
    """
    This function converts a CBOT month code and year to an IMM date (the third Wednesday of the month).
    
    Args:
        month_code (str): A single-letter CBOT month code.
        year (int): The year for which the IMM date is to be calculated.
    
    Returns:
        dt.date: The IMM date corresponding to the given month code and year.
    """
    if month_code not in MONTH_MAPPING:
        raise ValueError(f"Invalid CBOT month code: {month_code}")
    
    month = MONTH_MAPPING[month_code]
    
    return get_third_Wednesday(year, month)
