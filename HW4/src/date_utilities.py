# DateUtilities.py
# A collection of date utility functions for financial applications.
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
from datetime import timedelta, date
from pandas.tseries.holiday import USFederalHolidayCalendar
import holidays
import calendar
from calendar import monthrange
import numpy as np
import pandas as pd

NUMBER_SECONDS_IN_A_DAY = 86400.0

def convert_date_to_num_days(this_date): # convert a date to the number of days since 1970(?)
    if type(this_date) == list:
        return_days_list = []
        for one_date in this_date:
            return_days_list.append(datetime.datetime(one_date.year, one_date.month, one_date.day).timestamp()/NUMBER_SECONDS_IN_A_DAY)
        return return_days_list
    else:
        return datetime.datetime(this_date.year, this_date.month, this_date.day).timestamp()/NUMBER_SECONDS_IN_A_DAY
    
def convert_num_days_to_date(this_num_days):
    if type(this_num_days) == list:
        return_date_list = []
        for one_day in this_num_days:
            one_datetime = datetime.fromtimestamp(one_day * NUMBER_SECONDS_IN_A_DAY)
            one_date = one_datetime.date()
            return_date_list.append(one_date)
        return return_date_list
    else:
        return (datetime.fromtimestamp(this_num_days * NUMBER_SECONDS_IN_A_DAY)).date()



def find_index(list_of_values, value) -> int:
    """Find the index of the largest element in list_of_values less than or equal to value.


    Args:
        list_of_values (list): A sorted list of values.
        value (float): The value to compare against.

    Returns:
        int: The index of the largest element less than or equal to value.
    """
    target_index = np.max(np.flatnonzero( np.array(list_of_values) <= value ))
    #target_value = list_of_values[int(target_value)]
    return target_index

def find_index_for_dates(list_of_dates, target_date) -> int:
    """Find the index of the largest element in list_of_values less than or equal to value.


    Args:
        list_of_dates (list): A sorted list of dates.
        value (float): The value to compare against.

    Returns:
        int: The index of the largest date less than or equal to value.
    """
    list_of_datetimes = [datetime.datetime(this_date.year, this_date.month, this_date.day) for this_date in list_of_dates]
    list_of_values = [this_datetime.timestamp()/86400.0 for this_datetime in list_of_datetimes] # number of days since 1970(?)
    value = datetime.datetime(target_date.year, target_date.month, target_date.day).timestamp()/86400.0
    target_index = np.max(np.flatnonzero( np.array(list_of_values) < value ))
    #target_value = list_of_values[int(target_value)]
    return target_index

    for i in range(len(list_of_values)):
        if list_of_values[i] > value:
            return i - 1 if i > 0 else 0
    return len(list_of_values) - 1
month_mapping = {
        'F': 1,  # January
        'G': 2,  # February
        'H': 3,  # March
        'J': 4,  # April
        'K': 5,  # May
        'M': 6,  # June
        'N': 7,  # July
        'Q': 8,  # August
        'U': 9,  # September
        'V': 10, # October
        'X': 11, # November
        'Z': 12  # December
    }

def Convert_CBOT_Month_Code_to_IMM_Date(month_code: str, year: int) -> datetime.date:
    """Convert CBOT month code and year to IMM date (3rd Wednesday of the month).

    Args:
        month_code (str): CBOT month code (e.g., 'H' for March).
        year (int): Year as a four-digit integer.

    Returns:
        datetime.date: The IMM date corresponding to the given month code and year.
    """
    if month_code not in month_mapping:
        raise ValueError(f"Invalid month code: {month_code}")

    month = month_mapping[month_code]
    # Find the third Wednesday of the month
    first_day = datetime.date(year, month, 1)
    first_wednesday = first_day + datetime.timedelta(days=(2 - first_day.weekday() + 7) % 7)
    third_wednesday = first_wednesday + datetime.timedelta(weeks=2)

    return third_wednesday

def next_IMM_date(from_date: datetime.date) -> datetime.date:
    """Get the next IMM date after the given date.

    Args:
        from_date (datetime.date): The date from which to find the next IMM date.

    Returns:
        datetime.date: The next IMM date after the given date.
    """
    year = from_date.year
    imm_months = [3, 6, 9, 12]  # March, June, September, December

    for month in imm_months:
        imm_date = Convert_CBOT_Month_Code_to_IMM_Date(
            {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}[month], year)
        if imm_date > from_date:
            return imm_date

    # If no IMM date found in the current year, return the first IMM date of the next year
    return Convert_CBOT_Month_Code_to_IMM_Date('H', year + 1)

def num_bus_days(start_date: datetime.date, end_date: datetime.date) -> int:
    """Calculate the number of business days between two dates.

    Args:
        start_date (datetime.date): The start date.
        end_date (datetime.date): The end date.

    Returns:
        int: The number of business days between the two dates.
    """
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    delta_days = (end_date - start_date).days
    business_days = 0
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date).to_pydatetime()

    for day in range(delta_days):
        current_day = start_date + datetime.timedelta(days=day)
        if current_day.weekday() < 5 and current_day not in holidays:  # Monday to Friday are business days
            business_days += 1

    return business_days


def prev_business_day(from_date: datetime.date, cal=USFederalHolidayCalendar()) -> datetime.date:
    """Get the previous business day before the given date.

    Args:
        from_date (datetime.date): The date from which to find the previous business day.

    Returns:
        datetime.date: The previous business day before the given date.
    """
    holidays = cal.holidays(start=from_date + datetime.timedelta(days=-14), end=from_date).to_pydatetime()

    prev_day = from_date + datetime.timedelta(days=-1)
    while prev_day.weekday() >= 5 or prev_day in holidays:  # Skip weekends and holidays
        prev_day += datetime.timedelta(days=-1)

    return prev_day

def next_business_day(from_date: datetime.date, cal=USFederalHolidayCalendar()) -> datetime.date:
    """Get the next business day after the given date.

    Args:
        from_date (datetime.date): The date from which to find the next business day.

    Returns:
        datetime.date: The next business day after the given date.
    """
    holidays = cal.holidays(start=from_date, end=from_date + datetime.timedelta(days=14)).to_pydatetime()

    next_day = from_date + datetime.timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in holidays:  # Skip weekends and holidays
        next_day += datetime.timedelta(days=1)

    return next_day


def day_adjust(date_in: datetime.date, cal=USFederalHolidayCalendar()) -> datetime.date:
    """Adjust the given date to the next business day if it falls on a weekend or holiday.

    Args:
        date (datetime.date): The date to adjust.
        cal: Holiday calendar to use for adjustments.
    Returns:
        datetime.date: The adjusted business day.
    """
    holidays = cal.holidays(start=date_in, end=date_in + datetime.timedelta(days=14)).to_pydatetime()
    
    date_out = date_in
    while date_out.weekday() >= 5 or date_out in holidays:
        date_out += datetime.timedelta(days=1)
    return date_out


def add_years(date_in: datetime.date, num_years: int, cal=USFederalHolidayCalendar(), day_convention = "Modified Following", eom=False) -> datetime.date:
    """ Add number of years to date and adjust date to next bus day unless goes over the month (Modified Following)"""
    year = date_in.year
    month = date_in.month
    day = date_in.day
    year_new = year + num_years
    last_day_of_month = calendar.monthrange(year_new, month)[1]
    if day>last_day_of_month:
        day = last_day_of_month
    date_new = datetime.date(year_new, month, day)
    if eom and day == calendar.monthrange(year, month)[1]:
        date_new = date(year_new, month, calendar.monthrange(year_new, month)[1])
    if day_convention == "Modified Following":
        date_new_adj = day_adjust(date_new)
        if date_new_adj.month > date_new.month or date_new_adj.year > date_new.year: # go backwards until bus day
            date_new_adj = prev_business_day(date_new_adj)
        date_new = date_new_adj
    return date_new

def add_months(date: datetime.date, num_months: int, cal=USFederalHolidayCalendar(), day_convention = "Modified Following", eom=False) -> datetime.date:
    """ Add number of months to date and adjust date to next bus day unless goes over the month (Modified Following)"""
    year = date.year
    month = date.month
    day = date.day
    month_new = month + num_months-1
    years_to_add = month_new // 12
    month_new = 1+(month_new % 12)
    #month_new = month + months_to_add
    year_new = year + years_to_add
    while month_new > 12:
        month_new -= 12
        year_new += 1
    last_day_of_month = calendar.monthrange(year_new, month_new)[1]
    day_new = day
    if day_new>last_day_of_month:
        day_new = last_day_of_month
    date_new = datetime.date(year_new, month_new, day_new)
    if eom and day == calendar.monthrange(year, month)[1]:
        date_new = datetime.date(year_new, month_new, calendar.monthrange(year_new, month_new)[1])
    if day_convention == "Modified Following":
        date_new_adj = day_adjust(date_new)
        if date_new_adj.month > date_new.month or date_new_adj.year > date_new.year: # go backwards until bus day
            date_new_adj = prev_business_day(date_new_adj)
        date_new = date_new_adj
    return date_new

def year_fraction(start: datetime, end: datetime, day_count_convention = "ACT/360") -> float:
        """Calculate year fraction using ACT/360."""
        days = (end - start).days
        if day_count_convention == "ACT/360":
            return days / 360.0
        return days / 365.25

def get_fed_funds_future_settlement_date(ticker) -> datetime:
    """Get the settlement date for a Fed Funds Future based on its ticker and transaction date.

    Args:
        ticker (str): The ticker symbol of the Fed Funds Future (e.g., 'FFH3').
        transaction_date (datetime): The date of the transaction.

    Returns:
        datetime: The settlement date of the Fed Funds Future.
    """
    month_code = ticker[2]
    if month_code not in month_mapping:
        raise ValueError(f"Invalid month code in ticker: {month_code}")
    year_code = ticker[3]
    month = month_mapping[month_code]
    year = 2020+int(year_code)  # Assuming year is in the format '3' for 2023
    start_date = datetime.date(year, month, 1)
    end_date = datetime.date(year, month, monthrange(year, month)[1])
    
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date).to_pydatetime()
    settlement_date = next_business_day(end_date)
    return settlement_date

def add_term(this_date: datetime.date, term: str, cal=USFederalHolidayCalendar()) -> datetime.date:
    """Add a term (e.g., '5Y', '6M') to a date and adjust to next business day if needed.

    Args:
        this_date (datetime.date): The starting date.
        term (str): The term to add (e.g., '5Y' for 5 years, '6M' for 6 months).
        cal: Holiday calendar to use for adjustments.
    Returns:
        datetime.date: The adjusted date after adding the term.
    """
    num = int(term[:-1])
    unit = term[-1]
    if unit == 'Y':
        new_date = add_years(this_date, num, cal)
    elif unit == 'M':
        new_date = add_months(this_date, num, cal)
    else:
        raise ValueError(f"Invalid term unit: {unit}")
    return new_date

def arrange_dates_chronologically(date_list: list) -> list:
    """Arrange a list of dates in chronological order.

    Args:
        date_list (list): A list of datetime.date objects.

    Returns:
        list: A list of datetime.date objects arranged in chronological order.
    """
    return sorted(date_list)

def arrange_terms_chronologically(term_list: list) -> list:
    """Arrange a list of terms (e.g., '5Y', '6M') in chronological order.

    Args:
        term_list (list): A list of term strings.

    Returns:
        list: A list of term strings arranged in chronological order.
    """
    def term_to_months(term: str) -> int:
        num = int(term[:-1])
        unit = term[-1]
        if unit == 'Y':
            return num * 12
        elif unit == 'M':
            return num
        else:
            raise ValueError(f"Invalid term unit: {unit}")

    return sorted(term_list, key=term_to_months)

def us_sifma_holiday_dates(year: int) -> set:
    """Return a set of US SIFMA holiday dates for the given year.

    Args:
        year (int): The year for which to get the holiday dates.
    Returns:        
        set: A set of datetime.date objects representing the holiday dates.
    """

    return {}

def fomc_meeting_dates(year: int) -> list:
    """Return a list of FOMC meeting dates for the given year.

    Args:
        year (int): The year for which to get the FOMC meeting dates.
    Returns:
        list: A list of datetime.date objects representing the FOMC meeting dates.  
    """

    return []