import datetime
import calendar
from pandas.tseries.holiday import USFederalHolidayCalendar

"""
Table of Contents:

1. num_bus_days
2. prev_business_day
3. next_business_day
4. day_adjust
5. add_years
6. add_months
7. add_term 

"""


def num_bus_days(start_date: datetime.date, end_date: datetime.date) -> int:
    """
    Calculate the number of business days between two dates.

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
    """
    Get the previous business day before the given date.

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
    """
    Get the next business day after the given date.

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
    """
    Adjust the given date to the next business day if it falls on a weekend or holiday.

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
    """ 
    Add number of years to date and adjust date to next bus day unless goes over the month (Modified Following)
    
    Args:
        date_in (datetime.date): The initial date.
        num_years (int): The number of years to add.
        cal: Holiday calendar to use for adjustments.
        day_convention (str): The day adjustment convention to use. Default is "Modified Following".
        eom (bool): If True, adjust to end of month if the original date is end of month.
    
    Returns:
        datetime.date: The new adjusted date.
    """

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
    """ 
    Add number of months to date and adjust date to next bus day unless goes over the month (Modified Following)
    
    Args:
        date (datetime.date): The initial date.
        num_months (int): The number of months to add.
        cal: Holiday calendar to use for adjustments.
        day_convention (str): The day adjustment convention to use. Default is "Modified Following".
        eom (bool): If True, adjust to end of month if the original date is end of month.
        
    Returns:
        datetime.date: The new adjusted date.
    """
    
    year = date.year
    month = date.month
    day = date.day
    month_new = month + num_months-1
    years_to_add = month_new // 12
    month_new = 1+(month_new % 12) # month_new = month + months_to_add

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