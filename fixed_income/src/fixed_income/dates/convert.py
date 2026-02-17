import datetime as dt
from .constants import SECONDS_IN_A_DAY, EPOCH_DATE

"""
Table of Contents:

1. convert_to_list
2. convert_date_to_num_days
3. convert_num_days_to_date

"""

def convert_to_list(not_list):
    """
    This function converts a non-list input to a list.
    
    Args:
        not_list (any): A single value or a list/tuple of values.
    
    Returns: 
        (list): A list containing the input value(s).
    """
    if isinstance(not_list, (list, tuple)):
        return list(not_list)
    else:
        return [not_list]

def convert_date_to_num_days(all_dates) -> list[float]:
    """
    This function converts date(s) to the number of days since Jan 1, 1970 (Unix epoch).
    
    Args:
        all_dates (list): A single date or a list/tuple of dates (datetime.date or datetime.datetime).
        
    Returns:
        num_days_list (list[float]): A list of floats representing the number of days since Jan 1, 1970 for each input date.
    """

    # First convert the input to a list no matter if it's a single date or a list of dates
    all_dates = convert_to_list(all_dates)

    # Check if all the values in the list are of type datetime
    num_days_list = []
    for one_date in all_dates:
        if not isinstance(one_date, (dt.date, dt.datetime)):
            raise TypeError("All input values must be of type datetime.datetime or datetime.date")
        
        # Convert date to datetime if it's of type date
        if isinstance(one_date, dt.date) and not isinstance(one_date, dt.datetime):
            one_date = dt.datetime(one_date.year, one_date.month, one_date.day)
        
        # Change or add timezone info to UTC if not already in UTC
            # !! This assumes that naive datetime objects are in UTC !!
        one_date = one_date.replace(tzinfo=dt.timezone.utc) if one_date.tzinfo is None else one_date.astimezone(dt.timezone.utc)

        # Calculate the number of days since epoch
        num_days_list.append((one_date - EPOCH_DATE).total_seconds() / SECONDS_IN_A_DAY)
        
    return num_days_list

# Convert number of days since epoch back to date
def convert_num_days_to_date(num_days) -> list[dt.datetime]:
    """
    This function converts number(s) of days since Jan 1, 1970 (Unix epoch) back to date(s).

    Args:
        num_days (list): A single float/int or a list/tuple of floats/ints representing number of days since Jan 1, 1970.
    
    Returns:
        date_list (list[dt.datetime]): A list of datetime.datetime objects corresponding to the input number of days since epoch.
    """

    # First convert the input to a list no matter if it's a single number or a list of numbers
    num_days = convert_to_list(num_days)

    date_list = []
    for one_num in num_days:
        # Check if all the values in the list are of type int or float
        if not isinstance(one_num, (int, float)):
            raise TypeError("All input values must be of type int or float")
        
        # Calculate the date from number of days since epoch
        date_list.append(EPOCH_DATE + dt.timedelta(days=one_num))
        
    return date_list

