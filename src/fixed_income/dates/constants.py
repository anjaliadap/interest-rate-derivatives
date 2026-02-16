import datetime

SECONDS_IN_A_DAY = 86400.0
EPOCH_DATE = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

MONTH_MAPPING = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4,
    'K': 5, 'M': 6, 'N': 7, 'Q': 8,
    'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

IMM_MONTHS = [3, 6, 9, 12]