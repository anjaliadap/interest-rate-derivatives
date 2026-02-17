import pandas as pd

def read_excel_bbg_fff(
    path: str,
    ticker_col: str = "ticker",
    price_col: str = "close_price",
    oi_col: str = "open_interest"
) -> pd.DataFrame:

    df = pd.read_excel(path)

    # clean tickers
    df[ticker_col] = df[ticker_col].astype(str).str.strip()
    df[ticker_col] = df[ticker_col].astype(str).str.strip()
    df[ticker_col] = df[ticker_col].str.replace(" Comdty", "", regex=False)
    df[ticker_col] = df[ticker_col].str.replace("Comdty", "", regex=False)
    df[ticker_col] = df[ticker_col].str.replace(" ", "", regex=False)

    # convert numeric columns
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[oi_col] = pd.to_numeric(df[oi_col], errors="coerce")

    # drop rows missing key values
    df = df.dropna(subset=[ticker_col, price_col, oi_col])

    return df
