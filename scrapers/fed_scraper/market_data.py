
import pandas as pd
import yfinance as yf
from fredapi import Fred
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Setup API
FRED_API_KEY = '8c7a02f395ed21e87fcb2d424bff3f63'
fred = Fred(api_key=FRED_API_KEY)

def get_market_reality():
    start_date = '2010-01-01'

    print("Fetching USD/JPY data...")
    usdjpy = yf.download("JPY=X", start=start_date)
    usdjpy.columns = usdjpy.columns.get_level_values(0)

    print("Fetching Macro Yields and Oil...")
    tickers = {
        'yield_10y': 'DGS10',
        'yield_2y': 'DGS2',
        'oil_wti': 'DCOILWTICO',
        'vix': 'VIXCLS'
    }

    macro_data = pd.DataFrame()
    for name, ticker in tickers.items():
        macro_data[name] = fred.get_series(ticker, observation_start=start_date)

    combined = usdjpy['Close'].to_frame(name='usd_jpy_close').join(macro_data, how='inner')

    return combined

data = get_market_reality()
print(data.tail())
data.to_csv('market_data_benchmark.csv')