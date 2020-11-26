import numpy as np
import yfinance as yf

def retrieve(ticker_str, strt_date="2015-01-01", end_date="2020-01-01", ohlc='Close'):

	# columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
	data = yf.download(ticker_str, start=strt_date, end=end_date)

	# get returns from data
	returns_data = data[ohlc].pct_change(1)
	returns_data = returns_data.iloc[1:]
	log_ret = np.log(returns_data +1)
	
	
	return (data, returns_data, log_ret)