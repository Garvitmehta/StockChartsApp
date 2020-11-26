from django.shortcuts import render
from .data_retriever import *
from .stock_models import *
from .plotting import *

# Create your views here.
def model(request):
	import requests
	import json

	# NJFJY7DW1WXTJ4U4 -- alphavantage
	if request.method == 'POST':

		ticker = request.POST['ticker']
		start_date = request.POST['start_date']
		end_date = request.POST['end_date']
		
		# get stock quote
		api_request = requests.get("https://cloud.iexapis.com/stable/stock/" + ticker + "/quote?token=pk_164c554030a54634b6851c5dec4dbe97")

		# Retrieve historical stock data 
		(data, returns_data, log_ret) = retrieve(ticker, start_date, end_date)

		# make plot of historical stock price data and returns 
		historical_price_plot = saveBasicPlot(data, "quotes/static/plots", "historical_plot.jpg")
		historical_returns_plot = saveReturnsPlot(returns_data, "quotes/static/plots", "returns_plot.jpg")
		# historical_log_returns_plot = savelog_ret_plot(log_ret , "quotes/static/plots", "log_returns_plot.jpg")
		
		# models
		(arma, arma_res, model_summary) = ARMA_model(data)
		# PACF(data)


		#f = open('quotes/static/model_results/ARMA_Summary.txt', 'r')
		#file_content = f.read()
		#f.close()

		try:
			api = json.loads(api_request.content)
		except Exception as e:
			api = "Error..."
		return render(request, 'model.html', {'api': api,'file_content':model_summary})
		
	else:
		return render(request, 'model.html', {'ticker': "Enter a ticker symbol above."})

def home(request):
	return render(request, 'home.html', {})

def about(request):
	return render(request, 'about.html', {})

