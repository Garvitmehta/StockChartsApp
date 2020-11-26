from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from .plotting import *
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def ARMA_model(data, ohlc='Close'):

	data = data[ohlc]

	# choose best p, q parameters for our model using AIC optimization
	params = bestParams(data)
	model = ARIMA(data, order=(params[0], 0, params[2]))
	res = model.fit()

	#model_summary = res.summary().as_text()
	model_summary = res.summary()
	# write summary to file
	#fileobj = open("quotes/static/model_results/ARMA_Summary.txt", 'w')
	#fileobj.write(model_summary.as_text())
	#fileobj.close()

	fig, ax = plt.subplots(figsize=(10,8))
	ax = data.plot(ax=ax)
	fig = plot_predict(res, start=data.index[0], end=data.index[-1], ax=ax, plot_insample=False)
	legend = ax.legend(["Actual price", "Forecast", "95% Confidence Interval"], loc='upper left')

	fig.savefig("quotes/static/plots/forecast_vs_actual.jpg")
	return (model, res, model_summary)

def bestParams(data):

	ps = range(0, 8, 1)
	d = 1
	qs = range(0, 8, 1)

	# Create a list with all possible combination of parameters
	parameters = product(ps, qs)
	parameters_list = list(parameters)
	order_list = []

	for each in parameters_list:
	    each = list(each)
	    each.insert(1, 1)
	    each = tuple(each)
	    order_list.append(each)

	result_df = AIC_optimization(order_list, exog=data)
	return result_df['(p, d, q)'].iloc[0]

def AIC_optimization(order_list, exog):
    """
        Return dataframe with parameters and corresponding AIC
        
        order_list - list with (p, d, q) tuples
        exog - the exogenous variable
    """
    
    results = []
    
    for order in order_list:
        try: 
            model = SARIMAX(exog, order=order).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        results.append([order, model.aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']

    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df


# def PACF(data, ohlc='Close'):
# 	data = data[ohlc]
# 	returns = 100 * data.pct_change()

# 	model = arch_model(returns, p=3, q=0)
# 	model_fit = model.fit()
# 	model_fit.summary()
# 	rolling_predictions = []
# 	test_size = 365

# 	for i in range(test_size):
# 		train = returns[:-(test_size-i)]
# 		model = arch_model(train, p=3, q=0)
# 		model_fit = model.fit(disp='off')
# 		pred = model_fit.forecast(horizon=1)
# 		rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

# 	rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])


# 	plt.figure(figsize=(10,4))
# 	true, = plt.plot(returns[-365:])
# 	preds, = plt.plot(rolling_predictions)
# 	plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
# 	plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)

# 	fig.savefig("quotes/static/plots/.jpg")
