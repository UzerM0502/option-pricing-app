import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import math
import scipy.stats as stats


def get_data(stocks, start, end):
    stock_data = pdr.get_data_yahoo(stocks, start, end)
    stock_data = stock_data['Close']
    returns = stock_data.pct_change()
    mean_returns = stock_data.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix


stock_list = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stock_list]
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

mean_returns, cov_matrix = get_data(stocks, start_date, end_date)
print(mean_returns)
print(cov_matrix)