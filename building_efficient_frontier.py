import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import scipy.optimize as sco
import graphs.graphs as graphs


def form_stocks(request):
    return [request.form[key].upper() for key in request.form.keys()
            if request.form[key] != '']


def get_clean_data(stock_list, start, end): # Download and clean data from yahoo finance
    data = yf.Tickers(stock_list)
    data = data.history(start=start, end=end)['Close']
    data.ffill(axis=0, inplace=True)
    data.bfill(axis=0, inplace=True)
    
    if data.isnull().values.any():
        data.dropna(axis=1, inplace=True)
        
    wrong_stock = [stock for stock in stock_list 
                   if stock not in data.columns]
    
    return data, wrong_stock


def calc_port_perf(weights, mean_returns, cov_matrix): 
    portfolio_return = np.sum(mean_returns * weights) * 252  # 252 represent a year of trading days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)    
    return portfolio_return, portfolio_std


def max_sharp_ratio_port(mean_returns, cov_matrix, risk_free_rate=0.003):
    
    def negative_sharp(weights, mean_returns, cov_matrix, risk_free_rate=0.003):
        port_ret, port_var = calc_port_perf(weights, mean_returns, cov_matrix)
        return -(port_ret - risk_free_rate) / port_var
    
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0.0, 1.0) for asset in range(n_assets) )
    
    optimizer = sco.minimize(negative_sharp, n_assets*[1./n_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimizer


def min_variance_port(mean_returns, cov_matrix):
    
    def get_port_min_vol(weights, mean_returns, cov_matrix): 
        return calc_port_perf(weights, mean_returns, cov_matrix)[1]
    
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple ( (0.0, 1.0) for asset in range(n_assets))
    
    optimizer = sco.minimize(get_port_min_vol, n_assets*[1./n_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimizer


def efficient_frontier_random(df, num_portfolios, risk_free_rate=0.003):
    '''
    Calculate the portfolios stdev, returns, sharpe ratio and weights of each stock and return as np.array.
    '''
    
    returns = df.pct_change().dropna()
    cov_matrix = returns.cov()
    mean_daily_returns = returns.mean()

    
    results = np.zeros((len(cov_matrix) + 3, num_portfolios))  # Need +3 columns to sharpe, returns and volatility
    weights_record = []
    for i in range(num_portfolios):

        weights = np.array(np.random.random(len(cov_matrix)))  # Create a array with random weights from 0.0 to 1.0
        weights /= np.sum(weights)  # Divide by the sum of weights, making sum of array equals to 1.0
        
        weights_record.append(weights) 
        
        portfolio_return, portfolio_std_dev = calc_port_perf(weights, mean_daily_returns, cov_matrix)

        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = (results[0, i] - risk_free_rate)/ results[1, i]  # Sharpe ratio formula
        
        for j in range(len(weights)):
            results[j+3, i] = weights[j]
    return results, weights_record


def create_result_df(results, col_names):
    results_df = pd.DataFrame(results.T)
    results_df.columns = [name for name in col_names]    
    return results_df
    

def print_outputs(max_sharpe, min_volatility, num_portfolios):
    print('-'*70)
    print(f"Portfolio with max sharpe ratio in {num_portfolios} simulations\n{max_sharpe}")
    print('-'*70)
    print(f"Portfolio with min volatility in {num_portfolios} simulations\n{min_volatility}") 


def create_max_min_df(mean_returns, cov_matrix, risk_free_rate=0.003):
    max_sharp = max_sharp_ratio_port(mean_returns, cov_matrix)
    ret_max_sharpe, std_max_sharpe = calc_port_perf(max_sharp['x'], mean_returns, cov_matrix)

    min_var = min_variance_port(mean_returns, cov_matrix)
    ret_min_vol, std_min_vol = calc_port_perf(min_var['x'], mean_returns, cov_matrix)
    
    max_sharpe_df = pd.DataFrame(data={'Return': [ret_max_sharpe], 'Volatility':[std_max_sharpe]}, index=['Max_sharpe']).T
    max_sharpe_df.loc['Sharpe'] = (max_sharpe_df['Max_sharpe'][0] - risk_free_rate) / max_sharpe_df['Max_sharpe'][1]
    
    
    min_vol_df = pd.DataFrame(data={'Return': [ret_min_vol], 'Volatility':[std_min_vol]}, index=['Min_vol']).T
    min_vol_df.loc['Sharpe'] = (min_vol_df['Min_vol'][0] - risk_free_rate) / min_vol_df['Min_vol'][1]
    
    for i in range(0, len(mean_returns)):
        max_sharpe_df.loc[mean_returns.index[i]] = round(max_sharp['x'][i], 4)
        min_vol_df.loc[mean_returns.index[i]] = round(min_var['x'][i], 4)
        
    
    return max_sharpe_df, min_vol_df


def sortino_ratio(returns, N=252, risk_free_rate=0.003, min=False):
    if min:
        weights_min = min_variance_port(returns.mean(), returns.cov())
        port_returns = (weights_min['x'] * returns).sum(axis=1)
        mean = port_returns.mean() * N -risk_free_rate
        std_neg = port_returns.loc[port_returns < 0].std()*np.sqrt(N)
    
    else:
        weights_max = max_sharp_ratio_port(returns.mean(), returns.cov())
        port_returns = (weights_max['x'] * returns).sum(axis=1)
        mean = port_returns.mean() * N -risk_free_rate
        std_neg = port_returns.loc[port_returns < 0].std()*np.sqrt(N) 
    
    return mean/std_neg   
    

def create_output_df(returns, max_sharpe_df, min_var_df):
    sortino_min_var = sortino_ratio(returns, N=252, risk_free_rate=0.003, min=True)
    sortino_max_sharpe = sortino_ratio(returns, N=252, risk_free_rate=0.003)

    max_sharpe = pd.concat([max_sharpe_df.iloc[:3]['Max_sharpe'],
                            pd.Series({'Sortino' : sortino_max_sharpe}),
                            max_sharpe_df.iloc[3:]['Max_sharpe']])
    
    min_var = pd.concat([min_var_df.iloc[:3]['Min_vol'],
                         pd.Series({'Sortino' : sortino_min_var}),
                         min_var_df.iloc[3:]['Min_vol']])
    
    return max_sharpe, min_var


def main():
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2023, 1, 1)
    stocks = ['AAPL','AMZN','GOOG','VYM', 'NVDA', 'TSLA', 'PETR4.SA']
    data, wrong_stock = get_clean_data(stocks, start, end)
    

    # Could do for 1000, 10000 and 100000 simulations and show 3 tables.
    num_portfolios = 10000
    results, weights = efficient_frontier_random(data, num_portfolios, 0.0)

    
    col_names = ["Return", "Volatility", "Sharpe"] + [col for col in data.columns]
    results_df = create_result_df(results, col_names)
    
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    

    graph = graphs.plotly_ef_frontier(results_df, mean_returns, cov_matrix)
    max_sharpe, min_volatility = create_max_min_df(mean_returns, cov_matrix)

    # print_outputs(max_sharpe, min_volatility, num_portfolios)


    max_sharpe_output, min_var_output = create_output_df(returns, max_sharpe, min_volatility)
    print(max_sharpe_output, '\n', min_var_output)

    graphs.plot_portfolio_value(data,  max_sharpe_output)  
    
    
    

if __name__ == "__main__":
    main()