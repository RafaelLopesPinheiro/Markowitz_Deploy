import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import yfinance as yf
import datetime as dt
import plotly
import plotly.graph_objs as go
import scipy.optimize as sco
pd.set_option("display.max_rows", 100)


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


def negative_sharp(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    port_ret, port_var = calc_port_perf(mean_returns, weights, cov_matrix)
    return -(port_ret - risk_free_rate) / port_var


def max_sharp_ratio_port(mean_returns, cov_matrix, risk_free_rate=0.0):
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0, 1) for asset in range(n_assets) )
    
    optimizer = sco.minimize(negative_sharp, n_assets*[1./n_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimizer


def get_port_min_vol(weights, mean_returns, cov_matrix):
    min_vol = calc_port_perf(weights, mean_returns, cov_matrix)[1]
    return min_vol


def min_variance_port(mean_returns, cov_matrix):
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple ( (0.0, 1.0) for asset in range(n_assets))
    
    optimizer = sco.minimize(get_port_min_vol, n_assets*[1./n_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimizer



def efficient_frontier(df, num_portfolios, risk_free_rate=0.0):
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
    

def max_sharpe_and_min_vol(results_df):
    return (pd.DataFrame(results_df.iloc[results_df['Sharpe'].idxmax()])
            ,pd.DataFrame(results_df.iloc[results_df['Volatility'].idxmin()]))


def print_outputs(max_sharpe, min_volatility, num_portfolios):
    print('-'*70)
    print(f"Portfolio with max sharpe ratio in {num_portfolios} simulations\n{max_sharpe}")
    print('-'*70)
    print(f"Portfolio with min volatility in {num_portfolios} simulations\n{min_volatility}") 


def plotly_scatter_graph(results_df, mean_returns, cov_matrix, risk_free_rate=0, download=False):
    max_sharpe, min_volatility = max_sharpe_and_min_vol(results_df)

    
    min_vol = min_variance_port(mean_returns, cov_matrix)
    ret_min_vol, std_min_vol = calc_port_perf(min_vol['x'], mean_returns, cov_matrix)
    min_vol_sharpe = (ret_min_vol - risk_free_rate) / std_min_vol

    # Create trace for data, max_sharpe and min_vol portfolios
    data =[
        go.Scattergl(
        x = results_df.Volatility,
        y = results_df.Return,
        mode = 'markers',
        marker = dict(
            color = (results_df.Return - risk_free_rate)/results_df.Volatility,
            colorscale='RdBu',
            showscale=True,
            size=6,
            line= dict(width=1),
            colorbar=dict(title="Sharpe<br>Ratio")),
        name = 'portfolios',
        
        hovertemplate=
            '<i>Return</i>: %{y:.3f}'+
            '<br>Volatility: %{x:.3f}<br>'+
            'Sharpe: %{text:.3f}',
            text=results_df.Sharpe
            
        ),
                
        go.Scattergl(
            x = [std_min_vol],
            y = [ret_min_vol],
            mode = 'markers',
            marker_symbol = 'star-dot',
            marker = dict(size=20),
            name = 'min_volatility',
            
            hovertemplate =
            '<i>Return</i>: %{y:.3f}'+
            '<br>Volatility: %{x:.3f}<br>'+
            'Sharpe: %{text:.3f}',
            text=[min_vol_sharpe]
            ),
        
        go.Scattergl(
            x = max_sharpe.iloc[1],
            y = max_sharpe.iloc[0],
            mode = 'markers',
            marker_symbol = 'star-dot',
            marker = dict(size=20),
            name = 'max_sharpe',
            
            hovertemplate =
            '<i>Return</i>: %{y:.3f}'+
            '<br>Volatility: %{x:.3f}<br>'+
            'Sharpe: %{text:.3f}',
            text=max_sharpe.iloc[2]
            ),
        ]

    layout = go.Layout(
            xaxis = dict(
                title = 'Volatility',
                tickformat="%"
            ), 
            yaxis = dict(
                title = 'Return',
                tickformat="%"
            ),
            width = 700,
            height = 500,
            title = 'Efficient Frontier'
    )

    fig = go.Figure(data=data, layout=layout)  # Create the figure
    fig.update_layout(legend = dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    # fig.update_layout(title='Efficient Frontier')  # Can update anything in the figure or data points (trace)
    
    # fig.update_layout(hovermode='x')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # fig.show()
    if download:
        plotly.offline.plot(fig, filename='result.html')  # Download the figure if needed
    else:
        pass
    
    return graphJSON
    

def create_max_min_df(mean_returns, cov_matrix, stocks):
    max_sharp = max_sharp_ratio_port(mean_returns, cov_matrix)
    ret_max_sharpe, std_max_sharpe = calc_port_perf(max_sharp['x'], mean_returns, cov_matrix)

    min_var = min_variance_port(mean_returns, cov_matrix)
    ret_min_vol, std_min_vol = calc_port_perf(min_var['x'], mean_returns, cov_matrix)
    
    max_sharpe_df = pd.DataFrame(data={'Return': [ret_max_sharpe], 'Volatility':[std_max_sharpe]}, index=['Max_sharpe']).T
    max_sharpe_df.loc['Sharpe'] = max_sharpe_df['Max_sharpe'][0] / max_sharpe_df['Max_sharpe'][1]
    
    
    min_vol_df = pd.DataFrame(data={'Return': [ret_min_vol], 'Volatility':[std_min_vol]}, index=['Min_vol']).T
    min_vol_df.loc['Sharpe'] = min_vol_df['Min_vol'][0] / min_vol_df['Min_vol'][1]
    
    for i in range(0, len(mean_returns)):
        max_sharpe_df.loc[mean_returns.index[i]] = round(max_sharp['x'][i], 4)
        min_vol_df.loc[mean_returns.index[i]] = round(min_var['x'][i], 4)
        
    
    return max_sharpe_df, min_vol_df


def create_sharpe_df(results_df):
    max_sharpe = pd.DataFrame(data={"Max_Sharpe" : results_df.loc[results_df['Sharpe'].idxmax()]})
    return max_sharpe



def main():
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2023, 1, 1)
    stocks = ['NVDA','AMZN','GOOG','PETR4.SA']
    data, wrong_stock = get_clean_data(stocks, start, end)
    

    # Could do for 1000, 10000 and 100000 simulations and show 3 tables.
    num_portfolios = 100000
    results, weights = efficient_frontier(data, num_portfolios, 0.0)
    
    col_names = ["Return", "Volatility", "Sharpe"] + [col for col in data.columns]
    results_df = create_result_df(results, col_names)
    

    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    graph = plotly_scatter_graph(results_df, mean_returns, cov_matrix)
    
    max_sharpe, min_volatility = create_max_min_df(mean_returns, cov_matrix, stocks)
    print_outputs(max_sharpe, min_volatility, num_portfolios)  



if __name__ == "__main__":
    main()