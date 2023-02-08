import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import yfinance as yf
import datetime as dt
import plotly
import plotly.graph_objs as go
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


def efficient_frontier(df, num_portfolios, risk_free_rate=0):
    '''
    Returns the portfolios stdev, returns, sharpe ratio and weights of each stock.
    '''
    
    returns = df.pct_change()
    cov_matrix = returns.cov()
    mean_daily_returns = returns.mean()
    
    results = np.zeros((len(cov_matrix) + 3, num_portfolios))  # Need +3 columns to sharpe, returns and volatility
    
    for i in range(num_portfolios):

        weights = np.array(np.random.random(len(cov_matrix)))  # Create a array with random weights from 0.0 to 1.0
        weights /= np.sum(weights)  # Divide by the sum of weights, making sum of array equals to 1.0

        portfolio_return = np.sum(mean_daily_returns *  weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized std deviation (volatility)
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = (results[0, i] - risk_free_rate)/ results[1, i]  # Sharpe ratio formula
        
        for j in range(len(weights)):
            results[j+3, i] = weights[j]
    return results


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


def plotly_graph(results_df):
    max_sharpe, min_volatility = max_sharpe_and_min_vol(results_df)
    
    # create a trace for data, max_sharpe and min_vol
    data =[
        go.Scattergl(
        x = results_df.Volatility,
        y = results_df.Return,
        mode = 'markers',
        marker = dict(
            color = results_df.Volatility/results_df.Return,
            colorscale='RdBu',
            showscale=True,
            size=6,
            line= dict(width=1),
            colorbar=dict(title="Sharpe<br>Ratio")),
        name = 'data'
        ),
        
        go.Scattergl(
            x = max_sharpe.iloc[1].values,
            y = max_sharpe.iloc[0].values,
            mode = 'markers',
            marker_symbol = 'star-dot',
            marker = dict(size=20),
            name = 'max_sharpe'
            ),
        
        go.Scattergl(
            x = min_volatility.iloc[1].values,
            y = min_volatility.iloc[0].values,
            mode = 'markers',
            marker_symbol = 'star-dot',
            marker = dict(size=20),
            name = 'min_volatility'
            )]

    layout = go.Layout(
            xaxis = dict(
                title = 'Volatility',
            ), 
            yaxis = dict(
                title = 'Return'
            ),
            width = 900,
            height = 600,
            title = 'Efficient Frontier'
    )

    fig = go.Figure(data=data, layout=layout)  # Create the figure
    fig.update_layout(legend = dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    # fig.update_layout(title='Efficient Frontier')  # Can update anything in the figure or data points (trace)
    # plotly.offline.plot(fig, filename='result.html')  # Download the figure if needed
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    fig.show()
    
    return graphJSON
    

    

def main():
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2023, 1, 1)
    stocks = ['AAPL','AMZN','GOOG','NVDA']
    data, wrong_stock = get_clean_data(stocks, start, end)
    

    # Could do for 1000, 10000 and 100000 simulations and show 3 tables.
    num_portfolios = 10000
    results = efficient_frontier(data, num_portfolios, 0.05)
    
    col_names = ["Return", "Volatility", "Sharpe"] + [col for col in data.columns]
    results_df = create_result_df(results, col_names)
    print(results_df['Return'].mean())
    
    max_sharpe, min_volatility = max_sharpe_and_min_vol(results_df)  # Return the row of max_sharpe ratio and minimum volatility
    print_outputs(max_sharpe, min_volatility, num_portfolios)

    
    graph_JSON = plotly_graph(results_df)


if __name__ == "__main__":
    main()