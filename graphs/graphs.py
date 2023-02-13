import plotly
import plotly.graph_objs as go
import json


def plotly_ef_frontier(results_df, mean_returns, cov_matrix, risk_free_rate=0.003, download=False):
    from building_efficient_frontier import calc_port_perf, min_variance_port, max_sharp_ratio_port

    min_vol = min_variance_port(mean_returns, cov_matrix)
    ret_min_vol, std_min_vol = calc_port_perf(min_vol['x'], mean_returns, cov_matrix)
    min_vol_sharpe = (ret_min_vol - risk_free_rate) / std_min_vol


    max_sharpe = max_sharp_ratio_port(mean_returns, cov_matrix, risk_free_rate=0.003)
    ret_max_sharpe, std_max_sharpe = calc_port_perf(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_port = (ret_max_sharpe - risk_free_rate) / std_max_sharpe

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
            text=(results_df.Return - risk_free_rate)/results_df.Volatility
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
            x = [std_max_sharpe],
            y = [ret_max_sharpe],
            mode = 'markers',
            marker_symbol = 'star-dot',
            marker = dict(size=20),
            name = 'max_sharp_scipy',
            
            hovertemplate =
            '<i>Return</i>: %{y:.3f}'+
            '<br>Volatility: %{x:.3f}<br>'+
            'Sharpe: %{text:.3f}',
            text=[max_sharpe_port]
            ),
        ]

    layout = go.Layout(
            xaxis = dict(
                title = 'Volatility',
                tickformat=".1%"
            ), 
            yaxis = dict(
                title = 'Return',
                tickformat=".1%"
            ),
            width = 700,
            height = 500,
            title = 'Efficient Frontier'
    )

    fig = go.Figure(data=data, layout=layout)  # Create the figure
    fig.update_layout(legend = dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    # fig.update_layout(title='Efficient Frontier')  # Can update anything in the figure or data points (trace)
    
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    fig.show()
    if download:
        plotly.offline.plot(fig, filename='result.html')  # Download the figure if needed
    else:
        pass
    
    return graphJSON



def plot_portfolio_value(prices, max_sharpe, portfolio_init_value = 10000):
    """Build the portfolio performance since the beginning 

    Args:
        prices (_type_): _description_
        max_sharpe (_type_): _description_
        portfolio_value (int, optional): _description_. Defaults to 10000.
    """
    proportions = max_sharpe.T
    n_stocks = [(portfolio_init_value * proportions[i].values) / prices[i][0] for i in max_sharpe[3:].T]

    ## Multiply the initial number of stocks through out time

    for i, j in enumerate(prices):
        prices[f'{j}_port'] = (prices[j] * n_stocks[i])

    prices['port_total'] = prices.loc[:, prices.columns.values[len(n_stocks):]].sum(axis=1)  
    
     
    data = [go.Scatter(
            x=prices.index,
            y=prices.port_total,
            name='Portfolio',
            line=dict(
                width=1,
                
            )
    )]
    
    layout = go.Layout(
                xaxis = dict(
                    title = 'Years',
                    
                ), 
                yaxis = dict(
                    title = 'Portfolio Value',
               
                ),
                width = 700,
                height = 500,
                title = 'Portfolio Performance'
        )
    
    fig1 = go.Figure(data=data, layout=layout)
    fig1.update_layout(legend = dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig1.update_layout(hovermode='x')

    graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)    
    return graphJSON