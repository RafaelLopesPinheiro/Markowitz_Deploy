from flask import Flask, request, render_template
import building_efficient_frontier as model
import datetime as dt
import graphs.graphs as graphs

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return  render_template('index.html')
    

@app.route('/predict', methods=['POST'])
def calculate_efficient_frontier():  
    print(request.form)
    stocks = model.form_stocks(request=request)
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2023, 1, 1)
    
    data, wrong_stock = model.get_clean_data(stocks, start, end)
    
    ## FIX EFFICIENT FRONTIER CALCULATIONS 
    num_portfolios = 10000
    results, weights = model.efficient_frontier_random(data, num_portfolios, 0.0)
    
    col_names = ["Return", "Volatility", "Sharpe"] + [col for col in data.columns]
    results_df = model.create_result_df(results, col_names)
    

    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    graph_efficient_frontier = graphs.plotly_ef_frontier(results_df, mean_returns, cov_matrix)
    
    max_sharpe, min_volatility = model.create_max_min_df(mean_returns, cov_matrix)
    max_sharpe_output, min_volatility_output = model.create_output_df(returns, max_sharpe, min_volatility)
    
    
    print('MIN_VOL_OUTPUT = ', min_volatility_output)
    print('Max_sharp_output  = ', max_sharpe_output)
    
    graph_portfolio_value_max_sharpe = graphs.plot_portfolio_value(data, max_sharpe_output)
    
    # ADD GRAPH MIN VARIANCE PORT TO OUTPUTS 
    # graph_portfolio_value_min_var = graphs.plot_portfolio_value(data, min_volatility_output)
    

    return  render_template('predicted.html', data = [max_sharpe_output.to_html(), min_volatility_output.to_html(), graph_efficient_frontier,
                                                      graph_portfolio_value_max_sharpe, wrong_stock])


@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)