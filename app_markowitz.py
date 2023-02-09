from flask import Flask, request, render_template
import building_efficient_frontier as model
import datetime as dt

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return  render_template('index.html')
    

@app.route('/predict', methods=['POST'])
def calculate_efficient_frontier():  
    
    stocks = model.form_stocks(request=request)
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2023, 1, 1)
    
    data, wrong_stock = model.get_clean_data(stocks, start, end)
    
    ## FIX EFFICIENT FRONTIER CALCULATIONS 
    num_portfolios = 10000
    results, weights = model.efficient_frontier(data, num_portfolios, 0.0)
    
    col_names = ["Return", "Volatility", "Sharpe"] + [col for col in data.columns]
    results_df = model.create_result_df(results, col_names)
    

    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    graph_JSON = model.plotly_scatter_graph(results_df, mean_returns, cov_matrix)
    
    max_sharpe, min_volatility = model.create_max_min_df(mean_returns, cov_matrix, stocks)
    model.print_outputs(max_sharpe, min_volatility, num_portfolios)  
    max_sharpe = model.create_sharpe_df(results_df)
    max_sharpe_proportions = max_sharpe[3:]*100
    min_volatility_proportions = min_volatility[3:]*100
    
    return  render_template('predicted.html', data = [max_sharpe.to_html(), min_volatility.to_html(), graph_JSON,
                                                      max_sharpe_proportions.to_html(), min_volatility_proportions.to_html(), wrong_stock])


@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)