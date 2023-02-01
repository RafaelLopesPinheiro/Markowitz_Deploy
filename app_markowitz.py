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
    results = model.efficient_frontier(data, num_portfolios, risk_free_rate=0.05)
    
    
    col_names = ["Return", "Volatility", "Sharpe"] + [col for col in data.columns]
    results_df = model.create_result_df(results, col_names)
    print(results_df['Return'].mean())
    
    
    max_sharpe, min_volatility = model.max_sharpe_and_min_vol(results_df)
    graph_JSON = model.plotly_graph(results_df)

    return  render_template('predicted.html', data = [max_sharpe.to_html(), min_volatility.to_html(), graph_JSON, wrong_stock])


@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)