from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.metrics import root_mean_squared_error

def define_default_models(): 
    model_list = []

    linear_regression = LinearRegression()
    model_list.append(linear_regression)
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5)
    model_list.append(rf)
    xgb = xg.XGBRegressor(objective = 'reg:squarederror', n_estimators = 14, max_depth = 8, eval_metric = 'rmse')
    model_list.append(xgb)
    return model_list

def run_default_models(X_train, X_test, y_train, y_test): 
    results = defaultdict(dict)
    models = define_default_models()

    for model in models: 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_name = type(model).__name__
        results[model_name]['RMSE'] = root_mean_squared_error(y_test, y_pred)
    
    model_list = []
    for k,v in results.items(): 
        print(f"{k}")
        model_list.append(k)
        for k, v in v.items(): 
            print(f"{k} - {v}")
        print("\n")

    rmse_list = sorted(model_list, key = lambda x: (results[x]['RMSE']), reverse = True)
    print(f"In descending order of RMSE: {rmse_list}")


def run_model(X_train, X_test, y_train, y_test): 
    define_default_models()
    run_default_models(X_train, X_test, y_train, y_test)


        



