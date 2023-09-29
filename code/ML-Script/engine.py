from ML_Pipeline import arima, dataset, preprocess, reg

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def main():
    df = dataset.read_data('Avocado.csv')
    X_train, X_test, y_train, y_test = preprocess.train_test(df)


    lr_score = reg.regression(X_train, X_test, y_train, y_test, 'lr', LinearRegression)
    rf_score = reg.regression(X_train, X_test, y_train, y_test, 'rf', RandomForestRegressor)
    xgb_score = reg.regression(X_train, X_test, y_train, y_test, 'xgb', XGBRegressor)
    arima_Score = arima.arima_model(df)

    mse_dict ={'Linear Regression': lr_score[2],
               'Random Forest': rf_score[2],
               'XGBoost': xgb_score[2],
               'ARIMA': arima_Score[2]}
    
    print('Best model is {} having a MSE of {}'.format(min(mse_dict, key=mse_dict.get),min(mse_dict.values)))