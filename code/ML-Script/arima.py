from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def arima_model(data):
    df_ar = data.drop(columns=['Total Volume', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'region'])
    df_ar =df_ar.resample('W').mean()

    X=df_ar.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    for i in range(len(test)):
        model = ARIMA(history, order=(1,0,0))
        model_fit = model.fit(disp=0)
        output=model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[i]
        history.append(obs)

    r2 = r2_score(test,predictions)
    mae = mean_absolute_error(test,predictions)
    mse = mean_squared_error(test,predictions)
    return r2, mae, mse