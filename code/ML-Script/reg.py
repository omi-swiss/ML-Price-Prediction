from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def regression(X_train, X_test, y_train, y_test, model_name, model):
    pipe = Pipeline([('scaler', StandardScaler()), (model_name, model())])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae= mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return r2, mae, mse