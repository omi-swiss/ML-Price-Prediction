import pandas as pd

def read_data(filename):
    dataset = pd.read_csv(filename, index_col =0)
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset.set_index('Date', inplace = True)

    return dataset