import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
class Datareading():
    def __init__(self,dataframe):
        self.df = dataframe
    def dataprocessed(self):
        dff = pd.read_csv(self.df)
        Independent_data = dff.drop(["Rating"], 1)
        Response = dff["Rating"]
        return Independent_data,Response

def __Processing__(df,process = 'min-max'):
    if process == 'min-max':
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(df)
    else:
        process == 'standard'
        scaler = StandardScaler()
        dataset = scaler.fit_transform(df)
    return scaler,dataset

def next_batch(x,batch_size):
    def as_batch(datafX, start, count):
        if len(datafX.shape) == 3:
            return datafX[start:start + count,:,:]
        else:
            return datafX[start:start + count,:]
    for i in range(0, x.shape[0], batch_size):
        yield as_batch(x, i, batch_size)
