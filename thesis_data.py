#%%
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class dataloader():
    # function to load data
    def data_function(squared, trainsize, valsize, testsize, input_window, output_window):
        # load data set
        df = pd.read_csv('^GDAXI-10y.csv', delimiter=',') 
        # linear interpolation for missing values
        df['Adj Close'].interpolate(method='index', inplace=True)
        time = np.asarray(df['Date'])
        price = np.asarray(df['Adj Close'])
        log_returns = np.diff(np.log(price)) 

        if squared==False:
            data = log_returns 
        
        if squared==True:
            data = log_returns**2 
        # split data set
        training_size   = int(trainsize*len(data))
        test_size       = int(testsize*len(data))
        validation_size = int(valsize*len(data))

        train_data = data[:training_size]                     
        val_data   = data[training_size:len(data)-test_size]
        test_data  = data[training_size+validation_size+2:]
        train_val = data[:-len(test_data)]
        # apply minmaxscaler
        scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(train_data.reshape(-1, 1))
        train_arr = scaler1.transform(train_data.reshape(-1, 1))
        val_arr = scaler1.transform(val_data.reshape(-1, 1))  
        scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(train_val.reshape(-1, 1))     
        test_arr = scaler2.transform(test_data.reshape(-1, 1))

        # Generate Sequence 
        def transform_data(arr, input_window, output_window):
            x = np.asarray([arr[i : i + input_window] for i in range(len(arr) - input_window)])
            y = np.asarray([arr[i + output_window : i + input_window + output_window] for i in range(len(arr) - input_window)])
            x_var = torch.FloatTensor(torch.from_numpy(x).float())
            y_var = torch.FloatTensor(torch.from_numpy(y).float())
            return x_var, y_var
    
        x_train, y_train = transform_data(train_arr, input_window, output_window)
        x_val, y_val = transform_data(val_arr, input_window, output_window)
        x_test, y_test = transform_data(test_arr, input_window, output_window)

        return x_train, y_train, x_val, y_val, x_test, y_test, scaler1, scaler2, time, price
    # function to generate batches during training and evaluation
    def generate_batch_data(x, y, batch_size):
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch



