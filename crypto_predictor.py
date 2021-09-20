import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import concurrent.futures
import time


start_time = time.perf_counter()

def crypto_prediction():
    crypto_currency = 'XRP'
    against_currency = 'USD'

    start_date = dt.datetime(2016, 1, 1)
    end_date = dt.datetime.now()

    data = web.DataReader(
        f'{crypto_currency}-{against_currency}', 'yahoo', start_date, end_date
        )


    # data preparation
    print(data.head())
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    prediction_days = 60    #60 days
    future_day = 30
    
    x_train, y_train = [], []
    
    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    
    # Creating the Neural Network
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))    # indicates the price from bunch of diff values into one number
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(x_train, y_train, epochs=25, batch_size=32)    # training data
    
    
    # Model Testing
    test_start_date = dt.datetime(2020, 1, 1)
    test_end_date = dt.datetime.now()
    
    test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start_date, test_end_date)
    
    actual_prices = test_data['Close'].values
    
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    
    
    model_inputs= total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)
    
    x_test = []
    
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days: x, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predition_prices = model.predict(x_test)
    predition_prices = scaler.inverse_transform(predition_prices)
    
    
    #plugging matplotlib
    plt.plot(actual_prices, color='black', label='Actual Prices')    
    plt.plot(predition_prices, color='blue', label='Predition prices') 
    plt.title(f'{crypto_currency} price prediction')
    plt.xlabel('Time')  # x axis
    plt.ylabel('Price')  # y axis
    plt.legend(loc='upper left')
    plt.show()
    

    # Predict the upcoming day
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days: len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"\n\n Prediction: {prediction}")
    

# Threading
if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(crypto_prediction())
        time.sleep(1)

end_time = time.perf_counter()
total_time = end_time - start_time
print(f'\nTotal time taken to analyze is {total_time} Seconds')