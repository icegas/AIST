from matplotlib import pyplot as plt
import numpy as np
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.regularizers import L1L2
from math import sqrt
 
 
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df = df.drop(0)
	return df
 
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
def invert_scale(scaler, X, yhat):
	new_row = [x for x in X] + [yhat]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons, reg):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, kernel_regularizer=reg))#, return_sequences = True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model
 
# run a repeated experiment
def experiment(repeats, series, epochs, batch_size, neurons, reg):
	# transform data to be stationary
	raw_values = series.values
	diff_values = difference(raw_values, 1)
	# transform data to be supervised learning
	supervised = timeseries_to_supervised(diff_values, 1)
	supervised_values = supervised.values
	train_length = int(len(supervised_values) * 0.666) #2/3
	train, test = supervised_values[: train_length], supervised_values[train_length:]
	# transform the scale of the data
	scaler, train_scaled, test_scaled = scale(train, test)
	# run experiment
	error_scores = list()
	for r in range(repeats):
		# fit the model
		#batch_size = 4
		train_trimmed = train_scaled[2:, :]
		lstm_model = fit_lstm(train_trimmed, batch_size, epochs, neurons, reg)
		# forecast the entire training dataset to build up state for forecasting
		train_reshaped = train_trimmed[:, 0].reshape(len(train_trimmed), 1, 1)
		lstm_model.predict(train_reshaped, batch_size=batch_size)
		# forecast test dataset
		test_reshaped = test_scaled[:,0:-1]
		test_reshaped = test_reshaped.reshape(len(test_reshaped), 1, 1)
		output = lstm_model.predict(test_reshaped, batch_size=batch_size)
		predictions = list()
		for i in range(len(output)):
			yhat = output[i,0]
			X = test_scaled[i, 0:-1]
			# invert scaling
			yhat = invert_scale(scaler, X, yhat)
			# invert differencing
			yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
			# store forecast
			predictions.append(yhat)
		# report performance
		rmse = sqrt(mean_squared_error(raw_values[train_length:-2], predictions))
		print('%d) Test RMSE: %.3f' % (r+1, rmse))
		error_scores.append(rmse)
	return error_scores
 
 
# load dataset
#series = read_csv('/root/Desktop/AISTHack/code/shampoo-sales.csv')#, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# experiment

def Percent(arr):
	tmp = arr[0]

	arr = np.array([i * 100 / tmp for i in arr])

	return arr


def main():
    dataFile = '/root/Desktop/AISTHack/code/train/tickers_train.csv'
    data = read_csv(dataFile)
    temp = data['name']
    crypto = 'TRX'#"NANO" 
    TRX = np.array([])
    #print(data['priceBtc'].value)
    data = DataFrame(data)

    data = data[data['ticker'].str.contains(crypto)]

    for i in range(len(data)):
        TRX = np.append(TRX, data['priceBtc'].iloc[i])
    t = np.array([i for i in range(len(data))])

    TRX = Percent(TRX)
    series = Series(TRX)
    #plt.plot(t, TRX)
    #plt.savefig('TRXpercent.png')
    #plt.show()

    repeats = 3 #30
    results = DataFrame()
    # vary training epochs
    epochs = [50, 100]#, [500, 1000, 2000, 4000, 6000]
    regularizers = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
    for e in epochs:
    	results[str(e)] = experiment(repeats, series, e, 4, 1, regularizers[3])
    # summarize results
    print(results.describe())
    # save boxplot
    results.boxplot()


    #print(series.head())
    #series.plot()
    plt.show()

if __name__ == "__main__":
    main()