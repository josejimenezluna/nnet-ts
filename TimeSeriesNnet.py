import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import StandardScaler


class TimeSeriesNnet(object):
	def __init__(self, hidden_layers, activation_functions):
		self.hidden_layers = hidden_layers
		self.activation_functions = activation_functions

		if len(self.hidden_layers) != len(self.activation_functions):
			raise Exception("hidden_layers size must match activation_functions size")

	def fit(self, timeseries, lag, epochs):
		self.timeseries = np.array(timeseries) # Apply log transformation por variance stationarity
		self.lag = lag		
		self.n = len(timeseries)
		self.X = np.zeros((self.n - self.lag, self.lag))
		self.y = np.log(self.timeseries[self.lag:]) 
		self.epochs = epochs
		self.scaler = StandardScaler()

		print "Building regressor matrix"
		# Building X matrix
		for i in range(0, self.n - lag):
			self.X[i, :] = self.timeseries[range(i, i + lag)]

		print "Scaling data"
		self.scaler.fit(self.X)
		self.X = self.scaler.transform(self.X)

		print "Checking network consistency"
		# Neural net architecture
		self.nn = Sequential()
		self.nn.add(Dense(self.X.shape[1], self.hidden_layers[0]))
		self.nn.add(Activation(self.activation_functions[0]))

		for i, layer in enumerate(self.hidden_layers[:-1]):
			self.nn.add(Dense(self.hidden_layers[i], self.hidden_layers[i + 1]))
			self.nn.add(Activation(self.activation_functions[i]))

		# Add final node
		self.nn.add(Dense(self.hidden_layers[-1], 1))
		self.nn.compile(loss = 'mean_absolute_error', optimizer = 'sgd')

		print "Training neural net"
		# Train neural net
		self.nn.fit(self.X, self.y, nb_epoch = self.epochs)

	def predict(self):
		# Doing weird stuff to scale *only* the first value
		self.next_X = np.concatenate((np.array([self.y[-1]]), self.X[-1, :-1]), axis = 0)
		self.next_X = self.next_X.reshape((1, self.lag))
		self.next_X = self.scaler.transform(self.next_X)
		self.valid_x = self.next_X[0, 0]
		# Doing it right now
		self.next_X = np.concatenate((np.array([self.valid_x]), self.X[-1, :-1]), axis = 0)
		self.next_X = self.next_X.reshape((1, self.lag))		 
		self.next_y = self.nn.predict(self.next_X)
		return np.exp(self.next_y)

	def predict_ahead(self, n_ahead):
		# Store predictions and predict iteratively
		self.predictions = np.zeros(n_ahead)

		for i in range(n_ahead):
			self.current_x = self.timeseries[-self.lag:]
			self.current_x = self.current_x.reshape((1, self.lag))
			self.current_x = self.scaler.transform(self.current_x)
			self.next_pred = self.nn.predict(self.current_x)
			self.predictions[i] = np.exp(self.next_pred[0, 0])
			self.timeseries = np.concatenate((self.timeseries, np.exp(self.next_pred[0,:])), axis = 0)

		return self.predictions

