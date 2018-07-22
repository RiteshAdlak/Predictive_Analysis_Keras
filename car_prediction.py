import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
 
np.random.seed(101)

df = pd.read_csv('x_test.csv')

array = df.values

X = array[:,0:20] 
Y = array[:,20]

df.head()

model = Sequential()
model.add(Dense(21, input_dim=20, activation='relu' ,kernel_initializer='uniform'))
model.add(Dense(32,  activation='relu' ,kernel_initializer='uniform'))
model.add(Dense(32,  activation='relu' ,kernel_initializer='uniform'))
model.add(Dense(1,  activation='sigmoid' ,kernel_initializer='uniform'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=100, batch_size=50)

result = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], result[1]*100))




























