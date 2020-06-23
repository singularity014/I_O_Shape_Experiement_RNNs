'''
@author
Prafull SHARMA
'''
# ----------------------------------------------
from __future__ import print_function, division
from builtins import range, input
# Note: sudo pip install -U future
# for updating furutre package (if needed)
# ----------------------------------------------
from keras.models import Model
from keras.layers import Input,LSTM,GRU
import numpy as np
import matplotlib.pyplot as plt

'''
 T -> number of Time-Steps or Sequence Length
 D -> Dimension of each vector at each time step
 M -> Hidden Layer size
'''
T = 8
D = 2
M = 3

X = np.random.randn(1, T, D)
# This is just 1 sample of size T x D
# Think of it as a one sentence example sentence of T words and D vector length
print(X)


# First LSTM Model
def lstm1():
    input_ = Input(shape=(T, D))
    lstm_output = LSTM(M, return_state=True)(input_)
    model = Model(inputs=input_, outputs=lstm_output)
    o, h, c = model.predict(X)
    print('LSTM1 : ------')
    print(f'lstm1 output - {o}')
    print(f'lstm1 hidden state - {h}')
    print(f'lstm1 cell-state - {c}')
    print()

lstm1()

# First LSTM Model
def lstm2():
    input_ = Input(shape=(T, D))
    lstm_output = LSTM(M, return_state=True, return_sequences=True)(input_)
    model = Model(inputs=input_, outputs=lstm_output)
    o, h, c = model.predict(X)
    print('LSTM2 : ------')
    print(f'lstm2 output - {o}')
    print(f'lstm2 hidden state - {h}')
    print(f'lstm2 cell-state - {c}')
    print()

lstm2()