#import the required modules
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

# fix random seed for reproducibility
numpy.random.seed(7)

# define the raw dataset
alphabet="DIQMTQSPSSLSASVGDRVTITCKASQNIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCLQHISRPRTFGQGTVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"

# create mapping of characters to integers (0-213)
alphabet_num = [i for i in range (len(alphabet))]

# create mapping of integers to characters
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

#initialize the sequence length
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
  seq_in = alphabet_num[i:i + seq_length]
  seq_out = alphabet_num[i + seq_length]
  dataX.append([char for char in seq_in])
  dataY.append(seq_out)
  s_in = alphabet[i:i + seq_length]
  s_out = alphabet[i + seq_length]
  print(s_in, '->', s_out)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# create and fit the model
model = Sequential()
model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=2000, batch_size=len(dataX), verbose=2, shuffle=False)

# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

#create a function to generate prediction
def generate_prediction():
  subX='MTQ'
  res=alphabet.find(subX)
  pattern=[res,res+1,res+2]
  x = numpy.reshape(pattern, (1, len(pattern), 1))
  x = x / float(len(alphabet))
  prediction = model.predict(x, verbose=0)
  index = numpy.argmax(prediction)
  result = int_to_char[index]
  print(subX, "->", result)

generate_prediction()
