from keras.models import Sequential
import tensorflow as tf
from keras.layers import Embedding, Dense , GRU
rnn_nb = [32, 32] # Number of RNN nodes. Length of rnn_nb = number of RNN layers
fc_nb = [32] # Number of FC nodes. Length of fc_nb = number of FC layers
dropout_rate = 0.5 # Dropout after each layer
def get_model(_rnn_nb, _fc_nb):
 spec_start= tf.keras.Input(shape=(128,1))
 spec_x = spec_start
 for _r in _rnn_nb:
  spec_x = tf.keras.layer.GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True)(spec_x)
 for _f in _fc_nb:
  spec_x = TimeDistributed(Dense(_f))(spec_x)
  spec_x = Dropout(dropout_rate)(spec_x)
  spec_x = TimeDistributed(Dense(10))(spec_x)
  outputs = tf.keras.layers.Dense(units=2, activation='sigmoid',name='strong_out')(spec_x)
  model = tf.keras.Model(inputs=spec_start, outputs=outputs)
  print(model.summary())
  model.compile(
   optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.binary_crossentropy,
    #metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    metrics = ['accuracy']
  )
  model.summary()
 return model
##
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Embedding, Dense , GRU
dropout_rate = 0.5
inputs = tf.keras.Input(shape=(128,1))
x = tf.keras.layers.GRU(8,return_sequences=True)(inputs)
x = tf.keras.layers.GRU(16,return_sequences=False)(x)
spec_x = tf.keras.layers.Dropout(dropout_rate)(x)
outputs = tf.keras.layers.Dense(units=2, activation='sigmoid', name='strong_out')(spec_x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())


model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=tf.keras.losses.binary_crossentropy,metrics = ['accuracy'])
    #metrics=[tf.keras.metrics.sparse_categorical_accuracy]

train_history=model.fit(x_train3dD,x_labelonehot, epochs=30, batch_size=64,
          validation_split=0.2)
