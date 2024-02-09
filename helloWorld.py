
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

text = "Hello World!"

input_text = text[:-1]
target_text = text[1:]
print(input_text)
print(target_text)

chars = sorted(set(text))
chars_len = len(chars)
char_index = dict((char, i) for i, char in enumerate(chars))
print(char_index)


def to_one_hot(char):
    return tf.one_hot(char_index[char], chars_len)


source_data = np.array([to_one_hot(char) for i, char in enumerate(input_text)])
target_data = np.array([to_one_hot(char) for i, char in enumerate(target_text)])
source_data = np.expand_dims(source_data, axis=0)
target_data = np.expand_dims(target_data, axis=0)
print(source_data.shape)

#rnn = layers.SimpleRNN(units=4, return_sequences=True)
#out = rnn(source_data)
#print(out.shape)

model = models.Sequential([
    layers.SimpleRNN(units=4, return_sequences=True, input_shape=(source_data.shape[1], chars_len,)),
    layers.Dense(chars_len, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(source_data, target_data, epochs=1000)

model.summary()

print(tf.argmax(source_data, axis=2))
print(tf.argmax(target_data, axis=2))
print(tf.argmax(model.predict(source_data), axis=2))