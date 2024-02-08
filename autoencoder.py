import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

words = ["cat", "dog", "apple", "orange", "car", "airplane", "man", "woman", "drink", "eat", "neural", "network",
         "tensor", "flow"]
dict_len = len(words)
word_index = dict((word, i) for i, word in enumerate(words))


def to_one_hot(word):
    return tf.one_hot(word_index[word], dict_len)


source_data = np.array([to_one_hot(word) for i, word in enumerate(words)])
train_data = tf.random.shuffle(source_data)

model = models.Sequential()
model.add(layers.Dense(3, activation='linear', input_shape=(dict_len,), use_bias=False))
model.add(layers.Dense(dict_len, activation='softmax'))
model.summary()

print(train_data)
print(tf.argmax(train_data, axis=1))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_data, epochs=2000, verbose=0)

print(tf.argmax(model.predict(train_data), axis=1))
print(model.layers[0].weights)