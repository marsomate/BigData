# Word embedding tensorflow example
# based on: https://www.tensorflow.org/tutorials/text/word_embeddings

import io
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import tensorflow_datasets as tfds

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)

padded_shapes = ([None],())
train_batches = train_data.padded_batch(10, padded_shapes = padded_shapes)
test_batches = test_data.padded_batch(10, padded_shapes = padded_shapes)

train_batch, train_labels = next(iter(train_batches))
print(train_batch.numpy().shape)
print(train_batch.numpy())
print(train_labels.numpy())

embedding_dim=16
vocab_size = info.features['text'].encoder.vocab_size
print(vocab_size)

model = models.Sequential([
  layers.Embedding(vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
  vec = weights[num+1] # skip 0, it's padding.
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()