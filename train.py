import tensorflow  as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.learn as learn

train_set = []
caption = []

with open("data/flickr8ktrain.txt") as fin:
    for each in fin.readlines():
        train_set.append(each)


dataset = tf.data.Dataset.from_tensor_slices()
