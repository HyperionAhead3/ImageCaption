import tensorflow  as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.learn as learn
import configuration

train_set = []
caption = {}
config=configuration.ModelConfig
pic_path=""

with open("data/baidu.txt") as baidu:
    for eachline in baidu.readlines():
        sen = eachline.split()
        pic = sen[0].split(".")[0]
        cap = sen[1]
        if pic in caption:
            caption[pic].append(cap)
        else:
            caption[pic] = [cap]

with open("data/google.txt") as google:
    for eachline in google.readlines():
        sen = eachline.split()
        pic = sen[0].split(".")[0]
        cap = sen[1]
        if pic in caption:
            caption[pic].append(cap)
        else:
            caption[pic] = [cap]

with open("data/humanwrite.txt") as human:
    for eachline in human.readlines():
        sen = eachline.split()
        pic = sen[0].split(".")[0]
        cap = sen[1]
        if pic in caption:
            caption[pic].append(cap)
        else:
            caption[pic] = [cap]

with open("data/flickr8ktrain.txt") as fin:
    for each in fin.readlines():
        train_set.append(each)

for pic in train_set:
    image_path=

dataset = tf.data.Dataset.from_tensor_slices()
