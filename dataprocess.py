import jieba
import tensorflow  as tf
from PIL import Image

import Dictionary
import configuration


def get_caption():
    caption = {}
    pic_path = "/disk4/flick8k"

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

    return caption


def write_train_tfrecord():
    dic = Dictionary.Dictionary()
    config = configuration.TrainConfg()
    UNKOWN = config.UNKNOWN
    EOF = config.EOF

    caption = get_caption()

    train_set = []
    with open("data/flickr8ktrain.txt") as fin:
        for each in fin.readlines():
            train_set.append(each.strip("\n"))

    image = []
    cap = []
    for each in train_set:
        for each_caption in caption[each]:
            image.append((config.image_path + '/' + each + ".jpg"))

            tokens = jieba.cut(each_caption)
            caption_idx = []
            for each_token in tokens:
                if each_token in dic.word_idex:
                    caption_idx.append(dic.word_idex[each_token])
                else:
                    caption_idx.append(UNKOWN)
            caption_idx.append(EOF)
            cap.append(caption_idx)

    writer = tf.python_io.TFRecordWriter(config.image_path + "/train.tfrecords")
    for i in range(image.__len__()):
        x = Image.open(image[i])
        x = x.resize([299, 299])
        x_raw = x.tobytes()
        y = cap[i]
        example = tf.train.Example(features=tf.train.Features(feature={
            "caption": tf.train.Feature(int64_list=tf.train.Int64List(value=y)),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_raw]))
        }))
        writer.write(example.SerializeToString())

    writer.close()


def read_tfrecord_data(example):
    features = {"image": tf.FixedLenFeature([], tf.string), "caption": tf.VarLenFeature(tf.int64)}
    parse_data = tf.parse_single_example(example, features)
    image = tf.decode_raw(parse_data["image"], tf.uint8)
    image = tf.reshape(image, [299, 299, 3])

    caption = tf.sparse_tensor_to_dense(parse_data["caption"])
    caption = tf.cast(caption, tf.int16)

    return {"image": image, "caption": caption}

write_train_tfrecord()