import os

from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
from tensorflow.contrib.rnn import *
from tensorflow.python import pywrap_tensorflow

from configuration import *
from inception_v3 import *


class SCA_CNN(object):
    def __init__(self, mode, image, caption, is_training_inception, is_first_time):
        self.config = ModelConfig()
        self.is_first_time = is_first_time
        self.initializer = tf.random_uniform_initializer(-self.config.initializer_scale, self.config.initializer_scale)
        self.mode = mode
        self.image = image
        self.caption = caption
        self.caption_len = None
        self.target = None
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.batch_input()
        self.build(is_training_inception)

        assert mode in ["train", "eval", "inference"]
        if mode == "train":
            self.regularizer = l2_regularizer(0.00004)
        else:
            self.regularizer = None

    def variable_init(self):
        if (self.is_first_time):
            print("first load Inception weight")
            saver = tf.train.Saver(self.var_in_inception)
            path = self.config.ft_ckp_path
        else:
            saver = tf.train.Saver()
            path = self.config.model_path

        def initial_all_variable(sess):
            if (os.path.isfile(path)):
                print("initial_variable")
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, self.config.model_path)
            else:
                raise FileNotFoundError("check point file not found")

    def embedding_image(self, image_feature):
        # 8*8*2048
        # embedding the feature into embedding space
        with tf.variable_scope("image_embedding"):
            shape = image_feature.get_shape()
            net = slim.avg_pool2d(image_feature, shape[1:3], scope="avg_pool")
            net = slim.dropout(net, keep_prob=self.config.dropout_keep_prob, is_training=(self.mode == "train"))
            net = slim.flatten(net, scope="flatten")
            net = slim.fully_connected(net, self.config.embedding_size, activation_fn=None,
                                       weights_initializer=self.initializer, biases_initializer=None)
            return net

    def embedding_seq(self, word_idx_input):
        with tf.variable_scope("word_embedding"):
            word_embedding = tf.get_variable("embedding", [self.config.word_count, self.config.embedding_size],
                                             tf.float32,
                                             tf.initializers.random_uniform(-1, 1))
            return tf.nn.embedding_lookup(word_embedding, word_idx_input)

    def batch_input(self):
        self.caption_len =

    def build(self, is_training_inception):
        # get variables in inception v3
        reader = pywrap_tensorflow.NewCheckpointReader(self.config.ft_ckp_path)
        var_in_inception_name = reader.get_variable_to_shape_map()
        with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training_inception, 'updates_collections': None}):
            feature, inception_layers = inception_v3_base(self.image)
            self.var_in_inception = []
            with tf.variable_scope("", reuse=True):
                for v in var_in_inception_name:
                    var = slim.get_variables_by_name(v)
                    if var:
                        self.var_in_inception.append(var[0])

        # get variables not in inception v3
        self.var_not_in_inception = []
        for v in tf.trainable_variables():
            if v not in self.var_in_inception:
                self.var_not_in_inception.append(v)

        # embedding image feature
        image_embedding = self.embedding_image(feature)
        # [batch_size,sen_len,embedding_size]
        word_embedding = self.embedding_seq(self.caption)

        # RNN
        lstm = BasicLSTMCell(self.config.embedding_size, name="LSTMcell", reuse=True)
        if (self.mode == "train"):
            lstm = DropoutWrapper(lstm, self.config.lstm_dropout_keep_prob, self.config.lstm_dropout_keep_prob,
                                  dtype=tf.float32)

        with tf.variable_scope("LSTM", initializer=self.initializer) as lstm_scope:
            zero_state = lstm.zero_state(image_embedding.get_shape()[0], tf.float32)
            # init_state a tuple of (c,h)
            _, init_state = lstm(image_embedding, zero_state)

            if (self.mode == "inference"):
                state_feed = tf.placeholder(tf.float32, [None, sum(lstm.state_size)], "state_feed")
                state = tf.split(state_feed, 2, -1)
                # when inference the input is a word ,which means the sentence length is 1
                output, last_state = lstm(tf.squeeze(word_embedding, squeeze_dims=1), state, lstm_scope)
            else:
                seq_len = tf.reduce_sum(self.caption_len, [-1])
                output, _ = rnn.dynamic_rnn(lstm, self.caption, seq_len, init_state, tf.float32, scope=lstm_scope)

        output = tf.reshape(output, [-1, lstm.output_size])
        with tf.variable_scope("logits") as logits:
            logit = tf.layers.dense(output, self.config.word_count, kernel_initializer=self.initializer, name="map")

        if (self.mode == "inference"):
            tf.nn.softmax(logit, name="softmax")
        else:
            target = tf.reshape(self.target, [-1])
            weights = tf.to_float(tf.reshape(self.caption_len, [-1]))

            losses = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=logit)
            batch_losses = tf.div(tf.reduce_sum(tf.multiply(losses, weights)), tf.reduce_sum(weights),
                                  name="batch_losses")
            tf.losses.add_loss(batch_losses)
            total_losses = tf.losses.get_total_loss()

            # Add summaries.
            tf.summary.scalar("losses/batch_loss", batch_losses)
            tf.summary.scalar("losses/total_loss", total_losses)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_losses
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.
