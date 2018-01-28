import jieba
import tensorflow as tf

import configuration
import dataprocess
import Dictionary
import SCA_CNN


def start_train():
    config = configuration.TrainConfg()

    dataset = tf.data.TFRecordDataset(config.train_tfrecord)
    dataset = dataset.map(dataprocess.read_tfrecord_data)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(config.batch_size, padded_shapes={"caption": [None], "image": [None]})

    iterator = dataset.make_one_shot_interator()
    image, caption = iterator.get_next()
    model=SCA_CNN.SCA_CNN("train", image, caption, config.training_inception, is_first_time=True)

    learning_rate_decay_fn = None
    if config.training_inception:
        learning_rate = tf.constant(config.learning_rate_with_inception)
    else:
        learning_rate = tf.constant(config.learning_rate_without_inception)
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                     model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=training_config.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)


# Run training.
tf.contrib.slim.learning.train(
    train_op,
    train_dir,
    log_every_n_steps=FLAGS.log_every_n_steps,
    graph=g,
    global_step=model.global_step,
    number_of_steps=FLAGS.number_of_steps,
    init_fn=model.init_fn,
    saver=saver)

if __name__ == '__main__':
    start_train()
