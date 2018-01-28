class ModelConfig:
    def __init__(self):
        self.model_path = "model/SCA_model"
        self.ft_ckp_path = 'inception_v3.ckpt'
        self.embedding_size = 512
        self.dropout_keep_prob = 0.7
        self.initializer_scale = 0.08
        self.lstm_dropout_keep_prob = 0.7
        self.word_count = 6453


class TrainConfg:
    def __init__(self):
        self.UNKNOWN = 6451
        self.EOF = 6452
        self.image_path = "/disk4/flick8k"
        self.train_tfrecord = "/disk4/flick8k/train.tfrecords"
        self.batch_size = 64
        self.training_inception = False
        self.learning_rate_with_inception = 0.0005
        self.learning_rate_without_inception = 2
        self.decay_rate = 0.5
