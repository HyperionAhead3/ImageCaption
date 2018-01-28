class ModelConfig:
    def __init__(self):
        self.model_path = "model/SCA_model"
        self.ft_ckp_path = 'inception_v3.ckpt'
        self.embedding_size = 512
        self.dropout_keep_prob = 0.7
        self.initializer_scale = 0.08
        self.word_count = 6453
        self.lstm_dropout_keep_prob = 0.7
        self.UNKNOWN = 6451
        self.EOF = 6452


class TrainConfg:
    def __init__(self):
        self.image_path = "/disk4/flick8k"
