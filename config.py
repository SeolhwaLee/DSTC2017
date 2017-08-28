import os
from general_utils import get_logger


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.logger = get_logger(self.log_path)

    output_path = 'results/word2vec_lstm_v2/'
    model_output = output_path + 'model.weights_v2/'
    log_path = output_path + "log_v2.txt"

    lr = 0.001
    lr_decay = 0.9
    clip = -1
    nepoch_no_imprv = 3
    reload = False

    num_epochs = 20
    batch_size = 10

    # file name lists for training
    word2vec_filename = '../bilingual/wiki/wiki_en_model'

    train_filename = './data/train_dataset'
    dev_filename = './data/dev_dataset'
    test_filename = './data/test_dataset'
