import os
from general_utils import get_logger


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.logger = get_logger(self.log_path)

    output_path = 'results/mlp/'
    model_output = output_path + 'model.weights/'
    log_path = output_path + "log.txt"

    reload = False

    lr = 0.005
    lr_decay = 0.9
    beta = 0.01
    num_layer = 5
    num_epochs = 20
    batch_size = 10
    nepoch_no_imprv = 3


    embed_method = 'word2vec'

    # file name lists for training
    word2vec_filename = '../bilingual/wiki/wiki_en_model'

    train_filename = './data/train_dataset'
    dev_filename = './data/dev_dataset'
    test_filename = './data/test_dataset'
