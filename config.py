import os
from general_utils import get_logger


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        elif not os.path.exists(self.model_output):
            os.makedirs(self.model_output)

        self.logger = get_logger(self.log_path)

    output_path = 'results/mlp/'
    model_output = output_path + 'model.weights/'
    log_path = output_path + "log.txt"

    reload = False

    lr = 0.005
    num_epochs = 50
    batch_size = 10
    embed_method = 'word2vec'

    # file name lists for training
    word2vec_filename = '../bilingual/wiki/wiki_en_model'

    train_filename = './data/train_dataset'
    dev_filename = './data/dev_dataset'
    test_filename = './data/test_dataset'
