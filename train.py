from config import Config
from dnn_model import Dnn
from data_process import Data
from utterance_embed import UtteranceEmbed


def main(config):
    data = Data(config.train_filename, config.dev_filename, config.test_filename)
    train_data = data.train_set
    dev_data = data.dev_set
    test_data = data.test_set

    # load word2vec
    utter_embed = UtteranceEmbed(config.word2vec_filename)

    input_size = (utter_embed.get_vector_size() * 2) # concat size
    num_neuron = 7500

    model = Dnn(input_size, num_neuron, 3, utter_embed, config)
    model.build()
    model.train(train_data, dev_data, test_data)


if __name__ == "__main__":
    config = Config()

    main(config)
