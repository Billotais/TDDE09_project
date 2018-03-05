from neural_network.tagger import TaggerNN
from neural_network.parser import ParserNN
from perceptron.tagger import Tagger
import logging
import json
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from lstm.data_utils import *
import itertools, pickle, argparse
from neural_network.NeuralNetwork import NeuralNetwork
from lstm.biLSTM import BiLSTM
from neural_network.nlp_tools import load_data, accuracy, get_sentences, get_tags, get_trees


if len(sys.argv) != 2:
    exit("Usage:\n python train.py [path to config file] ")


logging.getLogger().setLevel(logging.INFO)


config = json.loads(open(sys.argv[1]).read())
config_nn = json.loads(open(config['NN_config']).read())

# Load tagger
if config['tagger'] == 'LSTM':
    config_lstm = json.loads(open(config['LSTM_config']).read())
    x_embeddings, x, _, tags, word_dict, _, tag_dict_inv, sent_lens = load_processed_data(config_lstm['processed_data_location'])
    x_dev, tags_dev, sent_lens_dev = load_processed_data(config_lstm['processed_dev_data_location'], True)

    logging.info("Data loaded")
    tagger = BiLSTM(config_lstm, tag_dict_inv, x.shape[1], x_embeddings, len(tag_dict_inv),word_dict)

    if config['train_tagger']:
        logging.info("Training LSTM")
        tagger.train(x, tags, x_dev, tags_dev, sent_lens, sent_lens_dev)
        logging.info("Finished training")
    else:    
        tagger.restore_sess()
        logging.info("Session restored!")
elif config['tagger'] == 'NN':
    tagger = TaggerNN(config_nn)
    tagger.train(train=config['train_tagger'])
elif config['tagger'] == 'Perceptron': 
    tagger = Tagger()
    train_data = load_data(config['train_data'])
    tagger.train(train_data, n_epochs=3, trunc_data=0)
else:
    exit("Usage:\n Undefined tagger. Should be either LSTM, NN or Perceptron ")

logging.info('Loaded tagger')

parser = ParserNN(config_nn, tagger)
parser.train(train=config['train_parser'])
logging.info('Loaded parser')

dev_data = load_data(config['test_data'])
gold_tags = list(itertools.chain(*get_tags(dev_data)))
gold_trees = list(itertools.chain(*get_trees(dev_data)))
pred_tags_lst = []
pred_trees_lst = []
pred_trees = []
correction = 0


for i, sentence in enumerate(get_sentences(dev_data)):
    print("\rEvaluated with sentence #{}".format(i), end="")
    # Hack to remove root since in the preprocessing, we used <PAD/> to denote <ROOT>
    sentence[0] = '<PAD/>'
    (pred_tags, pred_tree) = parser.parse(sentence)
    pred_tags[0] = '<ROOT>'
    pred_tags_lst += pred_tags
    pred_trees_lst += pred_tree
    pred_trees.append(pred_tree)
    correction += 1


print("Tagging accuracy: {:.2%}".format(accuracy(pred_tags_lst, gold_tags, correction)))
print("Unlabelled attachment score: {:.2%}".format(accuracy(pred_trees_lst, gold_trees, correction)))
print("Exact matches: {:.2%}".format(accuracy(pred_trees, get_trees(dev_data), 0)))
    