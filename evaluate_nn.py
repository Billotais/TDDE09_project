from tagger_nn import Tagger
from parser_nn import Parser
import logging
import json
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_utils import *
import itertools, pickle, argparse
from neuralNetwork/NeuralNetwork import NeuralNetwork
from lstm/biLSTM import BiLSTM
from nlp_tools import load_data, accuracy, get_sentences, get_tags, get_trees

logging.getLogger().setLevel(logging.INFO)


logging.getLogger().setLevel(logging.INFO)

if len(sys.argv) != 2:
    exit("Usage:\n python train.py [path to config file] ")

config = json.loads(open(sys.argv[1]).read())
config_nn = json.loads(open("./neural_network/train_nn_config.json").read())
if config['tagger'] == 'LSTM':
    config_lstm = json.loads(open("./lstm/train_config_lstm.json").read())
    x_embeddings, x, _, tags, word_dict, _, tag_dict_inv, sent_lens = load_processed_data(config_stm['processed_test_data_location'])
    logging.info("Data loaded")
    tagger = BiLSTM(config_lstm, tag_dict_inv, x.shape[1], x_embeddings, len(tag_dict_inv), word_dict)
    tagger.restore_sess()
    logging.info("Session restored!")
else:
    tagger = Tagger(config)
    tagger.train(load_data=False)
logging.info('Training Done')

parser = Parser(config_nn, tagger)
parser.train(load_data=True)
logging.info('Training Done')

dev_data = load_data('../../UD_English-EWT/en-ud-test.conllu')
gold_tags = list(itertools.chain(*get_tags(dev_data)))
gold_trees = list(itertools.chain(*get_trees(dev_data)))
pred_tags_lst = []
pred_trees_lst = []
pred_trees = []
correction = 0


for i, sentence in enumerate(get_sentences(dev_data)):
    print("\rEvaluated with sentence #{}".format(i), end="")
    # Hack to remove root
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
    