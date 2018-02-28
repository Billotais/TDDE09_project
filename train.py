from tagger import Tagger
from parser import Parser
import logging
import json
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_utils import *
from biLSTM import BiLSTM


logging.getLogger().setLevel(logging.INFO)

if len(sys.argv) != 2:
    exit("Usage:\n python train.py [path to config file] ")


config = json.loads(open(sys.argv[1]).read())
x_embeddings, x, y_onehot, tags, voc, voc_inv, tag_dict_inv, sent_lens = load_processed_data(config['processed_data_location'])
X_train, X_test, y_train, y_test = train_test_split(x, tags, test_size=0.1)
logging.info('Data loaded!')


biLSTM =BiLSTM(config, tag_dict_inv, x.shape[1], x_embeddings, len(tag_dict_inv))
biLSTM.train(X_train, y_train, X_test, y_test, sent_lens)

#tagger = Tagger()
#tagger.train(train_data, n_epochs=3, trunc_data=None)
#parser = Parser(tagger)
#parser.train(train_data, n_epochs=3, do_projectivize=True, trunc_data=None)
