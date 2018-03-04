import logging
import json
import sys
from data_utils import *
from biLSTM import BiLSTM


logging.getLogger().setLevel(logging.INFO)

if len(sys.argv) != 2:
    exit("Usage:\n python train.py [path to config file] ")

config = json.loads(open(sys.argv[1]).read())
x_embeddings, x, _, tags, _, _, tag_dict_inv, sent_lens = load_processed_data(config['processed_test_data_location'])
logging.info("Data loaded")

biLSTM =BiLSTM(config, tag_dict_inv, x.shape[1], x_embeddings, len(tag_dict_inv))
biLSTM.restore_sess()
#accuracy = biLSTM.evaluate(x, tags, sent_lens)
#logging.info("Test accuracy: sh{:04.2f}%".format(accuracy))