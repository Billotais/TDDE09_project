import logging
import json
import sys
from data_utils import *
from biLSTM import BiLSTM


logging.getLogger().setLevel(logging.INFO)

if len(sys.argv) != 2:
    exit("Usage:\n python train.py [path to config file] ")


config = json.loads(open(sys.argv[1]).read())
x_embeddings, x, y, tags, voc, voc_inv, tag_dict_inv, sent_lens = load_processed_data(config['processed_data_location'])
x_dev, tags_dev, sent_lens_dev = load_processed_data(config['processed_dev_data_location'], True)
logging.info('Data loaded!')


biLSTM =BiLSTM(config, tag_dict_inv, x.shape[1], x_embeddings, len(tag_dict_inv))
biLSTM.train(x, tags, x_dev, tags_dev, sent_lens, sent_lens_dev)

