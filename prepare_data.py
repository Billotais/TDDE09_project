import os
import sys
import json
import time
import shutil
import pickle
import logging
from nlp_tools import *
from data_utils import *
logging.getLogger().setLevel(logging.INFO)

if len(sys.argv) != 4:
    exit("Usage:\n python prepare_data.py [path to training data] [path to evaluation data] [path to training config file]")

#data
train_data = load_data(sys.argv[1])
dev_data = load_data(sys.argv[2])
params = json.loads(open(sys.argv[3]).read())

#Preprocess
x_embeddings, x, y_onehot, tags, voc, voc_inv, tag_dict_inv, sent_lens = process_data(train_data, params)


# Create a directory, everything related to the training will be saved in this directory
trained_dir = params['processed_data_location']
if os.path.exists(trained_dir):
    shutil.rmtree(trained_dir)
os.makedirs(trained_dir)

# Save trained parameters and files since predict.py needs them
with open(trained_dir + 'sent_lens.pickle', 'wb') as outfile:
    pickle.dump(sent_lens, outfile, pickle.HIGHEST_PROTOCOL)
    logging.info('Vector with sent_lens saved')

with open(trained_dir + 'x_embeddings.pickle', 'wb') as outfile:
    pickle.dump(x_embeddings, outfile, pickle.HIGHEST_PROTOCOL)
    logging.info('Embedding matrix X saved')

with open(trained_dir + 'x.pickle', 'wb') as outfile:
    pickle.dump(x, outfile, pickle.HIGHEST_PROTOCOL)
    logging.info('Data matrix X saved')

with open(trained_dir + 'y_onehot.pickle', 'wb') as outfile:
    pickle.dump(y_onehot, outfile, pickle.HIGHEST_PROTOCOL)
    logging.info('One hot matrix y saved')

with open(trained_dir + 'tags.pickle', 'wb') as outfile:
    pickle.dump(tags, outfile, pickle.HIGHEST_PROTOCOL)
    logging.info('Data matrix y saved')

with open(trained_dir + 'voc.json', 'w',encoding='utf-8') as outfile:
    json.dump(voc, outfile, indent=4, ensure_ascii=False)
    logging.info('Voc dict saved')

with open(trained_dir + 'voc_inv.json', 'w',encoding='utf-8') as outfile:
    json.dump(voc_inv, outfile, indent=4, ensure_ascii=False)
    logging.info('Voc_inv dict saved')

with open(trained_dir + 'tag_dict_inv.json', 'w',encoding='utf-8') as outfile:
    json.dump(tag_dict_inv, outfile, indent=4, ensure_ascii=False)
    logging.info('Tag dict saved')
