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

if len(sys.argv) != 5:
    exit("Usage:\n python prepare_data.py [path to training data] [path to evaluation data] [path to training config file]")

#data
train_data = load_data(sys.argv[1])
dev_data = load_data(sys.argv[2])
test_data = load_data(sys.argv[3])
config = json.loads(open(sys.argv[4]).read())

#Preprocess
#Training
x_embeddings, x, x_dev, y, tags, tags_dev, voc, voc_inv, tag_dict, tag_dict_inv, sent_lens, sent_lens_dev, tree, tree_dev = process_data(config, train_data, dev_data)

#Test
x_embeddings_test, x_test, y_test, tags_test, voc_test, voc_inv_test, tag_dict_test, tag_dict_inv_test, sent_lens_test, tree_test = process_data(config, test_data)


# Save data files
save_data_to_dir(make_dir(config['processed_data_location']), x_embeddings, x, y, tags, voc, voc_inv, tag_dict, tag_dict_inv, sent_lens, tree)
save_data_to_dir(make_dir(config['processed_dev_data_location']), x=x_dev, tags=tags_dev, sent_lens=sent_lens_dev, tree=tree_dev)
save_data_to_dir(make_dir(config['processed_test_data_location']), x_embeddings_test, x_test, y_test, tags_test, voc_test, voc_inv_test, tag_dict_test, tag_dict_inv_test, sent_lens_test, tree_dev)



#with open(trained_dir + 'x.pickle', 'wb') as outfile:
#    pickle.dump(x, outfile, pickle.HIGHEST_PROTOCOL)
#    logging.info('Data matrix X saved')
#
#with open(trained_dir + 'y_onehot.pickle', 'wb') as outfile:
#    pickle.dump(y_onehot, outfile, pickle.HIGHEST_PROTOCOL)
#    logging.info('One hot matrix y saved')
#
#with open(trained_dir + 'tags.pickle', 'wb') as outfile:
#    pickle.dump(tags, outfile, pickle.HIGHEST_PROTOCOL)
#    logging.info('Data matrix y saved')
#
#with open(trained_dir + 'voc.json', 'w',encoding='utf-8') as outfile:
#    json.dump(voc, outfile, indent=4, ensure_ascii=False)
#    logging.info('Voc dict saved')
#
#with open(trained_dir + 'voc_inv.json', 'w',encoding='utf-8') as outfile:
#    json.dump(voc_inv, outfile, indent=4, ensure_ascii=False)
#    logging.info('Voc_inv dict saved')
#
#with open(trained_dir + 'tag_dict_inv.json', 'w',encoding='utf-8') as outfile:
#    json.dump(tag_dict_inv, outfile, indent=4, ensure_ascii=False)
#    logging.info('Tag dict saved')
