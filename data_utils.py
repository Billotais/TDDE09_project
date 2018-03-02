import numpy as np
from gensim.models import KeyedVectors
import pickle
import json
import os
import shutil
import logging
logging.getLogger().setLevel(logging.INFO)


def get_trees(data):
    for sentence in data:
        out = []
        out.append(0)
        for word in sentence:
            out.append(int(word[6]))
        yield out

def load_data(filename):
    data = [[]]
    with open(filename, encoding='utf-8') as source:
        for line in source:
            if len(line) == 1: 
                data.append([])
            elif line.split()[0].isdigit():
                data[-1].append(line.split('\t'))
    if not data[-1]:
        data = data[:-1]

    return data

def accuracy(preds, golds, correction=0):
    count = -correction
    score = -correction
    for pred, gold in zip(preds, golds):
        count += 1
        if pred == gold:
            score +=1
    return score/count

def get_sentences(data):
    sents = []
    max_len = 0
    for sent in data:
        out = []
        for word in sent:
            out.append(word[1])
        sents.append(out)
        max_len = max(max_len, len(out))
    return sents, max_len

def get_tags(data):
    tags = []
    for sent in data:
        out = []
        for word in sent:
            out.append(word[3])
        tags.append(out)
    return tags

def load_embeddings(voc_inv, config):
    embeddings = []
    model = KeyedVectors.load_word2vec_format(config['wv_location'], binary=True, limit=config['emb_limit'])
    for i, _ in enumerate(voc_inv):
        if voc_inv[i] == '<BOS/>':
            embeddings.append(np.zeros(300))
        elif voc_inv[i] == '<EOS/>':
            embeddings.append(np.zeros(300))
        elif voc_inv[i] == '<PAD/':
            embeddings.append(np.zeros(300))
        elif voc_inv[i] in model:
            embeddings.append(model[voc_inv[i]])
        else:
            embeddings.append(np.random.uniform(-0.25, 0.25, 300))

    return embeddings

def get_tag_matrix(tag_dict):
    dim = len(tag_dict)
    tag_matrix = np.zeros((dim,dim))
    np.fill_diagonal(tag_matrix, 1)
    return tag_matrix

def build_dict(data):
    word_dict = {}
    word_dict_reverse = {}
    ind = 0
    for sent in data:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ind
                word_dict_reverse[ind] = word
                ind+=1
    return word_dict, word_dict_reverse

def pad_sents(sents, seq_len, pad = '<PAD/>'):
    padded_sents = []
    sent_lens = []
    for i in range(len(sents)):
        sent = sents[i]
        sent_len = len(sent)
        num_pad = seq_len - sent_len
        # padded_seq = '<BOS/>' + sent + '<EOS/>' + [pad] * num_pad
        padded_seq = sent + [pad] * num_pad
        padded_sents.append(padded_seq)
        sent_lens.append(sent_len)
    return padded_sents, sent_lens

def process_data(config, data, dev_data=None):
    x, max_len_sents = get_sentences(data)
    if dev_data is not None:
        x_test, max_test = get_sentences(dev_data)
        max_len_sents = max(max_len_sents, max_test)

    x, sent_lens = pad_sents(x, max_len_sents)
    if dev_data is not None:
        x_test, test_lens = pad_sents(x_test, max_len_sents)

    if dev_data is not None:
        voc, voc_inv = build_dict(x+x_test)
    else:
        voc, voc_inv = build_dict(x)

    tags = get_tags(data)
    if dev_data is not None:
        tags_test = get_tags(dev_data)

    tags, _ = pad_sents(tags, max_len_sents)
    if dev_data is not None:
        tags_test, _ = pad_sents(tags_test, max_len_sents)

    if dev_data is not None:
        tag_dict, tag_dict_inv = build_dict(tags+tags_test)
    else:
        tag_dict, tag_dict_inv = build_dict(tags)

    x = np.array([[voc[word] for word in sentence] for sentence in x])
    if dev_data is not None:
        x_test = np.array([[voc[word] for word in sentence] for sentence in x_test])

    tags = np.array([[tag_dict[word] for word in sentence] for sentence in tags])
    if dev_data is not None:
        tags_test = np.array([[tag_dict[word] for word in sentence] for sentence in tags_test])

    x_emb = load_embeddings(voc_inv,config)
    y = get_tag_matrix(tag_dict_inv)

    if dev_data is not None:
        return x_emb, x, x_test, y, tags, tags_test, voc, voc_inv, tag_dict_inv, sent_lens, test_lens
    else:
        return x_emb, x, y, tags, voc, voc_inv, tag_dict_inv, sent_lens


def load_processed_data(trained_dir, dev=False):
    if dev is False:
        voc = json.loads(open(trained_dir + 'voc.json').read())
        voc_inv = json.loads(open(trained_dir + 'voc_inv.json').read())
        tag_dict_inv = json.loads(open(trained_dir + 'tag_dict_inv.json').read())

        with open(trained_dir + 'y.pickle', 'rb') as input_file:
            fetched_y = pickle.load(input_file)
        y = np.array(fetched_y, dtype = np.float32)

        with open(trained_dir + 'x_embeddings.pickle', 'rb') as input_file:
            fetched_x_embeddings = pickle.load(input_file)
        x_embeddings = np.array(fetched_x_embeddings, dtype = np.float32)

    with open(trained_dir + 'x.pickle', 'rb') as input_file:
        fetched_x = pickle.load(input_file)
    x = np.array(fetched_x, dtype = np.float32)

    with open(trained_dir + 'sent_lens.pickle', 'rb') as input_file:
        fetched_sent_lens = pickle.load(input_file)
    sent_lens = np.array(fetched_sent_lens, dtype=np.float32)

    with open(trained_dir + 'tags.pickle', 'rb') as input_file:
        fetched_tags = pickle.load(input_file)
    tags = np.array(fetched_tags, dtype = np.float32)

    if dev is False:
        return x_embeddings, x, y, tags, voc, voc_inv, tag_dict_inv, sent_lens
    else:
        return x, tags, sent_lens

def make_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    return dir

def save_data_to_dir(dir, x_embeddings=None, x=None, y=None, tags=None, voc=None, voc_inv=None, tag_dict_inv=None, sent_lens=None):
    if sent_lens is not None:
        with open(dir + 'sent_lens.pickle', 'wb') as outfile:
            pickle.dump(sent_lens, outfile, pickle.HIGHEST_PROTOCOL)
            logging.info('Vector with sent_lens saved')
    if x_embeddings is not None:
        with open(dir + 'x_embeddings.pickle', 'wb') as outfile:
            pickle.dump(x_embeddings, outfile, pickle.HIGHEST_PROTOCOL)
            logging.info('Embedding matrix X saved')
    if x is not None:
        with open(dir + 'x.pickle', 'wb') as outfile:
            pickle.dump(x, outfile, pickle.HIGHEST_PROTOCOL)
            logging.info('Data matrix X_test saved')
    if y is not None:
        with open(dir + 'y.pickle', 'wb') as outfile:
            pickle.dump(y, outfile, pickle.HIGHEST_PROTOCOL)
            logging.info('One hot matrix y saved')
    if tags is not None:
        with open(dir + 'tags.pickle', 'wb') as outfile:
            pickle.dump(tags, outfile, pickle.HIGHEST_PROTOCOL)
            logging.info('Data matrix y saved')
    if voc is not None:
        with open(dir + 'voc.json', 'w') as outfile:
            json.dump(voc, outfile, indent=4, ensure_ascii=False)
            logging.info('Voc dict saved')
    if voc_inv is not None:
        with open(dir + 'voc_inv.json', 'w') as outfile:
            json.dump(voc_inv, outfile, indent=4, ensure_ascii=False)
            logging.info('Voc_inv dict saved')
    if tag_dict_inv is not None:
        with open(dir + 'tag_dict_inv.json', 'w') as outfile:
            json.dump(tag_dict_inv, outfile, indent=4, ensure_ascii=False)
            logging.info('Tag dict saved')



