from NeuralNetwork import NeuralNetwork
from math import log
import tensorflow as tf
from data_utils import *
from NeuralNetwork import NeuralNetwork

class Tagger():
    def __init__(self, config):
        """Initialises a new parser."""
        self.config = config['tagger']

        logging.getLogger().setLevel(logging.INFO)

        # Load embeddings
        self.x_embeddings, self.word_ids, self.tags_embeddings, self.tags, self.word_dict, self.word_dict_inv, self.tag_dict, self.tag_dict_inv, self.sent_lens, _ = load_processed_data(config['processed_data_location'])
        
        self.word_ids_dev, self.tags_dev, self.sent_lens_dev, _ = load_processed_data(config['processed_dev_data_location'], True)
        logging.info('Data loaded!')

        self.classifier = []

    def features(self, words, i, pred_tags):
        # Convert words to word ids
         
        tag_pad = '<PAD/>'
        word_pad = '<PAD/>'

        b_w = words[i] if (i > 0 and i < len(words)) else word_pad
        bm1_w = words[i-1] if (i-1 > 0 and i-1 < len(words)) else word_pad
        bm2_w = words[i-2] if (i-2 > 0 and i-2 < len(words)) else word_pad
        bp1_w = words[i+1] if (i+1 > 0 and i+1 < len(words)) else word_pad
        bp2_w = words[i+2] if (i+2 > 0 and i+2 < len(words)) else word_pad

        bm1_t = pred_tags[i-1] if (i-1 > 0 and i-1 < len(pred_tags)) else tag_pad
        bm2_t = pred_tags[i-2] if (i-2 > 0 and i-2 < len(pred_tags)) else tag_pad


        feat_w = [bm2_w, bm1_w, b_w, bp1_w, bp2_w]
        feat_t = [bm2_t, bm1_t]


        feat_w_ids = [ self.word_dict[w] if w in self.word_dict  else self.word_dict[word_pad] for w in feat_w]
        feat_w_ids = np.array(feat_w_ids)

        feat_t_ids = [self.tag_dict[t] for t in feat_t]
        feat_t_ids = np.array(feat_t_ids) 


        feat = np.concatenate((feat_w_ids, feat_t_ids), axis=0)
        feat = np.reshape(feat, (1,-1))

        return feat


    def tag(self, words):
        
        pred_tags = []
        for i in range(len(words)):
            feat = self.features(words, i, pred_tags)
            tag = self.classifier.predict(feat)
            tag = self.tag_dict_inv[str(tag)]
            pred_tags.append(tag)

        return pred_tags

    def train(self, load_data=False):
        """Trains the parser on training data.

        Args:
            data: Training data, a list of sentences with gold trees.
            n_epochs:
            trunc_data:
        """
        
        # Find pad value
        tag_pad = self.tag_dict['<PAD/>']
        word_pad = self.word_dict['<PAD/>']

        self.classifier = NeuralNetwork(self.config, self.x_embeddings, self.tags_embeddings, len(self.tag_dict_inv), feature_type = 'tagger_1')
        
        if load_data:
            self.classifier.restore_sess()
        else:
            logging.info('Training NN for tagger!')
        
            # Generate all configurations for training
            x_tagger, y_tagger = generate_data_for_tagger(self.word_ids, self.tags, self.sent_lens, tag_pad, word_pad)
            x_dev_tagger, y_dev_tagger = generate_data_for_tagger(self.word_ids_dev, self.tags_dev, self.sent_lens_dev, tag_pad, word_pad)

            self.classifier.train(x_tagger, y_tagger, x_dev_tagger, y_dev_tagger)

            logging.info('Training NN for parser!')


def generate_data_for_tagger(x, tags, sent_lens, tag_pad, word_pad):
    x_tagger = []
    y_tagger = []

    # Features: w_i-2, w_i-1, w_i, w_i+1, w_i+1, tag_i-1, tag_i-2
    for i in range(x.shape[0]):
        sent = x[i,:].tolist()
        tag = tags[i,:].tolist()

        # Pad sentence
        sent = [word_pad,word_pad] + sent[0:int(sent_lens[i])] + [word_pad, word_pad]
        tag = [tag_pad, tag_pad] + tag

        for j in range(2, len(sent) - 2):
            x_tagger.append([sent[j-2], sent[j-1], sent[j], sent[j+1], sent[j+2], tag[j-2], tag[j-1]])
            y_tagger.append(tag[j])

    x_tagger = np.array(x_tagger, dtype=np.float32)
    y_tagger = np.array(y_tagger, dtype=np.float32)

    return x_tagger, y_tagger        
            
