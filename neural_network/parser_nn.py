from NeuralNetwork import NeuralNetwork
from nlp_tools import softmax, get_sentences, get_tags, get_trees
from math import log
import tensorflow as tf
from data_utils import *
from NeuralNetwork import NeuralNetwork

class Parser():
    """A transition-based dependency parser.

    This parser implements the arc-standard algorithm for dependency
    parsing. When being presented with an input sentence, it first
    tags the sentence for parts of speech, and then uses a multi-class
    perceptron classifier to predict a sequence of *moves*
    (transitions) that construct a dependency tree for the input
    sentence. Moves are encoded as integers as follows:

    SHIFT = 0, LEFT-ARC = 1, RIGHT-ARC = 2

    At any given point in the predicted sequence, the state of the
    parser can be specified by: the index of the first word in the
    input sentence that the parser has not yet started to process; a
    stack holding the indices of those words that are currently being
    processed; and a partial dependency tree, represented as a list of
    indices such that `tree[i]` gives the index of the head (parent
    node) of the word at position `i`, or 0 in case the corresponding
    word has not yet been assigned a head.

    Attributes:
        tagger: A part-of-speech tagger.
        classifier: A multi-layer neural net classifier used to
            predict the next move of the parser.
    """

    def __init__(self, config, tagger):
        """Initialises a new parser."""
        self.config = config['parser']

        logging.getLogger().setLevel(logging.INFO)

        # Load embeddings
        self.x_embeddings, self.word_ids, self.tags_embeddings, self.tags, self.word_dict, self.word_dict_inv, self.tag_dict, self.tag_dict_inv, self.sent_lens, self.gold_tree = load_processed_data(config['processed_data_location'])
        
        self.word_ids_dev, self.tags_dev, self.sent_lens_dev, self.gold_tree_dev = load_processed_data(config['processed_dev_data_location'], True)
        logging.info('Data loaded!')

        
        self.tagger = tagger
        self.classifier = []


    def parse(self, words):
        """Parses a sentence.
        
        Args:
            words: The input sentence, a list of words.
        
        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """

        tags = self.tagger.tag(words)

        word_pad_id = self.word_dict['<PAD/>']
        tag_pad_id = self.tag_dict['<PAD/>']

        # Conver words to ids
        words = [ self.word_dict[w] if w in self.word_dict  else word_pad_id for w in words]
        # Convert tags to ids
        tags_ids = [self.tag_dict[t] for t in tags]

        # Parse
        i = 0
        stack = []
        pred_tree = [0] * len(words)
        while True:
            valid_moves = self.valid_moves(i, stack, pred_tree, words)
            
            if not valid_moves:
                break   

            feat = self.features(words, tags_ids, range(i, len(words)), stack, pred_tree,tag_pad_id , word_pad_id )

            feat = np.array(feat)
            feat = np.reshape(feat, (1,-1))
            

            valid_pred = np.zeros(3,dtype=bool) 
            valid_pred[valid_moves] = True

            move_to_do = self.classifier.predict(feat, valid_pred=valid_pred)
            
            (i, stack, pred_tree) = self.move(i, stack, pred_tree, move_to_do)
            
        return (tags, pred_tree)


    def valid_moves(self, i, stack, pred_tree, words):
        """Returns the valid moves for the specified parser
        configuration.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            The list of valid moves for the specified parser
                configuration.
        """        
        valid = []
        if i < len(words):
            valid.append(0)
        if len(stack) >= 3:
            valid.append(1)
            valid.append(2)
        
        return valid

    def move(self, i, stack, pred_tree, move):
        """Executes a single move.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            move: The move that the parser should make.
        
        Returns:
            The new parser configuration, represented as a triple
            containing the index of the new first unprocessed word,
            stack, and partial dependency tree.
        """
        
        
       
        if move == 0:
            stack.append(i)
            i += 1
        elif move == 1:
            pred_tree[stack[-2]] = stack[-1]
            stack.remove(stack[-2])
        elif move == 2:
            pred_tree[stack[-1]] = stack[-2] 
            stack.remove(stack[-1])
            
        return (i, stack, pred_tree)

    def features(self, words, tags, buffer, stack, parse, tag_pad , word_pad):
        """Extracts features for the specified parser configuration.

        Args:
            words: The input sentence, a list of words.
            gold_tags: The list of gold-standard tags for the input
                sentence.
            buffer: 
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            parse: The partial dependency tree.

        Returns:
            A feature vector for the specified configuration.
        """

        # Single word features
        b1_w = words[buffer[0]] if buffer else word_pad
        b1_t = tags[buffer[0]] if buffer else tag_pad

        b2_w = words[buffer[1]] if len(buffer) > 1 else word_pad
        b2_t = tags[buffer[1]] if len(buffer) > 1 else tag_pad

        b3_w = words[buffer[2]] if len(buffer) > 2 else word_pad
        b3_t = tags[buffer[2]] if len(buffer) > 2 else tag_pad

        s1_w = words[stack[-1]] if stack else word_pad
        s1_t = tags[stack[-1]] if stack else tag_pad

        s2_w = words[stack[-2]] if len(stack) > 1 else word_pad
        s2_t = tags[stack[-2]] if len(stack) > 1 else tag_pad

        s3_w = words[stack[-3]] if len(stack) > 2 else word_pad
        s3_t = tags[stack[-3]] if len(stack) > 2 else tag_pad


        def is_parent(parent, child):
            if child == 0:
                return False
            if parent == child:
                return True
            return is_parent(parent, parse[child])

        # Child that is the most on the left
        def lc1(parent):
            for i in range(0, len(words)):
                if is_parent(parent, i):
                    return i
            return -1
            
        # Child that is the most on the right
        def rc1(parent):
            for i in range(0, len(words), -1):
                if is_parent(parent, i):
                    return i
            return -1

        lc1_s1 = lc1(stack[-1]) if stack else -1
        rc1_s1 = rc1(stack[-1]) if stack else -1
        lc1_s2 = lc1(stack[-2]) if len(stack) > 1 else -1
        rc1_s2 = rc1(stack[-2]) if len(stack) > 1 else -1

        

        lc1_s1_t = tags[lc1_s1] if lc1_s1 >= 0 else tag_pad
        rc1_s1_t = tags[rc1_s1] if rc1_s1 >= 0 else tag_pad
        lc1_s2_t = tags[rc1_s2] if lc1_s2 >= 0 else tag_pad
        rc1_s2_t = tags[rc1_s2] if rc1_s2 >= 0 else tag_pad

        

        lc1_s1_w = words[lc1_s1] if lc1_s1 >= 0 else word_pad
        rc1_s1_w = words[rc1_s1] if rc1_s1 >= 0 else word_pad
        lc1_s2_w = words[rc1_s2] if lc1_s2 >= 0 else word_pad
        rc1_s2_w = words[rc1_s2] if rc1_s2 >= 0 else word_pad
        

        lc1_lc1_s1 = lc1(lc1_s1) if lc1_s1 >=0 else -1
        lc1_lc1_s2 = lc1(lc1_s2) if lc1_s2 >=0 else -1
        rc1_rc1_s1 = rc1(rc1_s1) if rc1_s1 >=0 else -1
        rc1_rc1_s2 = rc1(rc1_s2) if rc1_s2 >=0 else -1

        lc1_lc1_s1_t = tags[lc1_lc1_s1] if lc1_lc1_s1 >= 0 else tag_pad
        lc1_lc1_s2_t = tags[lc1_lc1_s2] if lc1_lc1_s2 >= 0 else tag_pad
        rc1_rc1_s1_t = tags[rc1_rc1_s1] if rc1_rc1_s1 >= 0 else tag_pad
        rc1_rc1_s2_t = tags[rc1_rc1_s2] if rc1_rc1_s2 >= 0 else tag_pad
        
        lc1_lc1_s1_w = words[lc1_lc1_s1] if lc1_lc1_s1 >= 0 else word_pad
        lc1_lc1_s2_w = words[lc1_lc1_s2] if lc1_lc1_s2 >= 0 else word_pad
        rc1_rc1_s1_w = words[rc1_rc1_s1] if rc1_rc1_s1 >= 0 else word_pad
        rc1_rc1_s2_w = words[rc1_rc1_s2] if rc1_rc1_s2 >= 0 else word_pad

        feat = [b1_w, b2_w, b3_w, s1_w, s2_w, s3_w, lc1_s1_w, rc1_s1_w, lc1_s2_w, rc1_s2_w, lc1_lc1_s1_w, lc1_lc1_s2_w, rc1_rc1_s1_w, rc1_rc1_s2_w, b1_t, b2_t, b3_t, s1_t, s2_t, s3_t, lc1_s1_t, rc1_s1_t, lc1_s2_t, rc1_s2_t ,lc1_lc1_s1_t, lc1_lc1_s2_t, rc1_rc1_s1_t, rc1_rc1_s2_t]
        """
        feat = [b1_w, b2_w, b3_w, s1_w, s2_w, s3_w, lc1_s1_w, rc1_s1_w, lc1_s2_w, rc1_s2_w,b1_t, b2_t, b3_t, s1_t, s2_t, s3_t, lc1_s1_t, rc1_s1_t, lc1_s2_t, rc1_s2_t]
        """
        return feat

    def train(self, load_data=False):
        """Trains the parser on training data.

        Args:
            data: Training data, a list of sentences with gold trees.
            n_epochs:
            trunc_data:
        """
        
        # Find pad value
        word_pad_id = self.word_dict['<PAD/>']
        tag_pad_id = self.tag_dict['<PAD/>']

        """
        # Find tags for training data
        for sent_id in range(self.tags.shape[0]):
            words = self.word_ids[sent_id, :int(self.sent_lens[sent_id])]
            words = words.tolist()

            words = [ str(int(w)) for w in words]
            # conver to word from ids
            #words = [ self.word_dict_inv[w] if w in self.word_dict_inv  else self.word_dict_inv[word_pad_id] for w in words]
            words = [ self.word_dict_inv[w] for w in words]

            #words_pad = [word_pad_id]*2 + words + [word_pad_id]*2

            #tags = [tag_pad_id]*2

            tag_pred = self.tagger.tag(words)
            tag_pred = [ self.tag_dict[w] for w in tag_pred]
            self.tags[sent_id,:int(self.sent_lens[sent_id])] = tag_pred

        """
        # Find tags using the tagger    
        x_parser, y_parser = self.generate_data_for_parser(self.word_ids, self.tags, self.gold_tree, self.sent_lens, tag_pad_id, word_pad_id)

        self.classifier = NeuralNetwork(self.config, self.x_embeddings, self.tags_embeddings, 3, feature_type = 'parser_1') 

        if load_data:
            self.classifier.restore_sess()   
        else:
            num_data = x_parser.shape[0]
            num_train = int(num_data*0.99)

            x_train_parser = x_parser[:num_train,:]
            y_train_parser = y_parser[:num_train]

            x_dev_parser = x_parser[num_train:,:]
            y_dev_parser = y_parser[num_train:]

            self.classifier.train(x_train_parser, y_train_parser, x_dev_parser, y_dev_parser)



    def gold_move(self, i, stack, pred_tree, gold_tree, sent_len):        
        if len(stack) < 3 and i < sent_len:
            return 0
        elif len(stack) <= 1 and i >= sent_len:
            return None
        
        elif stack[-1] == gold_tree[stack[-2]]:
            left_arc = True
            for t in range(len(gold_tree)):
                
                if gold_tree[t] == stack[-2]:
                    if pred_tree[t] != stack[-2]:
                        left_arc = False
            if left_arc:
                return 1
        elif stack[-2] == gold_tree[stack[-1]]:
            right_arc = True
            for t in range(len(gold_tree)):
                
                if gold_tree[t] == stack[-1]:
                    if pred_tree[t] != stack[-1]:
                        right_arc = False
            if right_arc:
                return 2
        return 0


    def generate_data_for_parser(self, x, tags, trees, sent_lens, tag_pad, word_pad):
        x_parser = []
        y_parser = []

        # Features: w_i-2, w_i-1, w_i, t_i, tag_i-1, tag_i-2
        for i in range(x.shape[0]):
            sent = x[i,:].tolist()
            tag = tags[i,:].tolist()
            tree = trees[i,:].tolist()

            # Pad sentence to include root
            sent = [word_pad]*1 + sent[0:int(sent_lens[i])]
            tag = [tag_pad]*1 + tag[0:int(sent_lens[i])]
            
            tree = [0]*1 + tree[0:int(sent_lens[i])]
            
            tree = [int(i) for i in tree]

            pred_tree = [0] * len(tree)
            stack = []

            buffer_pos = 0
            while True:
                move_to_do = self.gold_move(buffer_pos, stack, pred_tree, tree, len(sent))
                
                if move_to_do == None:
                    break

                feat = self.features(sent, tag, range(buffer_pos, len(sent)), stack, pred_tree,tag_pad , word_pad)

                x_parser.append(feat)
                y_parser.append(move_to_do)
                (buffer_pos, stack, pred_tree) = self.move(buffer_pos, stack, pred_tree, move_to_do)
                

        x_parser = np.array(x_parser, dtype=np.float32)
        y_parser = np.array(y_parser, dtype=np.float32)

        return x_parser, y_parser