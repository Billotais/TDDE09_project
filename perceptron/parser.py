from perceptron.perceptron import Perceptron
from perceptron.nlp_tools import softmax, get_sentences, get_tags, get_trees
from math import log
from copy import deepcopy

class Parser():
    """A transition-based dependency parser.

    This parser implements the arc-standard algorithm for dependency
    parsing. When being presented with an input sentence, it first
    tags the sentence for parts of speech, and then uses a multi-class
    perceptron classifier to predict a sequence of *moves*
    (transitions) that construct a dependency tree for the input
    sentence. Moves are encoded as integers as follows:

    SHIFT = 0, LEFT-ARC = 1, RIGHT-ARC = 2, SWAP = 3

    At any given point in the predicted sequence, the state of the
    parser can be specified by: a buffer containing the words in the
    input sentence that the parser has not yet started to process; a
    stack holding the indices of those words that are currently being
    processed; and a partial dependency tree, represented as a list of
    indices such that `tree[i]` gives the index of the head (parent
    node) of the word at position `i`, or 0 in case the corresponding
    word has not yet been assigned a head.

    Attributes:
        tagger: A part-of-speech tagger.
        classifier: A multi-class perceptron classifier used to
            predict the next move of the parser.
    """

    def __init__(self, tagger):
        """Initializes a new parser."""
        self.tagger = tagger
        self.classifier = Perceptron()

    def initial_config(self,words):
        """Initializes the config for the parser
        
        Args:
            words: the words of a sentence
            
        Returns:
            a initial parser config
        """
        config = {}
        config['score'] = 0
        config['pred_tree'] = [0] * len(words)
        config['stack'] = []
        config['buffer'] = list(range(len(words)))
        config['next_move'] = 0
        config['is_gold'] = True
        return config

    def predict(self, feat, candidates):
        """Calls the predict function of the classifier and applies the softmax
            function on the scores
        
        Args:
            feat: a feature vector
            candidates: the possible moves
        
        Returns:
            the possible moves with their respective scores
        """
        _, scores = self.classifier.predict(feat, candidates)
        if scores:
            # apply softmax on the scores 
            scores_lst = [(k, v) for k, v in scores.items()]
            softmax_scores = softmax(list(zip(*scores_lst))[1])
            scores = dict(list(zip(list(zip(*scores_lst))[0], softmax_scores)))
        return scores
    
    def update_and_reset_config(self, config, feat, gold_move):
        """This functions is called when the gold_tree falls of the beam. It
        updates the classifier and resets the parser config such that only the
        gold configuration is in the beam.
        
        Args:
            config: the parser gold config
            feat: a featire vector
            gold_move: the correct move
            
        Returns:
            the new config
        """
        config['next_move'] = gold_move
        self.classifier.update(feat,gold_move)
        return [config]

    def parse(self, words, gold_tree=None, beam_size=10):
        """Parses a sentence and also updates the classifier 
        if a gold tree was passed to the function

        Args:
            words: The input sentence, a list of words.
            gold_tree: if a gold_tree is passed, the classifier is trained
            beam_size: the width of the beam, when using beam search

        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """
        if gold_tree:
            word_order = self.get_word_order(gold_tree)
        tags = self.tagger.tag(words)
        possible_configs = [self.initial_config(words)]
        while any(config['next_move'] != None for config in possible_configs):
            old_possible_configs = possible_configs
            possible_configs = []
            for config in old_possible_configs:
                config = self.move(config)
                candidates = self.valid_moves(config)
                if candidates:
                    feat = self.features(words, tags, config)
                    scores = self.predict(feat, candidates)
                    if gold_tree:
                        gold_move = self.gold_move(config, gold_tree, \
                                                    word_order)
                        if config['is_gold'] and gold_move not in scores:
                            possible_configs = self.update_and_reset_config( \
                                                        config, feat, gold_move)
                            break
                    # add new configs for the possible moves
                    for curr_move, curr_score in scores.items():
                        # create a copy of the config and append it to the list
                        new_config = deepcopy(config)
                        if curr_score > 0:
                            new_config['score'] += log(curr_score)
                        else:
                            new_config['score'] += float("-inf")
                        new_config['next_move'] = curr_move
                        if gold_tree and gold_move != curr_move:
                            new_config['is_gold'] = False
                        possible_configs.append(new_config)
                else:
                    config['next_move'] = None
                    possible_configs.append(config)
            # delete the configs with the lowest scores
            while len(possible_configs) > beam_size:
                worst_conf_ind, worst_conf = \
                    min(enumerate(possible_configs), 
                        key = lambda t: t[1]['score'])
                if gold_tree and worst_conf['is_gold'] == True:
                    feat = self.features(words, tags, worst_conf)
                    possible_configs = self.update_and_reset_config( \
                                    worst_conf, feat, worst_conf['next_move'])
                else:
                    del possible_configs[worst_conf_ind]
        # return best tree
        best_config = max(possible_configs, key = lambda t: t['score'])
        return tags, best_config['pred_tree']

    def valid_moves(self, config):
        """Returns the valid moves for the specified parser
        configuration.

        Args:
            config: the current parser configuration

        Returns:
            The list of valid moves for the specified parser
                configuration.
        """
        moves = []
        if len(config['buffer']) > 0:
            moves.append(0)
        if len(config['stack']) > 2:
            moves.append(1)
        if len(config['stack']) > 1:
            moves.append(2)
        if len(config['stack']) > 2 and config['stack'][-1]>config['stack'][-2]:
            moves.append(3)
        return moves

    def move(self, config):
        """Executes a single move.

        Args:
            config: the current parser configuration

        Returns:
            The new parser configuration
        """
        if config['next_move'] == 0:
            config['stack'].append(config['buffer'].pop(0))
        elif config['next_move'] == 1:
            config['pred_tree'][config['stack'][-2]] = config['stack'][-1]
            del config['stack'][-2]
        elif config['next_move'] == 2:
            config['pred_tree'][config['stack'][-1]] = config['stack'][-2]
            del config['stack'][-1]
        elif config['next_move'] == 3:
            config['buffer'].insert(0, config['stack'].pop(-2))
        return config

    def is_descendant(self, tree, ancestor, descendant):
        """Returns true if a certain node is a descendant of another node or
            ancestor == descendant
        
        Args:
            tree: the dependency tree
            ancestor: the ancestor node
            descendant: the descendant node
            
        Returns:
            True or False
        """
        if ancestor == descendant:
            return True
        if descendant:
            return self.is_descendant(tree, ancestor, tree[descendant])
        else:
            return False

    def get_word_order(self, gold_tree):
        """Returns the word order such that the tree would be projective
        
        Args:
            gold_tree: the pependency tree of a sentence
            
        Returns:
            list of word indices
        """
        words = list(range(len(gold_tree)))
        tree = gold_tree.copy()
        word_order = [words.pop(0)]
        del tree[0]
        while words:
            node = word_order[-1]
            # children and their children
            if node in tree:
                for i in range(len(words)):
                    if self.is_descendant(gold_tree, node, words[i]):
                        word_order.append(words.pop(i))
                        del tree[i]
                        break
            # siblings and their children
            elif gold_tree[node] in tree:
                for i in range(len(words)):
                    if self.is_descendant(gold_tree, gold_tree[node], words[i]):
                        word_order.append(words.pop(i))
                        del tree[i]
                        break
            # parent
            elif gold_tree[node] in words:
                ind = words.index(gold_tree[node])
                word_order.append(words.pop(ind))
                del tree[ind]
            else:
                while node:
                    node = gold_tree[node]
                    # relatives
                    if node in tree:
                        for i in range(len(words)):
                            if self.is_descendant(gold_tree, node, words[i]):
                                word_order.append(words.pop(i))
                                del tree[i]
                                break
                        break
                    # ancestors
                    if node in words:
                        ind = words.index(node)
                        word_order.append(words.pop(ind))
                        del tree[ind]
        return word_order    

    def train(self, data, beam_size=10, n_epochs=1, trunc_data=None):
        """Trains the parser on training data.

        Args:
            data: Training data, a list of sentences with gold trees.
            beam_size: the width of the beam, when using beam search
            n_epochs: for how many epochs the parser should be trained 
            trunc_data: if it should stop after processing only a port of the
                        data (only used during development)
        """
        print("Training syntactic parser:")
        for e in range(n_epochs):
            print("Epoch:", e+1, "/", n_epochs)
            train_sentences_tags_trees = zip(   get_sentences(data), \
                                                get_tags(data), \
                                                get_trees(data) )
            for i, (words, gold_tags, gold_tree) in \
                                        enumerate(train_sentences_tags_trees):
                self.parse(words, gold_tree, beam_size=beam_size)
                print("\rUpdated with sentence #{}".format(i), end="")
                if trunc_data and i >= trunc_data:
                    break
            print("")
        self.finalize()

    def gold_move(self, config, gold_tree, word_order):
        """Returns the gold-standard move for the specified parser
        configuration.

        The gold-standard move is the first possible move from the
        following list: LEFT-ARC, RIGHT-ARC, SHIFT, SWAP. 

        Args:
            buffer: the current configuration of the parser
            gold_tree: The gold-standard dependency tree.
            word_order: the projective word order

        Returns:
            The gold-standard move for the specified parser
            configuration, or `None` if no move is possible.
        """
        buffer = config['buffer']
        stack = config['stack']
        pred_tree = config['pred_tree']
        left_arc_possible = False
        if len(stack) > 2 and stack[-1] == gold_tree[stack[-2]]:
            left_arc_possible = True
            for j in range(len(pred_tree)):
                if gold_tree[j] == stack[-2]:
                    if pred_tree[j] == 0:
                        left_arc_possible = False
        right_arc_possible = False
        if len(stack) > 1 and stack[-2] == gold_tree[stack[-1]]:
            right_arc_possible = True
            for j in range(len(pred_tree)):
                if gold_tree[j] == stack[-1]:
                    if pred_tree[j] == 0:
                        right_arc_possible = False
        swap_possible = False
        if len(stack) > 2 and \
            word_order.index(stack[-1]) < word_order.index(stack[-2]):
            swap_possible = True
        if left_arc_possible:
            return 1
        elif right_arc_possible:
            return 2
        elif swap_possible:
            return 3
        elif len(buffer) > 0:
            return 0
        else:
            return None

    def features(self, words, tags, config):
        """Extracts features for the specified parser configuration.

        Args:
            words: The input sentence, a list of words.
            tags: The list of tags for the input sentence.
            config: the current configuration of the parser

        Returns:
            A feature vector for the specified configuration.
        """
        buffer = config['buffer']
        stack = config['stack']
        pred_tree = config['pred_tree']

        feat = []

        # Single word features
        b1_w = words[buffer[0]] if buffer else "<empty>"
        b1_t = tags[buffer[0]] if buffer else "<empty>"
        b1_wt = b1_w + " " + b1_t

        b2_w = words[buffer[1]] if len(buffer) > 1 else "<empty>"
        b2_t = tags[buffer[1]] if len(buffer) > 1 else "<empty>"
        b2_wt = b2_w + " " + b2_t

        b3_w = words[buffer[2]] if len(buffer) > 2 else "<empty>"
        b3_t = tags[buffer[2]] if len(buffer) > 2 else "<empty>"
        b3_wt = b3_w + " " + b3_t

        s1_w = words[stack[-1]] if stack else "<empty>"
        s1_t = tags[stack[-1]] if stack else "<empty>"
        s1_wt = s1_w + " " + s1_t

        s2_w = words[stack[-2]] if len(stack) > 1 else "<empty>"
        s2_t = tags[stack[-2]] if len(stack) > 1 else "<empty>"
        s2_wt = s2_w + " " + s2_t

        '''
        for i in pred_tree:
            if stack and pred_tree[stack[-1]] == i:
                feat.append("tag" + str(i) + str(tags[i]))
        '''

        # Triple word features

        def is_parent(parent, child):
            if child == 0:
                return False
            if parent == child:
                return True
            return is_parent(parent, pred_tree[child])

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

        s2_t_s1_t_b1_t = s2_t + " " + s1_t + " " + b1_t
        if lc1_s1 >= 0:
            s2_t_s1_t_lc1_s1_t = s2_t + " " + s1_t + " " + tags[lc1_s1]
        else:
            s2_t_s1_t_lc1_s1_t = "<empty>"
        if rc1_s1 >= 0:
            s2_t_s1_t_rc1_s1_t = s2_t + " " + s1_t + " " + tags[rc1_s1]
        else:
            s2_t_s1_t_rc1_s1_t = "<empty>"
        if lc1_s2 >= 0:
            s2_t_s1_t_lc1_s2_t = s2_t + " " + s1_t + " " + tags[rc1_s2]
        else:
            s2_t_s1_t_lc1_s2_t = "<empty>"
        if rc1_s2 >= 0:
            s2_t_s1_t_rc1_s2_t = s2_t + " " + s1_t + " " + tags[rc1_s2]
        else:
            s2_t_s1_t_rc1_s2_t = "<empty>"
        if lc1_s2 >= 0:
            s2_t_s1_w_rc1_s2_t = s2_t + " " + s1_w + " " + tags[rc1_s2]
        else:
            s2_t_s1_w_rc1_s2_t = "<empty>"
        if lc1_s1 >= 0:
            s2_t_s1_w_lc1_s1_t = s2_t + " " + s1_w + " " + tags[lc1_s1]
        else:
            s2_t_s1_w_lc1_s1_t = "<empty>"

        feat.append("b1_w:" + b1_w)
        feat.append("b1_t:" + b1_t)
        feat.append("b1_wt:" + b1_wt)

        feat.append("b2_w:" + b2_w)
        feat.append("b2_t:" + b2_t)
        feat.append("b2_wt:" + b2_wt)

        feat.append("b3_w:" + b3_w)
        feat.append("b3_t:" + b3_t)
        feat.append("b3_wt:" + b3_wt)

        feat.append("s1_w:" + s1_w)
        feat.append("s1_t:" + s1_t)
        feat.append("s1_wt:" + s1_wt)

        feat.append("s2_w:" + s2_w)
        feat.append("s2_t:" + s2_t)
        feat.append("s2_wt:" + s2_wt)

        feat.append("s1_wt_s2_wt:" + s1_wt + " " + s2_wt)
        feat.append("s1_wt_s2_w:" + s1_wt + " " + s2_w)
        feat.append("s1_wt_s2_t:" + s1_wt + " " + s2_t)
        feat.append("s1_w_s2_wt:" + s1_w + " " + s2_wt)
        feat.append("s1_t_s2_wt:" + s1_t + " " + s2_wt)
        feat.append("s1_w_s2_w:" + s1_w + " " + s2_w)
        feat.append("s1_t_s2_t:" + s1_t + " " + s2_t)
        feat.append("s1_t_b1_t:" + s1_t + " " + b1_t)

        feat.append("s2_t_s1_t_b1_t:" + s2_t_s1_t_b1_t)
        feat.append("s2_t_s1_t_lc1_s1_t:" + s2_t_s1_t_lc1_s1_t)
        feat.append("s2_t_s1_t_rc1_s1_t:" + s2_t_s1_t_rc1_s1_t)
        feat.append("s2_t_s1_t_lc1_s2_t:" + s2_t_s1_t_lc1_s2_t)
        feat.append("s2_t_s1_t_rc1_s2_t:" + s2_t_s1_t_rc1_s2_t)
        feat.append("s2_t_s1_w_rc1_s2_t:" + s2_t_s1_w_rc1_s2_t)
        feat.append("s2_t_s1_w_lc1_s1_t:" + s2_t_s1_w_lc1_s1_t)


        return feat

    def finalize(self):
        """Averages the weight vectors."""
        self.classifier.finalize()