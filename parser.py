from perceptron import Perceptron
from projectivize2 import projectivize
from nlp_tools import get_sentences, get_tags, get_trees

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
        classifier: A multi-class perceptron classifier used to
            predict the next move of the parser.
    """

    def __init__(self, tagger):
        """Initialises a new parser."""
        self.tagger = tagger
        self.classifier = Perceptron()

    def parse(self, words):
        """Parses a sentence.

        Args:
            words: The input sentence, a list of words.

        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """
        #"""
        tags = self.tagger.tag(words)
        pred_tree = [0] * len(words)
        stack = []
        i = 0
        while self.valid_moves(i, stack, pred_tree):
            feat = self.features(words, tags, i, stack, pred_tree)
            candidates = self.valid_moves(i, stack, pred_tree)
            move = self.classifier.predict(feat, candidates)
            i, stack, pred_tree = self.move(i, stack, pred_tree, move)
        return tags, pred_tree

    def valid_moves(self, i, stack, pred_tree):
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
        moves = []
        if i < len(pred_tree):
            moves.append(0)
        if len(stack) > 2:
            moves.append(1)
        if len(stack) > 1:
            moves.append(2)
        return moves

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
            del stack[-2]
        elif move == 2:
            pred_tree[stack[-1]] = stack[-2]
            del stack[-1]
        return i, stack, pred_tree

    def update(self, words, gold_tags, gold_tree):
        """Updates the move classifier with a single training
        instance.

        Args:
            words: The input sentence, a list of words.
            gold_tags: The list of gold-standard tags for the input
                sentence.
            gold_tree: The gold-standard tree for the sentence.

        Returns:
            A pair consisting of the predicted tags and the predicted
            dependency tree for the input sentence.
        """
        tags = self.tagger.tag(words)
        pred_tree = [0] * len(words)
        stack = []
        i = 0
        while self.valid_moves(i, stack, pred_tree):
            feat = self.features(words, tags, i, stack, pred_tree)
            gold_move = self.gold_move(i, stack, pred_tree, gold_tree)
            move = self.classifier.update(feat,gold_move)
            i, stack, pred_tree = self.move(i, stack, pred_tree, gold_move)
        return tags, pred_tree

    def train(self, data, n_epochs=1, do_projectivize=True, trunc_data=None):
        """Trains the parser on training data.

        Args:
            data: Training data, a list of sentences with gold trees.
            n_epochs:
            do_projectivize:
            trunc_data:
        """
        print("Training syntactic parser:")
        for e in range(n_epochs):
            print("Epoch:", e+1, "/", n_epochs)
            if do_projectivize:
                train_sentences_tags_trees = zip(   get_sentences(projectivize(data)), \
                                                    get_tags(projectivize(data)), \
                                                    get_trees(projectivize(data)) )
            else:
                train_sentences_tags_trees = zip(   get_sentences(data), \
                                                    get_tags(data), \
                                                    get_trees(data) )
            for i, (words, gold_tags, gold_tree) in enumerate(train_sentences_tags_trees):
                self.update(words, gold_tags, gold_tree)
                print("\rUpdated with sentence #{}".format(i), end="")
                if trunc_data and i >= trunc_data:
                    break
            print("")
        self.finalize()

    def gold_move(self, i, stack, pred_tree, gold_tree):
        """Returns the gold-standard move for the specified parser
        configuration.

        The gold-standard move is the first possible move from the
        following list: LEFT-ARC, RIGHT-ARC, SHIFT. LEFT-ARC is
        possible if the topmost word on the stack is the gold-standard
        head of the second-topmost word, and all words that have the
        second-topmost word on the stack as their gold-standard head
        have already been assigned their head in the predicted tree.
        Symmetric conditions apply to RIGHT-ARC. SHIFT is possible if
        at least one word in the input sentence still requires
        processing.

        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            gold_tree: The gold-standard dependency tree.

        Returns:
            The gold-standard move for the specified parser
            configuration, or `None` if no move is possible.
        """
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
        if left_arc_possible:
            return 1
        elif right_arc_possible:
            return 2
        elif i < len(pred_tree):
            return 0
        else:
            return None

    def features(self, words, tags, i, stack, parse):
        """Extracts features for the specified parser configuration.

        Args:
            words: The input sentence, a list of words.
            gold_tags: The list of gold-standard tags for the input
                sentence.
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            parse: The partial dependency tree.

        Returns:
            A feature vector for the specified configuration.
        """
        feat = []

        # Single word features
        b1_w = words[i] if len(words) > i else "<empty>"
        b1_t = tags[i] if len(words) > i else "<empty>"
        b1_wt = b1_w + " " + b1_t

        s1_w = words[stack[-1]] if stack else "<empty>"
        s1_t = tags[stack[-1]] if stack else "<empty>"
        s1_wt = s1_w + " " + s1_t

        s2_w = words[stack[-2]] if len(stack) > 1 else "<empty>"
        s2_t = tags[stack[-2]] if len(stack) > 1 else "<empty>"
        s2_wt = s2_w + " " + s2_t

        # Double words features

        s1_wt_s2_wt = s1_wt + " " + s2_wt
        s1_wt_s2_w = s1_wt + " " + s2_w
        s1_wt_s2_t = s1_wt + " " + s2_t
        s1_w_s2_wt = s1_w + " " + s2_wt
        s1_t_s2_wt = s1_t + " " + s2_wt
        s1_w_s2_w = s1_w + " " + s2_w
        s1_t_s2_t = s1_t + " " + s2_t
        s1_t_b1_t = s1_t + " " + b1_t




        feat.append("b1_w:" + b1_w)
        feat.append("b1_t:" + b1_t)
        feat.append("b1_wt:" + b1_wt)

        feat.append("s1_w:" + s1_w)
        feat.append("s1_t:" + s1_t)
        feat.append("s1_wt:" + s1_wt)

        feat.append("s2_w:" + s2_w)
        feat.append("s2_t:" + s2_t)
        feat.append("s2_wt:" + s2_wt)

        feat.append("s1_wt_s2_wt:" + s1_wt_s2_wt)
        feat.append("s1_wt_s2_w:" + s1_wt_s2_w)
        feat.append("s1_wt_s2_t:" + s1_wt_s2_t)
        feat.append("s1_w_s2_wt:" + s1_w_s2_wt)
        feat.append("s1_t_s2_wt:" + s1_t_s2_wt)
        feat.append("s1_w_s2_w:" + s1_w_s2_w)
        feat.append("s1_t_s2_t:" + s1_t_s2_t)
        feat.append("s1_t_b1_t:" + s1_t_b1_t)




        return feat

    def finalize(self):
        """Averages the weight vectors."""
        self.classifier.finalize()
