from perceptron import Perceptron
from nlp_tools import get_sentences, get_tags

class Tagger():
    """A part-of-speech tagger based on a multi-class perceptron
    classifier.

    This tagger implements a simple, left-to-right tagging algorithm
    where the prediction of the tag for the next word in the sentence
    can be based on the surrounding words and the previously
    predicted tags. The exact features that this prediction is based
    on can be controlled with the `features()` method, which should
    return a feature vector that can be used as an input to the
    multi-class perceptron.

    Attributes:
        classifier: A multi-class perceptron classifier.
    """

    def __init__(self):
        """Initialises a new tagger."""
        self.classifier = Perceptron()

    def features(self, words, i, pred_tags):
        """Extracts features for the specified tagger configuration.
        
        Args:
            words: The input sentence, a list of words.
            i: The index of the word that is currently being tagged.
            pred_tags: The list of previously predicted tags.
        
        Returns:
            A feature vector for the specified configuration.
        """
        features = []
        for n in range(4):
            features.append("w_0="+words[i])
        if words[i][0] == words[i][0].upper():
            features.append("capital_word")
        if words[i] == words[i].lower():
            features.append("lowercase")
        if i > 0:
            features.append("t_-1="+pred_tags[i-1])
            features.append("suff1_-1="+words[i-1][-1])
            features.append("suff2_-1="+words[i-1][-2:])
            features.append("suff3_-1="+words[i-1][-3:])
            features.append("pre2_-1="+words[i-1][:2])
        if i+1 < len(words):
            features.append("w_1"+words[i+1])
            features.append("suff1_1="+words[i+1][-1])
            features.append("suff2_1="+words[i+1][-2:])
            features.append("suff3_1="+words[i+1][-3:])
            features.append("pre1_1="+words[i+1][0])
            features.append("pre2_1="+words[i+1][:2])
            features.append("pre3_1="+words[i+1][:3])
        if i+2 < len(words):
            features.append("w_2"+words[i+2])
        if i+3 < len(words):
            features.append("w_3"+words[i+3])
        features.append("suff1_0="+words[i][-1])
        features.append("suff2_0="+words[i][-2:])
        features.append("suff3_0="+words[i][-3:])
        features.append("pre1_0="+words[i][0])
        features.append("pre2_0="+words[i][:2])
        features.append("pre3_0="+words[i][:3])
        return features

    def tag(self, words):
        """Tags a sentence with part-of-speech tags.

        Args:
            words: The input sentence, a list of words.

        Returns:
            The list of predicted tags for the input sentence.
        """
        pred_tags = []
        for i in range(len(words)):
            feat = self.features(words, i, pred_tags)
            tag, _ = self.classifier.predict(feat)
            pred_tags.append(tag)
        return pred_tags

    def update(self, words, gold_tags):
        """Updates the tagger with a single training instance.

        Args:
            words: The list of words in the input sentence.
            gold_tags: The list of gold-standard tags for the input
                sentence.

        Returns:
            The list of predicted tags for the input sentence.
        """
        pred_tags = []
        for i in range(len(words)):
            feat = self.features(words, i, pred_tags)
            pred_tags.append(self.classifier.update(feat, gold_tags[i]))
        return pred_tags

    def train(self, data, n_epochs=1, trunc_data=None):
        """Train a new tagger on training data.

        Args:
            data: Training data, a list of tagged sentences.
        """
        
        print("Training POS tagger")
        for e in range(n_epochs):
            print("Epoch:", e+1, "/", n_epochs)
            train_sentences_tags = zip( get_sentences(data), get_tags(data) )
            for i, (words, tags) in enumerate(train_sentences_tags):
                print("\rUpdated with sentence #{}".format(i), end="")
                pred_tags = self.update(words,tags)
                if trunc_data and i >= trunc_data:
                    break
            print("")
        self.finalize()

    def finalize(self):
        """Finalizes the classifier by averaging its weight vectors."""
        self.classifier.finalize()