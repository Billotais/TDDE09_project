from perceptron import Perceptron

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
        features.append("w_0="+words[i])
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
            pred_tags.append(self.classifier.predict(feat))
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

    def train(self, data, n_epochs=1):
        """Train a new tagger on training data.

        Args:
            data: Training data, a list of tagged sentences.
        """
        for e in range(n_epochs):
            for sentence in data:
                words = []
                tags = []
                for word in sentence:
                    words.append(word[0])
                    tags.append(word[1])
                pred_tags = self.update(words,tags)
        self.finalize()

    def finalize(self):
        """Finalizes the classifier by averaging its weight vectors."""
        self.classifier.finalize()