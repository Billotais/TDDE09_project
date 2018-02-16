# test file for POS tagging

from tagger import Tagger
from nlp_tools import load_data, get_sentences, get_tags

def accuracy(classifier, data):
    correct = 0
    len_data = 0
    for sentence, gold_tags in zip( get_sentences(data), get_tags(data) ):
        len_data += len(sentence)
        for pred_tag,gold_tag in zip(classifier.tag(sentence), gold_tags):
            if pred_tag == gold_tag:
                correct += 1
    return(correct/len_data)

test_data = load_data("../UD_English/en-ud-test.conllu")
dev_data = load_data("../UD_English/en-ud-dev.conllu")
train_data = load_data("../UD_English/en-ud-train.conllu")

tagger = Tagger()
tagger.train(train_data, 1)

print("Dev-Accuracy {:.2%}".format(accuracy(tagger, dev_data)))
print("Test-Accuracy {:.2%}".format(accuracy(tagger, test_data)))
