import sys
from parser import Parser
from tagger import Tagger
from projectivize2 import projectivize
from nlp_tools import load_data, get_sentences, get_tags, get_trees

if len(sys.argv) != 3:
    exit("Usage:\n python evalute_system.py [path to training data] [path to evaluation data]")

n_examples = None
train_data = load_data(sys.argv[1])
do_projectivize = True
tagger = Tagger()
tagger.train(train_data, 3)

parser = Parser(tagger)
parser.train(train_data, 3, do_projectivize, n_examples)

acc_k = acc_n = 0
uas_k = uas_n = 0
dev_data = load_data(sys.argv[2])
dev_sentences_tags_trees = zip( get_sentences(dev_data), \
                                get_tags(dev_data), \
                                get_trees(dev_data) )
for i, (words, gold_tags, gold_tree) in enumerate(dev_sentences_tags_trees):
    pred_tags, pred_tree = parser.parse(words)
    acc_k += sum(int(g == p) for g, p in zip(gold_tags, pred_tags)) - 1
    acc_n += len(words) - 1
    uas_k += sum(int(g == p) for g, p in zip(gold_tree, pred_tree)) - 1
    uas_n += len(words) - 1
    print("\rParsing sentence #{}".format(i), end="")
print("")
print("Tagging accuracy: {:.2%}".format(acc_k / acc_n))
print("Unlabelled attachment score: {:.2%}".format(uas_k / uas_n))
