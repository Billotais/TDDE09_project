import sys, itertools
from parser_hybrid import Parser
from tagger import Tagger
from nlp_tools import load_data, accuracy, get_sentences, get_tags, get_trees

if len(sys.argv) != 3:
    exit("Usage:\n python evalute_system.py [path to training data] [path to evaluation data]")

train_data = load_data(sys.argv[1])
tagger = Tagger()
tagger.train(train_data, n_epochs=3, trunc_data=None)
parser = Parser(tagger)
parser.train(train_data, n_epochs=3, trunc_data=None)

print("Evaluation:")
dev_data = load_data(sys.argv[2])
gold_tags = list(itertools.chain(*get_tags(dev_data)))
gold_trees = list(itertools.chain(*get_trees(dev_data)))
pred_tags_lst = []
pred_trees_lst = []
correction = 0
for i, sentence in enumerate(get_sentences(dev_data)):
    print("\rEvaluated with sentence #{}".format(i), end="")
    pred_tags, pred_tree = parser.parse(sentence, beam_thresh=10, beam_size=1)
    pred_tags_lst += pred_tags
    pred_trees_lst += pred_tree
    correction += 1
print("")

print("Tagging accuracy: {:.2%}".format(accuracy(pred_tags_lst, gold_tags, correction)))
print("Unlabelled attachment score: {:.2%}".format(accuracy(pred_trees_lst, gold_trees, correction)))
