import itertools, pickle, argparse
from parser import Parser
from tagger import Tagger
from nlp_tools import load_data, accuracy, get_sentences, get_tags, get_trees

argparser = argparse.ArgumentParser(description='Evaluate the NLP system.')
argparser.add_argument('evaluation_data', metavar='path to evaluation data', type=str,
                    help='data for evaluating the tagger and the parser in CoNLL-U format')
argparser.add_argument('-t', '--train', metavar='path to training data', type=str,
                    help='data for training the tagger and the parser in CoNLL-U format')
argparser.add_argument('-l', '--load', metavar='file', type=str, 
                    help='load existing config from file')
argparser.add_argument('-s', '--save', metavar='file', type=str, 
                    help='save config to file')
args = argparser.parse_args()

if not args.train and not args.load:
    exit("error: either option --train or option --load is required")

if args.train:
    train_data = load_data(args.train)
    tagger = Tagger()
    tagger.train(train_data, n_epochs=3, trunc_data=None)
    parser = Parser(tagger)
    parser.train(train_data, n_epochs=3, trunc_data=None)

if args.train and args.save:
    with open(args.save, 'wb') as output:
        pickle.dump(parser, output, -1)

if args.load:
    with open(args.load, 'rb') as input:
        parser = pickle.load(input)

print("Evaluation:")
dev_data = load_data(args.evaluation_data)
gold_tags = list(itertools.chain(*get_tags(dev_data)))
gold_trees = list(itertools.chain(*get_trees(dev_data)))
pred_tags_lst = []
pred_trees_lst = []
pred_trees = []
correction = 0
for i, sentence in enumerate(get_sentences(dev_data)):
    print("\rEvaluated with sentence #{}".format(i), end="")
    pred_tags, pred_tree = parser.parse(sentence, beam_thresh=0, beam_size=1)
    pred_tags_lst += pred_tags
    pred_trees_lst += pred_tree
    pred_trees.append(pred_tree)
    correction += 1
print("")

print("Tagging accuracy: {:.2%}".format(accuracy(pred_tags_lst, gold_tags, correction)))
print("Unlabelled attachment score: {:.2%}".format(accuracy(pred_trees_lst, gold_trees, correction)))
print("Exact matches: {:.2%}".format(accuracy(pred_trees, get_trees(dev_data), 0)))
