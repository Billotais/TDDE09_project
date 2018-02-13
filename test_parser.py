from parser import Parser

def trees(data):
    for sentence in data:
        out = ([], [], [])
        out[0].append("<ROOT>")
        out[1].append("<ROOT>")
        out[2].append(0)
        for word in sentence:
            out[0].append(word[1])
            out[1].append(word[3])
            out[2].append(int(word[6]))
        yield out


def load_data(filename):
    data = [[]]
    with open(filename) as source:
        for line in source:
            if len(line) == 1: 
                data.append([])
            elif line.split()[0].isdigit():
                data[-1].append(line.split())
    if not data[-1]:
        data = data[:-1]
    return data


n_examples = 200    # Set to None to train on all examples
#n_examples = None    # Set to None to train on all examples

parser = Parser()
#train_data = load_data("../UD_English/en-ud-train.conllu")
train_data = load_data("../UD_English/en-ud-train-projective.conllu")
for i, (words, gold_tags, gold_tree) in enumerate(trees(train_data)):
    parser.update(words, gold_tags, gold_tree)
    print("\rUpdated with sentence #{}".format(i), end="")
    if n_examples and i >= n_examples:
        break
print("")
parser.finalize()

acc_k = acc_n = 0
uas_k = uas_n = 0
dev_data = load_data("../UD_English/en-ud-dev.conllu")
for i, (words, gold_tags, gold_tree) in enumerate(trees(dev_data)):
    pred_tags, pred_tree = parser.parse(words)
    acc_k += sum(int(g == p) for g, p in zip(gold_tags, pred_tags)) - 1
    acc_n += len(words) - 1
    uas_k += sum(int(g == p) for g, p in zip(gold_tree, pred_tree)) - 1
    uas_n += len(words) - 1
    print("\rParsing sentence #{}".format(i), end="")
print("")
print("Tagging accuracy: {:.2%}".format(acc_k / acc_n))
print("Unlabelled attachment score: {:.2%}".format(uas_k / uas_n))