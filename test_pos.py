# test file for POS tagging

from tagger import Tagger

def load_data(filename):
    data = [[]]
    with open(filename) as source:
        for line in source:
            if len(line) == 1: 
                data.append([])
            elif line.split()[0].isdigit():
                data[-1].append((line.split()[1], line.split()[3]))
    return data

def accuracy(classifier, data):
    correct = 0
    len_data = 0
    for item in data:
        sentence = []
        gold_tags = []
        len_data += len(item)
        for i in range(len(item)):
            sentence.append(item[i][0])
            gold_tags.append(item[i][1])
        for pred_tag,gold_tag in zip(classifier.tag(sentence), gold_tags):
            if pred_tag == gold_tag:
                correct += 1
    return(correct/len_data)

test_data = load_data("../UD_English/en-ud-test.conllu")
dev_data = load_data("../UD_English/en-ud-dev.conllu")
train_data = load_data("../UD_English/en-ud-train.conllu")

tagger = Tagger()
tagger.train(train_data, 2)

print("Dev-Accuracy {:.2%}".format(accuracy(tagger, dev_data)))
print("Test-Accuracy {:.2%}".format(accuracy(tagger, test_data)))
