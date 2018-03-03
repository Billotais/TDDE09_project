import math

def load_data(filename):
    """Load some data from a file in the CONLLU format"""
    data = [[]]
    with open(filename, encoding='utf-8') as source:
        for line in source:
            if len(line) == 1: 
                data.append([])
            elif line.split()[0].isdigit():
                data[-1].append(line.split('\t'))
    if not data[-1]:
        data = data[:-1]
    return data

def softmax(z):
    if len(z) == 1:
        return [1]
    z_exp = [1.1**i for i in z]
    sum_z_exp = sum(z_exp)
    return [i / sum_z_exp for i in z_exp]

def accuracy(preds, golds, correction=0):
    count = -correction
    score = -correction
    for pred, gold in zip(preds, golds):
        count += 1
        if pred == gold:
            score +=1
    return score/count

def get_sentences(data):
    for sentence in data:
        out = []
        out.append("<ROOT>")
        for word in sentence:
            out.append(word[1])
        yield out

def get_tags(data):
    for sentence in data:
        out = []
        out.append("<ROOT>")
        for word in sentence:
            out.append(word[3])
        yield out

def get_trees(data):
    for sentence in data:
        out = []
        out.append(0)
        for word in sentence:
            out.append(int(word[6]))
        yield out
