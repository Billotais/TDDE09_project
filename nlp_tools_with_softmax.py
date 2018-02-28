import math

def load_data(filename):
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

def accuracy(preds, golds, correction=0):
    count = -correction
    score = -correction
    for pred, gold in zip(preds, golds):
        count += 1
        if pred == gold:
            score +=1
    return score/count

def softmax(z):
    z_exp = [math.exp(i) for i in z]
    sum_z_exp = sum(z_exp)
    return [round(i / sum_z_exp, 3) for i in z_exp]

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
