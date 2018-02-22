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
