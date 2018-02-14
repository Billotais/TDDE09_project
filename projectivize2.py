import numpy as np

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

def trees(data):
    for sentence in data:
        out = ([], [])
        out[0].append("<ROOT>")
        out[1].append(0)
        for word in sentence:
            out[0].append(word[1])
            out[1].append(int(word[6]))
        yield out

def calc_score_matrix(sentence, parent):
    A = np.ones((len(sentence), len(sentence)))*float("-inf")
    for i in range(len(sentence)):
        cost = 0
        ancestor = parent[i]
        while ancestor != 0:
            A[i][ancestor] = cost
            ancestor = parent[ancestor]
            cost -= 1
    return A


def eisner(sentence, A):
    n = len(sentence)
    """
    T1 = np.zeros((n,n))
    T2 = np.zeros((n,n))
    T3 = np.ones((n,n))*float("-inf")
    T4 = np.ones((n,n))*float("-inf")
    """
    T1 = np.ones((n,n))*float("-inf")
    T2 = np.ones((n,n))*float("-inf")
    T3 = np.ones((n,n))*float("-inf")
    T4 = np.ones((n,n))*float("-inf")
    np.fill_diagonal(T1, 0)
    np.fill_diagonal(T2, 0)
    np.fill_diagonal(T3, 0)
    np.fill_diagonal(T4, 0)
    for k in range(0, n):
        for i in range(k-2, 0):
            best_score = float("-inf")
            for j in range(i, k-1+1):
                curr_score = T2[i][j] + T1[j+1][k] + A[i][k]
                if curr_score > best_score:
                    best_score = curr_score
            T4[i][k] = best_score
            best_score = float("-inf")
            for j in range(i, k-1+1):
                curr_score =  T2[i][j] + T1[j+1][k] + A[k][i]
                if curr_score > best_score:
                    best_score = curr_score
            T3[i][k] = best_score
            best_score = float("-inf")
            for j in range (i+1, k+1):
                curr_score = T4[i][j] + T2[j][k]
                if curr_score > best_score:
                    best_score = curr_score
            T2[i][k] = best_score
            best_score = float("-inf")
            for j in range(i, k-1+1):
                curr_score =  T1[i][j] + T3[j][k]
                if curr_score > best_score:
                    best_score = curr_score
            T1[i][k] = best_score
        if T2[0][-1] != 0:
            print(T2[0][-1])


dev_data = load_data("../UD_English/en-ud-train.conllu")

for sentence, parent in trees(dev_data):
    A = calc_score_matrix(sentence, parent)
    eisner(sentence, A)


    #print(A)
