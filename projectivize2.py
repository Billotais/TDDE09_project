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
            A[ancestor][i] = cost
            ancestor = parent[ancestor]
            cost -= 1
        A[ancestor][i] = cost
    return A

def restore_tree(tree, T1_back, T2_back, T3_back, T4_back, T_name_in, inds_in):
    if T_name_in == "A":
        tree[inds_in[1]] = inds_in[0]
        return tree
    elif T_name_in == "T1":
        T_back = T1_back
    elif T_name_in == "T2":
        T_back = T2_back
    elif T_name_in == "T3":
        T_back = T3_back
    elif T_name_in == "T4":
        T_back = T4_back
    if inds_in[0] != inds_in[1]:
        for T_name_out, inds_out in T_back[inds_in[0]][inds_in[1]].items():
            tree = restore_tree(tree, T1_back, T2_back, T3_back, T4_back, \
                                    T_name_out, inds_out)
    return tree

def eisner(sentence, A):
    n = len(sentence)
    T1 = np.zeros((n,n))
    T2 = np.zeros((n,n))
    T3 = np.zeros((n,n))
    T4 = np.zeros((n,n))
    T1_back = {i: {} for i in range(n)}
    T2_back = {i: {} for i in range(n)}
    T3_back = {i: {} for i in range(n)}
    T4_back = {i: {} for i in range(n)}
    for k in range(1, n):
        for i in range(k-1, -1, -1):
            best_score = float("-inf")
            for j in range(i, k):
                curr_score = T2[i][j] + T1[j+1][k] + A[i][k]
                if curr_score > best_score:
                    best_score = curr_score
                    T4_back[i][k] = {"T2": (i,j), "T1": (j+1,k), "A": (i,k)}
            T4[i][k] = best_score
            best_score = float("-inf")
            for j in range(i, k):
                curr_score = T2[i][j] + T1[j+1][k] + A[k][i]
                if curr_score > best_score:
                    best_score = curr_score
                    T3_back[i][k] = {"T2": (i,j), "T1": (j+1,k), "A": (k,i)}
            T3[i][k] = best_score
            best_score = float("-inf")
            for j in range (i+1, k+1):
                curr_score = T4[i][j] + T2[j][k]
                if curr_score > best_score:
                    best_score = curr_score
                    T2_back[i][k] = {"T4": (i,j), "T2": (j,k)}
            T2[i][k] = best_score
            best_score = float("-inf")
            for j in range(i, k):
                curr_score =  T1[i][j] + T3[j][k]
                if curr_score > best_score:
                    best_score = curr_score
                    T1_back[i][k] = {"T1": (i,j), "T3": (j,k)}
            T1[i][k] = best_score
    tree = restore_tree([0]*n, T1_back, T2_back, T3_back, T4_back, "T2", (0, n-1))
    return tree

#dev_data = load_data("../UD_English/en-ud-dev-projective.conllu")
dev_data = load_data("../UD_English/en-ud-dev.conllu")

for sentence, parent in trees(dev_data):
    A = calc_score_matrix(sentence, parent)
    proj_parent = eisner(sentence, A)
    if proj_parent != parent:
        print(proj_parent, parent)
