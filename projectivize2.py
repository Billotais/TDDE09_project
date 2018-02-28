from nlp_tools import get_trees

def calc_score_matrix(parent):
    n = len(parent)
    A = {i: {j: float("-inf") for j in range(n)} for i in range(n)}
    for i in range(len(parent)):
        score = 0
        ancestor = parent[i]
        while ancestor != 0:
            A[ancestor][i] = score
            ancestor = parent[ancestor]
            score -= 1
        A[ancestor][i] = score
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

def eisner(A):
    n = len(A)
    T1 = {i: {i: 0} for i in range(n)}
    T2 = {i: {i: 0} for i in range(n)}
    T3 = {i: {} for i in range(n)}
    T4 = {i: {} for i in range(n)}
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

def projectivize(data):
    for parent, sentence in zip(get_trees(data), data):
        A = calc_score_matrix(parent)
        proj_parent = eisner(A)
        for i in range(len(sentence)):
            sentence[i][6] = str(proj_parent[i+1])
        yield sentence

def cmd_projectivize():
    import sys
    data = [[]]
    for line in sys.stdin:
        if len(line) == 1:
            data.append([])
        elif line.split()[0].isdigit():
            data[-1].append(line[:-1].split("\t"))
    if not data[-1]:
        data = data[:-1]
    for sentence in projectivize(data):
        for word in sentence:
            print("\t".join(word))
        print("")

if __name__ == "__main__":
    cmd_projectivize()
