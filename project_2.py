import numpy as np
from sklearn.cluster import KMeans

# Load server KP descriptors
S = np.load('descriptors_150k.mat', allow_pickle=True)
d = np.array(S['descriptors']).T
d = d[np.lexsort((d[:, 128],))]

# Build voc. tree
b = 4
depth = 5

def hi_kmeans(data, b, depth):
    num_samples, num_features = data.shape
    tree_idx = np.zeros((num_samples, depth), dtype=int)
    
    kmeans = KMeans(n_clusters=b)
    idx = kmeans.fit_predict(data)
    tree_idx[:, depth - 1] = idx
    depth -= 1
    
    sub = []
    for i in range(b):
        if (np.sum(idx == i) >= b) and depth > 1:
            aux, aux_idx = hi_kmeans(data[idx == i], b, depth)
            sub.extend(aux)
            tree_idx[idx == i, :aux_idx.shape[1]] = aux_idx
        elif (np.sum(idx == i) >= b) and depth == 1:
            kmeans = KMeans(n_clusters=b)
            aux_idx = kmeans.fit_predict(data[idx == i])
            C = kmeans.cluster_centers_
            aux = {'centers': C, 'sub': []}
            sub.append(aux)
            tree_idx[idx == i, 0] = aux_idx
        elif (np.sum(idx == i) < b):
            aux = {'centers': np.empty((0, num_features)), 'sub': []}
            sub.append(aux)
            tree_idx[idx == i, :] = 1
        
    tree = {'centers': kmeans.cluster_centers_, 'sub': sub}
    
    return tree, tree_idx

def hi_push(tree, data):
    # This function pushes a query feature into the given tree and outputs the
    # path this feature follows until it reaches a leaf node.
    
    path = []
    path_c = 'tree.centers'
    path_str = 'tree.sub'
    
    for i in range(tree['depth']):
        center_aux = eval(path_c)
        d = np.linalg.norm(data - np.array(center_aux).astype(float), axis=1)
        min_idx = np.argmin(d)
        path.append(min_idx)
        
        path_str = path_str[:-3] + 'sub[' + str(path[i]) + '].sub'
        str_aux = eval(path_str)
        
        path_c = path_c[:-7] + 'sub[' + str(path[i]) + '].centers'
        
        if str_aux is None:
            if eval(path_c) is None:
                path.append(1)
            else:
                center_aux = eval(path_c)
                d = np.linalg.norm(data - np.array(center_aux).astype(float), axis=1)
                min_idx = np.argmin(d)
                path.append(min_idx)
            break
    
    return path

tree, idx = hi_kmeans(d[:, :128], b, depth)
tree['depth'] = depth
tree['b'] = b
idx = np.fliplr(idx)
words = np.unique(idx, axis=0)

# Compute TF-IDF weights for each visual word (w(i,j))
K = np.max(d[:, 128])  # # total number of obj
F = np.zeros((K, 1))  # # of words in object j
Ki = np.zeros((1, len(words)))  # # of obj containing word i
f = np.zeros((K, len(words)))  # # of times word i appears in obj j

for i in range(len(words)):
    occurrences = np.sum(np.all(idx == words[i], axis=1), axis=0) == depth
    obj_occ = d[:, 128] * np.double(occurrences)
    for j in range(K):
        f[j, i] = np.sum(obj_occ == j)
        F[j, 0] += f[j, i]
        if f[j, i] > 0:
            Ki[0, i] += 1

a = (f / F)
b = np.log2(K / Ki)
w = np.dot(a, b.T)

# Get descriptors of the query images
C = np.load('descriptors_Q_150k.mat', allow_pickle=True)
d_Q = np.array(C['descriptors_Q']).T
# Select a percentage
perc = 0.9
perm = np.random.permutation(d_Q.shape[0])
sel = perm[:int(perc * d_Q.shape[0])]
d_Q = d_Q[sel, :]
# Order by object
d_Q = d_Q[np.lexsort((d_Q[:, 128],))]

# Push every KP into the vocabulary tree and find which path follows
query_paths = np.zeros((d_Q.shape[0], depth), dtype=int)
for q in range(d_Q.shape[0]):
    aux = hi_push(tree, d_Q[q, :128])
    query_paths[q, :aux.shape[0]] = aux

# Assign a visual word to each KP
kp_words = np.zeros((d_Q.shape[0],), dtype=int)
for k in range(len(words)):
    index = np.all(query_paths == words[k], axis=1) == depth
    kp_words = np.double(index) * k + kp_words

# Build score matrix
K_Q = np.max(d_Q[:, 128])  # Number of query objects
score = np.zeros((K_Q, K))  # Scores of every query obj to every server obj

for k in range(K_Q):
    aux = np.zeros((1, K))
    objQ_words = np.double(d_Q[:, 128] == k) * kp_words
    for i in range(len(words)):
        rep_word = np.sum(objQ_words == i)  # how many times word i appears in query object k
        for j in range(K):
            a = (f[j, i] / F[j, 0])
            b = np.log2(K / Ki[0, i])
            tfidf = a * b
            score[k, j] += tfidf * rep_word

# Compute Recall Rate
max_idx = np.argmax(score, axis=1)
corrects = np.sum(max_idx == np.arange(1, 51))
recall_rate = corrects / K_Q

score_5 = np.copy(score)
corrects_5 = 0
for n in range(5):
    max_idx = np.argmax(score_5, axis=1)
    corrects_5 += np.sum(max_idx == np.arange(1, 51))
    score_5[:, max_idx] = 0
recall_rate_5 = corrects_5 / K_Q
