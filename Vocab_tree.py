from sklearn.cluster import KMeans
import pickle
import numpy as np
import copy

class Tree:

  def __init__(self, data, depth, center=None) -> None:
    self.depth = depth #what is the current depth of the node
    self.children = []
    self.center = center
    self.data = data #(n_samples, n_features)
    self.logK_Ki = None
    self.visual_word_no = None
    self.kmeans = None


  def IsEmpty(self, b):
    # return len(self.data) == 0 old version
    return len(self.data) <= b+1
  
  def cluster(self, branch_factor):
    #use KMeans algorithnm on data
    # kmeans = KMeans(n_clusters=branch_factor, random_state=0, n_init="auto")
    self.kmeans = KMeans(n_clusters=branch_factor, random_state=10)
    labels = self.kmeans.fit_predict(self.data, y=None, sample_weight=None)
    for i in range(branch_factor):
      # idx_i = [j for j,label in enumerate(labels) if label == i]
      data_i = [d for j,d in enumerate(self.data) if labels[j] == i]
      cluster_center_i = self.kmeans.cluster_centers_[i,:]
      child = Tree(data=data_i, depth=self.depth + 1,center=cluster_center_i)
      self.children.append(child)
    

  def IsLeafNode(self, max_depth):
    return max_depth == self.depth
  
  # def SetLogK_Ki(self):
  #   pass

  # def Push(self):
  #   pass


class HKMeans():
  
  def __init__(self,data, b, depth):
    self.data = data
    self.b = b
    self.max_depth = depth
    self.root = Tree(self.data, 0)
    self.vw_no = 0
    
  
  def ConstructVocab(self, data, b, depth, node):
    if depth != 0 and not node.IsEmpty(b):
      node.cluster(self.b)
      for child in node.children:
        self.ConstructVocab(child.data, self.b, depth-1, child)
    else:
      pass

  def No_visualWords(self, node):
    if node.IsLeafNode(self.max_depth):
      node.visual_word_no = self.vw_no
      self.vw_no += 1
    elif not node.IsEmpty(self.b):
      for child in node.children:
        self.No_visualWords(child)
    else:
      pass

  def Assign_Ki(self, Ki_vector, node):
    if node.IsLeafNode(self.max_depth):
      K = 50
      node.logK_Ki = np.log(K/Ki_vector[node.visual_word_no])
    elif not node.IsEmpty(self.b):
      for child in node.children:
        self.Assign_Ki(Ki_vector, child)
    else:
      pass  



def Transform_data(filename):
  # Read dictionary pkl file
  # with open('database_ft.pkl', 'rb') as fp:
  with open(filename, 'rb') as fp:
    database_ft = pickle.load(fp)
  
  database_ft_ls = []
  for key in database_ft:
    # for i in range(3):
    # print(len(database_ft[key]))
    for v in database_ft[key][0]:
      database_ft_ls.append(v)
      # print(len(database_ft[key][i]))

  data = np.zeros((len(database_ft_ls), len(database_ft_ls[0])))
  for i,v in enumerate(database_ft_ls):
    data[i,:] = v
  
  return data


def Binary_presence_vector(obj_vectors, hk_means_):
  p_vector = np.zeros((hk_means_.vw_no,1), dtype = int)
  for v in obj_vectors:
    hk_means_obj = copy.copy(hk_means_)
    vw_found = False
    current_node = hk_means_obj.root
    v = np.array(v, dtype=float)
    v = np.reshape(v,(1,len(v)))
    # print(v.shape)
    while not vw_found:
      # v.reshape(1,-1)
      ind = current_node.kmeans.predict(v)
      current_node = current_node.children[ind[0]]
      if current_node.IsLeafNode(hk_means_obj.max_depth):
        p_vector[current_node.visual_word_no] = 1
        vw_found = True
  
  return p_vector

def Fij_vector(obj_vectors, hk_means_):
  fij_vector = np.zeros((hk_means_.vw_no,1), dtype = int)
  for v in obj_vectors:
    hk_means_obj = copy.copy(hk_means_)
    vw_found = False
    current_node = hk_means_obj.root
    v = np.array(v, dtype=float)
    v = np.reshape(v,(1,len(v)))
    # print(v.shape)
    while not vw_found:
      # v.reshape(1,-1)
      ind = current_node.kmeans.predict(v)
      current_node = current_node.children[ind[0]]
      if current_node.IsLeafNode(hk_means_obj.max_depth):
        fij_vector[current_node.visual_word_no] += 1
        vw_found = True
  
  return fij_vector

def BPandFij(obj_vectors, hk_means_):
  p_vector = np.zeros((hk_means_.vw_no,1), dtype = int)
  fij_vector = np.zeros((hk_means_.vw_no,1), dtype = int)
  for v in obj_vectors:
    hk_means_obj = copy.copy(hk_means_)
    vw_found = False
    current_node = hk_means_obj.root
    v = np.array(v, dtype=float)
    # print(v.shape)
    v = np.reshape(v,(1,len(v)))
    while not vw_found:
      # v.reshape(1,-1)
      ind = current_node.kmeans.predict(v)
      current_node = current_node.children[ind[0]]
      if current_node.IsLeafNode(hk_means_obj.max_depth):
        p_vector[current_node.visual_word_no] = 1
        fij_vector[current_node.visual_word_no] += 1
        vw_found = True
      elif current_node.IsEmpty(hk_means_obj.b):
        vw_found = True
      
  
  return p_vector, fij_vector

def Read_data_obj(filename):
  # Read dictionary pkl file
  # with open('database_ft.pkl', 'rb') as fp:
  with open(filename, 'rb') as fp:
    database_ft = pickle.load(fp)

  data_dict = {} 
  
  temp_ls = []
  for key in database_ft:
    temp_ls = []
    # for i in range(3):
    for v in database_ft[key][0]:
      temp_ls.append(v)
    data_dict[key] = temp_ls
  
  return data_dict

def Read_query_obj(filename):
  # Read dictionary pkl file
  # with open('database_ft.pkl', 'rb') as fp:
  with open(filename, 'rb') as fp:
    query_ft = pickle.load(fp)

  data_dict = {} 
  
  temp_ls = []
  for key in query_ft:
    temp_ls = []
    print(len(query_ft[key]))
    for v in query_ft[key]:
      temp_ls.append(v)
    data_dict[key] = temp_ls
  
  return data_dict

def main():

  database_data = Transform_data('database_ft_new.pkl')
  # print(database_data.shape )

  branch = 5
  depth = 7
  hk_means_obj = HKMeans(database_data, branch, depth)
  hk_means_obj.ConstructVocab(hk_means_obj.data, branch, depth, hk_means_obj.root)

  hk_means_obj.No_visualWords(hk_means_obj.root)
  # print(hk_means_obj.vw_no)

  database_dict = Read_data_obj('database_ft_new.pkl')

  Ki_vector = np.zeros((hk_means_obj.vw_no,1), dtype=int)
  fij_vectors = {}
  for key in database_dict:
    obj_vectors = database_dict[key]
    p_vector, fij = BPandFij(obj_vectors, hk_means_obj)
    Ki_vector += p_vector
    fij_vectors[key] = fij
  
  print(Ki_vector)
  # np.save('b5_d7_Ki_new', Ki_vector)
  # with open('b5_d7_Fij_new.pkl', 'wb') as fp:
  #   pickle.dump(fij_vectors, fp)
  

  hk_means_obj.Assign_Ki(Ki_vector, hk_means_obj.root)

  

  # TF_IDF_scores_db = np.zeros((hk_means_obj.vw_no, 50)) 
  # for i, key in enumerate(fij_vectors.keys()):
  #   for j, ki in enumerate(Ki_vector):
  #     TF_IDF_scores_db[j,i] = fij_vectors[key][j]/len(database_dict[key]) * np.log(50/ki)

  #Ki and Fij commented above for database extraction

  Ki_v = np.load('b5_d7_Ki_new.npy')
  # print(len(Ki_v))
  # TF_IDF_scores_q = query(hk_means_obj,Ki_v)
  ratio = 0.9
  query_ratio(hk_means_obj, Ki_v, ratio)


def query(hkmeans, Ki_v):
  query_dict = Read_query_obj('query_ft_new.pkl')
  
  
  fij_vectors = {}
  for key in query_dict:
    obj_vectors = query_dict[key]
    p_vector, fij = BPandFij(obj_vectors, hkmeans)
    fij_vectors[key] = fij

  with open('b5_d7_Fij_query_new.pkl', 'wb') as fp:
    pickle.dump(fij_vectors, fp)  

  TF_IDF_scores_query = np.zeros((hkmeans.vw_no, 50)) 
  for i, key in enumerate(fij_vectors.keys()):
    for j, ki in enumerate(Ki_v):
      TF_IDF_scores_query[j,i] = fij_vectors[key][j]/len(query_dict[key]) * np.log(50/ki)

  return TF_IDF_scores_query

def TF_IDF_comparison():
  with open('b5_d7_Fij_query_09.pkl', 'rb') as fp:
    fij_query = pickle.load(fp)
  with open('b5_d7_Fij_new.pkl', 'rb') as fp:
    fij_datab = pickle.load(fp)
  Ki_v = np.load('b5_d7_Ki_new.npy')
   
  # Fj_q = np.zeros((len(Ki_v), 50)) 
  # Fj_d = np.zeros((len(Ki_v), 50))  
  Fj_q = np.zeros((50,1)) 
  Fj_d = np.zeros((50,1))  

  for key in fij_datab:
    for val in fij_datab[key]:
      # if key == 1 : #del this
      #   print(val) #delete this
      Fj_d[key-1] += val

  for key in fij_query:
    for val in fij_query[key]:
      Fj_q[key-1] += val

  
  TF_IDF_scores_datab = np.zeros((len(Ki_v), 50)) 
  TF_IDF_scores_query = np.zeros((len(Ki_v), 50)) 
  for key in fij_query:
    for j, ki in enumerate(Ki_v):
      TF_IDF_scores_query[j,key-1] = fij_query[key][j]/Fj_q[key-1] * np.log(50/ki)

  for key in fij_datab:
    for j, ki in enumerate(Ki_v):
      TF_IDF_scores_datab[j,key-1] = fij_datab[key][j]/Fj_d[key-1] * np.log(50/ki)
    
  # np.save('TFIDF_db_b4_d3', TF_IDF_scores_datab)
  # np.save('TFIDF_q_b4_d3', TF_IDF_scores_query)

  TFIDF_query_comp = np.zeros((50, 50)) #query x datab
  for i in range(50): 
    for j in range(50):
      temp = TF_IDF_scores_datab[:,j] - TF_IDF_scores_query[:,i]
      TFIDF_query_comp[i,j] = np.linalg.norm(temp)

  # TFIDF_sorted = np.sort(TFIDF_query_comp, axis = 1)
  TFIDF_argsort = np.argsort(TFIDF_query_comp, axis = 1)
  top1_recall = np.zeros((50,1))
  top5_recall = np.zeros((50,1))

  for i in range(50):
    if TFIDF_argsort[i,0] == i:
      top1_recall[i] = 1
    for j in range(5):
      if TFIDF_argsort[i,j] == i:
        top5_recall[i] = 1

  avg_top1 = np.sum(top1_recall) / len(top1_recall)
  avg_top5 = np.sum(top5_recall) / len(top5_recall)
  print(avg_top1)
  print(avg_top5)
  # print(len(Ki_v))
  # print(TFIDF_query_comp)
  # database_dict = Read_data_obj('database_ft.pkl')

  # print(*TFIDF_argsort)
  # print(TF_IDF_scores_datab)
  # print(TF_IDF_scores_query)
  # print(fij_datab)
  # print(Fj_d)
  # print(Fj_q)
  # print(*Ki_v)

def query_ratio(hkmeans, Ki_v, ratio):
  query_dict = Read_query_obj('query_ft_new.pkl')
  
  fij_vectors = {}
  for key in query_dict:
    N = len(query_dict[key])
    obj_vectors = query_dict[key][:int(ratio*N)]
    p_vector, fij = BPandFij(obj_vectors, hkmeans)
    fij_vectors[key] = fij

  with open('b5_d7_Fij_query_09.pkl', 'wb') as fp:
    pickle.dump(fij_vectors, fp)  

  # TF_IDF_scores_query = np.zeros((hkmeans.vw_no, 50)) 
  # for i, key in enumerate(fij_vectors.keys()):
  #   for j, ki in enumerate(Ki_v):
  #     TF_IDF_scores_query[j,i] = fij_vectors[key][j]/len(query_dict[key]) * np.log(50/ki)

  # return TF_IDF_scores_query


  

if __name__ == "__main__":
  main()
  # query_ratio(ratio)
  TF_IDF_comparison()