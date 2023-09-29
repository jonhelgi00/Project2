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


  def IsEmpty(self):
    return len(self.data) == 0
  
  def cluster(self, branch_factor):
    #use KMeans algorithnm on data
    # kmeans = KMeans(n_clusters=branch_factor, random_state=0, n_init="auto")
    self.kmeans = KMeans(n_clusters=branch_factor)
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
    if depth != 0 and not node.IsEmpty():
      node.cluster(self.b)
      for child in node.children:
        self.ConstructVocab(child.data, self.b, depth-1, child)
    else:
      pass

  def No_visualWords(self, node):
    if node.IsLeafNode(self.max_depth):
      node.visual_word_no = self.vw_no
      self.vw_no += 1
    elif not node.IsEmpty():
      for child in node.children:
        self.No_visualWords(child)
    else:
      pass



def Transform_data(filename):
  # Read dictionary pkl file
  # with open('database_ft.pkl', 'rb') as fp:
  with open(filename, 'rb') as fp:
    database_ft = pickle.load(fp)
  
  database_ft_ls = []
  for key in database_ft:
    for i in range(3):
      for v in database_ft[key][i]:
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
    while not vw_found:
      v = np.array(v)
      v.reshape((1,len(v)))
      # v.reshape(1,-1)
      print(v.shape)
      ind = current_node.kmeans.predict(v)
      current_node = current_node.children[ind]
      if current_node.isLeafNode(hk_means_obj.max_depth):
        p_vector[current_node.vw_no] = 1
        vw_found = True
  
  return p_vector

def Read_data_obj(filename):
  # Read dictionary pkl file
  # with open('database_ft.pkl', 'rb') as fp:
  with open(filename, 'rb') as fp:
    database_ft = pickle.load(fp)

  data_dict = {} 
  
  temp_ls = []
  for key in database_ft:
    temp_ls = []
    for i in range(3):
      for v in database_ft[key][i]:
        temp_ls.append(v)
    data_dict[key] = temp_ls
  
  return data_dict



def main():

  database_data = Transform_data('database_ft.pkl')
  # print(database_data.shape )

  branch = 2
  depth = 2
  hk_means_obj = HKMeans(database_data, branch, depth)
  hk_means_obj.ConstructVocab(hk_means_obj.data, branch, depth, hk_means_obj.root)

  hk_means_obj.No_visualWords(hk_means_obj.root)
  # print(hk_means_obj.vw_no)

  database_dict = Read_data_obj('database_ft.pkl')
  # print(len(database_dict[1]))
  p_vector = Binary_presence_vector(database_dict[1], hk_means_obj)



  



main()  
    