from sklearn.cluster import KMeans
import pickle
import numpy as np

class Tree:

  def __init__(self, data, depth, center=None) -> None:
    self.depth = depth #what is the current depth of the node
    self.children = []
    self.center = center
    self.data = data #(n_samples, n_features)
    self.logK_Ki = None


  def IsEmpty(self):
    return len(self.data) == 0
  
  def cluster(self, branch_factor):
    #use KMeans algorithnm on data
    # kmeans = KMeans(n_clusters=branch_factor, random_state=0, n_init="auto")
    kmeans = KMeans(n_clusters=branch_factor)
    labels = kmeans.fit_predict(self.data, y=None, sample_weight=None)
    for i in range(branch_factor):
      # idx_i = [j for j,label in enumerate(labels) if label == i]
      data_i = [d for j,d in enumerate(self.data) if labels[j] == i]
      cluster_center_i = kmeans.cluster_centers_[i,:]
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
    
  
  def ConstructVocab(self, data, b, depth, node):
    if depth != 0 and not node.IsEmpty():
      node.cluster(self.b)
      for child in node.children:
        self.ConstructVocab(child.data, self.b, depth-1, child)
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


def main():

  database_data = Transform_data('database_ft.pkl')
  # print(database_data.shape )

  branch = 2
  depth = 2
  hk_means_obj = HKMeans(database_data, branch, depth)
  hk_means_obj.ConstructVocab(hk_means_obj.data, branch, depth, hk_means_obj.root)
  
  


main()  
    