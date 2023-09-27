from sklearn.cluster import KMeans
import numpy as np

class Tree:

  def __init__(self, data, depth, center) -> None:
    self.depth = depth #what is the current depth of the node
    self.children = []
    self.center = center
    self.data = data
    self.logK_Ki = None


  def IsEmpty(self):
    return self.children == None
  
  def cluster(self, branch_factor):
    #use KMeans algorithnm on data
    kmeans = KMeans(n_clusters=branch_factor, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(self.data, y=None, sample_weight=None)
    for i in range(branch_factor):
      # idx_i = [j for j,label in enumerate(labels) if label == i]
      data_i = [d for j,d in enumerate(self.data) if labels[j] == i]
      cluster_center_i = kmeans.cluster_centers_[i,:]
      child = Tree(data=data_i, depth=self.depth + 1,center=cluster_center_i)
      self.children.append(child)
    

  def IsLeafNode(self, max_depth):
    return max_depth == self.depth
  
  def SetLogK_Ki(self):
    pass

  def Push(self):
    pass

    
