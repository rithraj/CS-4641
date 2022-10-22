import numpy as np
from kmeans import pairwise_dist

class DBSCAN(object):
    
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
        
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        cluster_idx = [-1] * len(self.dataset)
        cluster_count = 0
        visited = set()

        for i in range(len(self.dataset)):
            if i in visited:
                continue
            else:
                visited.add(i)
                neighbors = self.regionQuery(i)
                if len(neighbors) >= self.minPts:
                    self.expandCluster(i, neighbors, cluster_count, cluster_idx, visited)
                    cluster_count += 1

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() may be helpful here
            A while loop may be better than a for loop
        """
        visitedIndices.add(index)
        cluster_idx[index] = C
        i = 0
        while True:
            if i == len(neighborIndices):
                break

            n = neighborIndices[i]
            if n not in visitedIndices:
                neighbor = np.int64(n)
                visitedIndices.add(neighbor)
                neighbors = self.regionQuery(neighbor)

                if len(neighbors) >= self.minPts:
                    neighborIndices = np.sort(np.unique(np.concatenate((neighborIndices, neighbors))))
                    i = 0
                
                if cluster_idx[neighbor] == -1:
                    cluster_idx[neighbor] = C
            i += 1

        
    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        indices = np.argwhere(pairwise_dist(self.dataset, self.dataset[pointIndex]).flatten() < self.eps).flatten()
        return indices