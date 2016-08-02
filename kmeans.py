import numpy as np
import pandas as pd

class Kmeans(object):
    # initialize the K means
    # create k centroids
    def __init__(self,dataset,k=3):
        '''
        set up the clusters and the dataset

        inputs:
            dataset = the data to cluster
            k = the number of centroids or clusters
        '''
        self.data = dataset
        self.k = k
        self.centroids = []
        self.points_in_centroid = []
        for i in range(self.k):
            self.centroids.append(self.create_centroid())
            self.points_in_centroid.append([])

    def create_centroid(self):
        choice = np.random.choice(len(self.data))
        return self.data[choice]

    def assign_points_to_centroid(self):
        '''
        compute distances between each point and a centroid, assign each point to its closest centroid
        '''
        self.points_in_centroid = []
        for i in range(self.k):
            self.points_in_centroid.append([])

        centroid_distance = []
        for centroid in self.centroids:
            dist_squared = np.sum((self.data - centroid)**2, axis=1)
            centroid_distance.append(dist_squared)

        # list of each point and its distances to each centroid
        distances_point_to_each_centroid = []
        for idx_datapoint in range(len(self.data)):
            dist_list = []
            for centroid_dist in centroid_distance:
                dist_list.append(centroid_dist[idx_datapoint])
            distances_point_to_each_centroid.append(dist_list)

        for idx_datapoint in range(len(self.data)):
            index_of_centroid = np.argmin(distances_point_to_each_centroid[idx_datapoint])
            self.assign_point(index_of_centroid,idx_datapoint)

    def assign_point(self,index_of_centroid, index_of_point):
        self.points_in_centroid[index_of_centroid].append(index_of_point)

    def _iteration(self):
        '''
        move the centroid to the mean of the clustered points
        called from run_iterations method

        prints the centroid locations
        '''
        self.assign_points_to_centroid()
        for centroid_location in range(len(self.centroids)):
            self.centroids[centroid_location] = np.mean(self.data[self.points_in_centroid[centroid_location]],axis=0)
        print self.centroids

    def run_iterations(self,num=100):
        '''
        for num iterations, run _iteration method
        '''
        for iteration in range(num):
            self._iteration()

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    data = load_iris()
    km = Kmeans(data['data'],3)
    km.run_iterations(10)