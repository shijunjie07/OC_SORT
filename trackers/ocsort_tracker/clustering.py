# --------------------------------
# cluster detections
# adapted from SportsTrack's original GA Implementation
# 
# @author: Shi Junjie
# Fri 3rd Jan 2025
# --------------------------------

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .track import KalmanBoxTracker

class Clustering:
    """
    Class to perform clustering on detection data using DBSCAN.
    """    
    def __init__(self, eps, min_samples=3):
        """
        Initialize the clustering parameters.

        Args:
            eps (float): Maximum distance between samples to be considered as a cluster.
            min_samples (int, optional): Minimum number of samples to form a cluster.
                                         Defaults to 3.
        """
        self.eps = eps
        self.min_samples = min_samples

    def get_cluster(self, stracks:list[KalmanBoxTracker]):
        """
        Cluster the given stracks data and separate outliers.

        Args:
            stracks (list[KalmanBoxTracker]): frame activated stracks (output_stracks)

        Returns:
            tuple: A dictionary of clustered data and a list of outliers.
        """
        scaler = StandardScaler()

        # X = detections[:, :2]
        X = np.asarray([strack.xtwh[:2] for strack in stracks])

        # normalize the data
        X_scaled = scaler.fit_transform(X)
            
        # perform DBSCAN clustering
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db.fit_predict(X_scaled)
        
        # create a dictionary for clustered data
        # and a list for outliers
        unique_labels = set(labels)
        clustered_data = {}
        outliers = []
        
        for label in unique_labels:
            if label == -1:  # Outliers
                outlier_indices = np.where(labels == label)[0]
                outliers = [stracks[idx] for idx in outlier_indices]
                
                # update STracks
                for idx in outlier_indices:
                    stracks[idx].prev_cluster_num = None

            else:  # Clustered data
                cluster_indices = np.where(labels == label)[0]
                clustered_data[int(label)] = [stracks[idx] for idx in cluster_indices]
                
                # update STracks
                for idx in cluster_indices:
                    stracks[idx].prev_cluster_num = label
                
        return clustered_data, outliers