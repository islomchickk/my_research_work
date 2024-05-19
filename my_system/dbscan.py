import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler


class Cluster:
    def __init__(self):
        self.scaler = StandardScaler()
        pass
    
    def preprocessing(self, data):
        data = self.scaler.fit_transform(data)
        return normalize(data)

    def fit(self, train_meta):
        train_meta_normed = self.preprocessing(train_meta)
        clusters = DBSCAN(eps=1, min_samples=20).fit_predict(train_meta_normed)
        train_meta['cluster'] = clusters
        self.mass_0 = train_meta[train_meta['cluster']==0].drop(['cluster', 'best_model'], axis=1).mean()
        self.mass_1 = train_meta[train_meta['cluster']==-1].drop(['cluster', 'best_model'], axis=1).mean()
        return train_meta

    def predict_cluster(self, X_test_meta):
        new_data = self.preprocessing(X_test_meta)
        print(new_data.reshape(1, -1).shape, self.mass_0.values.reshape(1, -1).shape)
        distances_to_mass_0 = euclidean_distances(new_data.reshape(1, -1), self.mass_0.values.reshape(1, -1))
        distances_to_mass_1 = euclidean_distances(new_data.reshape(1, -1), self.mass_1.values.reshape(1, -1))
        return 1 if distances_to_mass_1 < distances_to_mass_0 else 0