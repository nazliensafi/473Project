import numpy as np
import pandas
import pandas as pd

import data as DATA
import metrics


class KMeans:
    def __init__(self, k):
        self.k = k
        self.history = []
        self.centroid_history = []
        pass

    def _assign_cluster_to_samples(self, data, centroids_df, distance_fx):
        """
        Iterates over every sample and finds closest cluster based on distance function.
        Returns DataFrame with 'cluster id' and 'centroid distance'.
        :param data:
        :param centroids_df:
        :param distance_fx:
        :return:
        """
        self.centroid_history.append(centroids_df)
        clusters_df = pandas.DataFrame()
        for sample_index, sample in data.iterrows():
            centroid_distances = []

            for centroid_index, centroid in centroids_df.iterrows():
                distance = distance_fx(sample, centroid)
                centroid_distances.append(distance)

            min_centroid_distance = min(centroid_distances)
            min_distance_centroid_index = centroid_distances.index(min_centroid_distance)

            centroid_s = pd.Series(name=sample_index)

            centroid_s['cluster id'] = min_distance_centroid_index
            centroid_s['centroid distance'] = min_centroid_distance
            clusters_df = clusters_df.append(centroid_s)

        return clusters_df

    def _distance(self, s1, s2):
        s_abs_diff = abs(s1 - s2)
        s_squared = s_abs_diff ** 2
        sqrt_squared_sum = s_squared.sum() ** (1/2)

        return sqrt_squared_sum

    def _next_centroids(self, current_clusters_df):
        """
        Calculates the average of every feature of every sample in the same cluster.
        Returns new centroid for the cluster.
        :param current_clusters_df:
        :return:
        """
        data = DATA.X_train.merge(current_clusters_df, left_index=True, right_index=True)
        next_centroids = data.groupby(['cluster id']).mean()
        return next_centroids

    def runner(self):
        """
        Initializes centroids to random samples. Assigns a cluster to every sample, based on distance to centroid.
        Calculates new centroid for every cluster, based on samples belonging to that cluster.
        Assigns a new cluster to every sample, based on the updated centroids.
        If all samples remain in the same cluster, terminate.
        Else, recalculate cluster centroids, and repeat.
        :return:
        """
        current_centroids_df = DATA.X_train.sample(self.k)
        current_clusters_df = self._assign_cluster_to_samples(DATA.X_train, current_centroids_df, self._distance)
        while True:
            next_centroids_df = self._next_centroids(current_clusters_df)
            next_clusters_df = self._assign_cluster_to_samples(DATA.X_train, current_centroids_df, self._distance)

            current_clusters_df = next_clusters_df
            current_centroids_df = next_centroids_df

            if next_clusters_df.equals(current_clusters_df):
                break

        return current_clusters_df, current_centroids_df


solver = KMeans(k=2)

r1 = DATA.X_train.iloc[0]
r2 = DATA.X_train.iloc[1]

a, b = solver.runner()

f1score = metrics.f1(DATA.y_train, a['cluster id'])
print(f1score)
print('break')