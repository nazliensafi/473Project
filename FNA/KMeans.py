from collections import defaultdict

import numpy as np
import pandas
import pandas as pd

import data as DATA
import metrics


class KMeans:
    def __init__(self):
        self.history = []
        self.centroid_history = defaultdict(list)
        self.run_number = 0
        self.best_centroids = None

    def _assign_cluster_to_samples(self, data, centroids_df, distance_fx, train=False):
        """
        Iterates over every sample and finds closest cluster based on distance function.
        Returns DataFrame with 'cluster id' and 'centroid distance'.
        :param data:
        :param centroids_df:
        :param distance_fx:
        :return:
        """
        if train:
            self.centroid_history[f'run_{self.run_number}'].append(centroids_df)

        clusters_df = pandas.DataFrame()
        for sample_index, sample in data.iterrows():
            centroid_distances = []

            for centroid_index, centroid in centroids_df.iterrows():
                distance = distance_fx(sample, centroid)
                centroid_distances.append(distance)

            min_centroid_distance = min(centroid_distances)
            min_distance_centroid_index = centroid_distances.index(min_centroid_distance)

            centroid_s = pd.Series(name=sample_index, dtype=np.float64)

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

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, n_runs=1, k=2, verbose=False):
        """
        Initializes centroids to random samples. Assigns a cluster to every sample, based on distance to centroid.
        Calculates new centroid for every cluster, based on samples belonging to that cluster.
        Assigns a new cluster to every sample, based on the updated centroids.
        If all samples remain in the same cluster, terminate.
        Else, recalculate cluster centroids, and repeat.
        :return:
        """
        best_clusters = None
        best_centroids = None
        best_score = None

        for i in range(n_runs):
            self.run_number = i + 1

            current_centroids_df = X_train.sample(k)
            current_clusters_df = self._assign_cluster_to_samples(X_train, current_centroids_df, self._distance, train=True)
            if verbose:
                print(f'RUN {i+1}')
                print('initial centroids')
                print(current_centroids_df)

            while True:
                next_centroids_df = self._next_centroids(current_clusters_df)
                next_clusters_df = self._assign_cluster_to_samples(DATA.X_train, current_centroids_df, self._distance, train=True)

                current_clusters_df = next_clusters_df
                current_centroids_df = next_centroids_df

                if next_clusters_df.equals(current_clusters_df):
                    break

            cluster_ids = current_clusters_df['cluster id']
            inverted_cluster_ids = current_clusters_df['cluster id'].replace({0: 1, 1: 0})

            f1score = metrics.f1(y_train, cluster_ids)
            f1score_inverted = metrics.f1(y_train, inverted_cluster_ids)
            current_clusters_df['cluster id'] = cluster_ids if f1score >= f1score_inverted else inverted_cluster_ids
            f1score = f1score if f1score >= f1score_inverted else f1score_inverted

            if verbose:
                print('Final centroids')
                print(current_centroids_df)
                print(f'f1score: {f1score}\n\n')

            if best_score is None or f1score > best_score:
                best_clusters = current_clusters_df
                best_centroids = current_centroids_df
                best_score = f1score

        self.best_centroids = best_centroids

        return best_clusters, best_centroids, best_score

    def predict(self, X):
        clusters_df = self._assign_cluster_to_samples(X, self.best_centroids, self._distance)
        return clusters_df['cluster id']
