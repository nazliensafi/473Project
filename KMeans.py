import numpy as np
import pandas
import pandas as pd

import data as DATA
import metrics


class KMeans:
    def __init__(self, data):
        self.data = data
        self.centroid_history = []
        self.centroids = pd.DataFrame()
        pass

    def _assign_cluster_to_samples(self, data, centroids_df, distance_fx, training=False):
        """
        Iterates over every sample and finds closest cluster based on distance function.
        Returns DataFrame with 'cluster id' and 'centroid distance'.
        :param data:
        :param centroids_df:
        :param distance_fx:
        :return:
        """
        if training:
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
        sqrt_squared_sum = s_squared.sum() ** (1 / 2)

        return sqrt_squared_sum

    def _next_centroids(self, current_clusters_df):
        """
        Calculates the average of every feature of every sample in the same cluster.
        Returns new centroid for the cluster.
        :param current_clusters_df:
        :return:
        """
        self.data = self.data.merge(current_clusters_df, left_index=True, right_index=True)
        next_centroids = self.data.groupby(['cluster id']).mean()
        return next_centroids

    def fit(self, initial_centroids=None, k=2):
        """
        Initializes centroids to random samples. Assigns a cluster to every sample, based on distance to centroid.
        Calculates new centroid for every cluster, based on samples belonging to that cluster.
        Assigns a new cluster to every sample, based on the updated centroids.
        If all samples remain in the same cluster, terminate.
        Else, recalculate cluster centroids, and repeat.
        :return:
        """
        current_centroids_df = initial_centroids if initial_centroids is not None else self.data.sample(k)
        current_clusters_df = self._assign_cluster_to_samples(self.data, current_centroids_df, self._distance,
                                                              training=True)
        while True:
            next_centroids_df = self._next_centroids(current_clusters_df)
            next_clusters_df = self._assign_cluster_to_samples(self.data, current_centroids_df, self._distance,
                                                               training=True)

            current_clusters_df = next_clusters_df
            current_centroids_df = next_centroids_df

            if next_clusters_df.equals(current_clusters_df):
                break

        self.centroids = current_centroids_df
        return current_centroids_df

    def invert_y(self, y):
        # When using more than 2 centroids, we need to iterate through all combinations of values
        # i.e. k = 3
        # Let (C-A, 0) denote centroid A given id 0
        # We need to iterate through:
        # (C-A, 0), (C-B, 1), (C-C, 2)
        # (C-A, 0), (C-B, 2), (C-C, 1)
        # (C-A, 1), (C-B, 0), (C-C, 3) ...
        # this has to be done because we are using kmeans clustering as a classifier
        def invert(val):
            return 1 if val == 0 else 0

        inverted_y = y.apply(lambda x: invert(x))
        return inverted_y

    def predict(self, data, distance=None):
        clusters_df = self._assign_cluster_to_samples(data, self.centroids, self._distance,
                                                             training=True)
        y = clusters_df['cluster id']
        return y


if __name__ == '__main__':
    data = DATA.X_train
    model = KMeans(data)
    centroids = model.fit()
    y_train = model.predict(DATA.X_train)
    y_train_inverted = model.invert_y(y_train)

    f1score_train = metrics.f1(DATA.y_train, y_train)
    f1score_train_inverted = metrics.f1(DATA.y_train, y_train_inverted)

    print(f'f1score: {f1score_train}')
    print(f'f1score_inverted: {f1score_train_inverted}')

    y_test = model.predict(DATA.X_test)
    if f1score_train < f1score_train_inverted:
        y_test = model.invert_y(y_test)

    f1score_test = metrics.f1(DATA.y_test, y_test)
    print(f'f1score_test: {f1score_test}')





# f1score = metrics.f1(DATA.y_train, a['cluster id'])
# TODO: Since cluster randomly assigns 0 or 1 value, we cannot use f1score directly.
#  Instead, we need to determine what cluster 0 and cluster 1 are.
#  To do so, we compare to the training data twice, once as cluster 0 = positive, and once as cluster 0 = negative
#  Then we select the combination with a better score, and use that to compare to the test data


# training kmeans is a bit weird because it's unsupervised. We're randomly selecting initial centroids from X_train
# and comparing the results with X_test?

# print(metrics.f1())
