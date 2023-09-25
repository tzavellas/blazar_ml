import numpy as np
import os
import pandas as pd


class DatasetCreator:

    def __init__(self, features_df, labels_df, feature_labels):
        self.features = features_df
        self.labels = labels_df
        self.feature_labels = feature_labels

    def __call__(self):
        self.dataset, skipped = DatasetCreator.stitch(
            self.features, self.labels, self.feature_labels)
        return self.dataset, skipped

    def write(self, dataset_file):
        out_df = pd.DataFrame(self.dataset).dropna()
        if os.path.exists(dataset_file):
            os.remove(dataset_file)
        out_df.to_csv(dataset_file, header=False, index=False, na_rep='NaN')

    @staticmethod
    def stitch(features_df, labels_df, feature_labels):
        n_features = features_df.loc[:,
                                     feature_labels[0]:feature_labels[1]].shape[1]
        n_labels, count = labels_df.shape
        dataset = np.zeros((count - 1, n_features + n_labels))
        skipped = []
        for idx, row in features_df.iterrows():
            s = 'y_{}'.format(idx)
            x = row[feature_labels[0]:feature_labels[1]]
            if row['success']:
                y = labels_df.loc[:, s].to_numpy()
                dataset[idx, :] = np.append(x.to_numpy(), y)
            else:
                dataset[idx, :] = np.ones(dataset.shape[1]) * np.nan
                skipped.append(s)
        return dataset, skipped
