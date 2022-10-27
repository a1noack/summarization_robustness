import os
import datasets
import numpy as np
import pandas as pd


class SummarizationDataset:
    """Three summarization datasets are currently supported: cnn_dailymail, gigaword, and xsum.

    Args:
        name (str): The name of the dataset (cnn_dailymail, gigaword, or xsum) OR the path
            to a csv file where each row of the file has two columns—where the first column of each line is
            the document to be summarized and the second column of each line is the ground truth summary.
        split (str): The split of the dataset. E.g. test, train, val (or validation?).
    """
    def __init__(self, name='gigaword', split='test', cache_dir=''):
        if name in ['cnn_dailymail', 'gigaword', 'xsum']:
            # use HuggingFace datasets package to download dataset
            if name == 'cnn_dailymail':
                # cnn_dailymail dataset needs subset specified. Use 3.0.0.
                dataset = datasets.load_dataset(name, '3.0.0', cache_dir=cache_dir)[split]
            else:
                dataset = datasets.load_dataset(name, cache_dir=cache_dir)[split]

            # convert pyarrow Table to pandas DataFrame
            self.dataset_df = dataset._data.to_pandas()

            # drop cnn_dailymail dataset's extra column for data id
            if name in ['cnn_dailymail', 'xsum']:
                self.dataset_df = self.dataset_df.drop(['id'], axis=1, errors='ignore')
            self.has_ground_truths = True
        else:
            self.dataset_df = pd.read_csv(os.path.join(cache_dir, name), header=None)
            # if the input dataset only has one column, add a second column with cells that contain empty strings
            if self.dataset_df.shape[1] == 1:
                self.has_ground_truths = False
                self.dataset_df[1] = [''] * len(self.dataset_df)
            else:
                self.has_ground_truths = True

        # make sure that there are two columns in dataframe
        assert self.dataset_df.shape[1] == 2

        # make sure every dataframe has the same column names
        self.dataset_df.columns = ['document', 'summary']

        self._i = 0

    def __next__(self):
        if self._i >= len(self.dataset_df):
            raise StopIteration

        document, summary = self.dataset_df.iloc[[self._i]].values[0]
        self._i += 1

        return document, summary

    def __getitem__(self, i):
        if isinstance(i, (int, np.int64, np.int32)):  # allow user to index dataset using numpy int types
            result = self.dataset_df.iloc[[i]].values[0]
        else:
            result = [list(ex) for ex in self.dataset_df.iloc[i].values]

        return result

    def __len__(self):
        return len(self.dataset_df)
