import random
import warnings
import librosa
import numpy as np


class Transform(object):
    def transform_data(self, data):
        # Mandatory to be defined by subclasses
        raise NotImplementedError("Abstract object")

    def transform_label(self, label):
        # Do nothing, to be changed in subclasses if needed
        return label

    def _apply_transform(self, sample_no_index):
        data, label = sample_no_index
        if type(data) is tuple:  # meaning there is more than one data_input (could be duet, triplet...)
            data = list(data)
            for k in range(len(data)):
                data[k] = self.transform_data(data[k])
            data = tuple(data)
        else:
            data = self.transform_data(data)
        label = self.transform_label(label)
        return data, label

    def __call__(self, sample):
        """Apply the transformation
        Args:
            sample: tuple, a sample defined by a DataLoad class
        Returns:
            tuple
            The transformed tuple
        """
        if type(sample[1]) is int:  # Means there is an index, may be another way to make it cleaner
            sample_data, index = sample
            sample_data = self._apply_transform(sample_data)
            sample = sample_data, index
        else:
            sample = self._apply_transform(sample)
        return sample


class TimeShift(Transform):
    def __init__(self, mean=0, std=90):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        data = sample
        shift = int(np.random.normal(self.mean, self.std))\

        if type(data) is tuple:  # meaning there is more than one data_input (could be duet, triplet...)
            data = list(data)
            for k in range(len(data)):
                data[k] = np.roll(data[k], shift, axis=1)
            data = tuple(data)
        else:
            data = np.roll(data, shift, axis=1)

#         if len(label.shape) == 2:
#             label = np.roll(label, shift, axis=0)  # strong label only

#         sample = (data, label)
        return data