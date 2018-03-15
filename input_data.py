import os
import nibabel
import numpy


class DataSet:

    def __init__(self, image_dir, label_dir):
        self._images = self._read_dir(image_dir)
        self._labels = self._read_dir(label_dir)
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = len(self._images)

    def _read_dir(self, dir):
        res = []
        files = os.listdir(dir)
        for file in files:
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                res.append(nibabel.load(dir+'/'+file).get_data())
        return res

    def next_batch(self, batch_size):

        if batch_size > self._num_examples:
            print("request batch size exceeds current size, automatically adapt to current size.")
            self._shuffle_data()
            return self._images, self._labels
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            self._shuffle_data()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

    def _shuffle_data(self):
        # Shuffle the data
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = numpy.array(self._images)[perm]
        self._labels = numpy.array(self._labels)[perm]
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
