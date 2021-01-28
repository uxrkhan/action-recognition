import os
import csv
import random
import glob
import numpy as np
import operator
import threading
from processor import process_image
from keras.utils import to_categorical
from keras.preprocessing import image


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    def gen(*args, **kwargs):
        return threadsafe_iterator(func(*args, **kwargs))

    return gen


class Data(object):
    def __init__(self, seq_length, image_shape=(224, 224, 3), root_dir='data'):
        self.seq_length = seq_length
        self.root_dir = root_dir
        self.seq_path = os.path.join(root_dir, 'sequences')
        self.image_shape = image_shape
        self.max_frames = 300
        self.csv_data = self.read_csv_data()
        self.classes = self.get_classes()
        self.data = self.get_data()
        self.train_data, self.test_data = self.split_data()

    def read_csv_data(self):
        with open(os.path.join(self.root_dir, 'data_file.csv'), 'r') as csvfile:
            lines = csv.reader(csvfile)
            csv_data = list(lines)
        return csv_data

    def get_classes(self):
        classes = []
        for line in self.csv_data:
            if line[1] not in classes:
                classes.append(line[1])
        return sorted(classes)

    def get_data(self):
        data = []
        for line in self.csv_data:
            n_frames = int(line[3])
            class_name = line[1]
            if self.seq_length <= n_frames <= self.max_frames and class_name in self.classes:
                data.append(line)
        return data

    def split_data(self):
        train, test = [], []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    @threadsafe_generator
    def frame_generator(self, batch_size, generator_type, data_type):
        data = self.train_data if generator_type == 'train' else self.test_data
        print('Creating %s generator with %d samples.' % (generator_type, len(data)))
        while True:
            X, y = [], []
            for _ in range(batch_size):
                sample_name = random.choice(data)
                if data_type is 'images':
                    frames = self.get_frames(sample_name)
                    frames = self.resample_frames(frames, self.seq_length)
                    sequence = [self.get_image_as_npy(img_name, self.image_shape) for img_name in frames]
                else:
                    sequence = self.get_extracted_sequence(data_type, sample_name)
                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")
                X.append(sequence)
                y.append(self.get_class_one_hot(sample_name[1]))
            yield np.array(X), np.array(y)

    def get_frames(self, sample_name):
        file_path = os.path.join(self.root_dir, sample_name[0], sample_name[1])
        file_name = sample_name[2]
        frames = sorted(glob.glob(os.path.join(file_path, file_name + '.jpg')))
        return frames

    @staticmethod
    def resample_frames(frames, size):
        n_frames = len(frames)
        assert n_frames >= size
        step_size = n_frames // size
        resampled_frames = [frames[i] for i in range(0, n_frames, step_size)]
        return resampled_frames

    @staticmethod
    def get_image_as_npy(img_name, target_shape):
        height, width, _ = target_shape
        img = image.load_img(img_name, target_shape=(height, width))
        img_npy = image.img_to_array(img)
        img_npy_n = (img_npy / 255.).astype(np.float32)  # normalise
        return img_npy_n

    def get_extracted_sequence(self, data_type, sample_name):
        file_name = sample_name[2]
        path = os.path.join(self.seq_path, file_name + '-' + str(self.seq_length) + '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_class_one_hot(self, class_name):
        class_index = self.classes.index(class_name)
        one_hot_labels = to_categorical(class_index, len(self.classes))
        assert len(one_hot_labels) == len(self.classes)
        return one_hot_labels
