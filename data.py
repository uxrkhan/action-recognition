import os
import csv
import random
import glob
import numpy as np
import operator
import threading
from processor import process_image
from keras.utils import to_categorical


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


class Data():
    def __init__(self, seq_length, image_shape=(224, 224, 3), root_dir='data'):
        self.seq_length = seq_length
        self.root_dir = root_dir
        self.seq_path = os.path.join(root_dir, 'sequences')
        self.image_shape = image_shape
        self.max_frames = 300
        self.csv_data = self.read_csv_data()
        self.classes = self.get_classes()
        self.data = self.get_data()

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

    def split_train_test(self):
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    @threadsafe_generator
    def frame_generator(self, batch_size, generator_type, data_type):
        train, test = self.split_train_test()
        data = train if generator_type == 'train' else test
        print('Creating %s generator with %d samples.' % (generator_type, len(data)))
        while True:
            X, y = [], []
            for _ in range(batch_size):
                sample = random.choice(data)
                if data_type is 'images':
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_frames(frames, self.seq_length)
                    sequence = self.build_image_sequence(frames)
                else:
                    sequence = self.get_extracted_sequence(data_type, sample)
                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")
                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))
            yield np.array(X), np.array(y)
