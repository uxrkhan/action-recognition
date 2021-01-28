import os
import time
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.convolutional import (Conv3D, MaxPooling3D)
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import EarlyStopper, CSVLogger, ModelCheckpoint, TensorBoard
from data import Data

use_deprecated = True
seq_length = 40
image_shape = (80, 80, 3)
data_type = 'images'
model_name = 'conv_3d'
load_to_memory = False
batch_size = 32
n_epoch = 100

# FETCH DATASET
data = Data(seq_length, image_shape)
n_classes = len(data.classes)
steps_per_epoch = (len(data.data) * 0.7) // batch_size

# GENERATE DATA
generator = data.frame_generator(batch_size, 'train', data_type)
val_generator = data.frame_generator(batch_size, 'test', data_type)

# MODEL
input_shape = (seq_length, 80, 80, 3)
model = Sequential()
model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
model.add(Conv3D(128, (3, 3, 3), activation='relu'))
model.add(Conv3D(128, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
model.add(Conv3D(256, (3, 3, 3), activation='relu'))
model.add(Conv3D(256, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
loss = 'categorical_cross_entropy'
metrics = ['accuracy', 'top_k_categorical_accuracy']
optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
print(model.summary)

# CONFIGURE LOGGERS
tb_log_dir = os.path.join('data', 'logs', model_name + '-tb')
csv_log_file = os.path.join('data', 'logs', model_name + '-training-' + str(time.time()) + '.log')
save_weights_path = os.path.join('data', 'checkpoints', 'model' + '-' + data_type + '.{epoch:03d}-{val_loss:.3f).hdf5}')

tb = TensorBoard(tb_log_dir)
early_stopper = EarlyStopper(patience=5)
csv_logger = CSVLogger(csv_log_file)
checkpointer = ModelCheckpoint(filepath=save_weights_path, verbose=1, save_best_only=True)

# TRAIN
if use_deprecated:
    model.fit_generator(
        generator=generator,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=40,
        epochs=n_epoch,
        workers=4,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        verbose=1
    )
else:
    model.fit_generator(
        X=generator,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=40,
        epochs=n_epoch,
        workers=4,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        verbose=1
    )
