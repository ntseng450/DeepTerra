from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TerminateOnNaN
from keras.optimizers import Adam, SGD
import os

batch_size = 4
num_classes = 3
epochs = 120
data_augmentation = True
num_predictions = 3
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


datagen = ImageDataGenerator( featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False,
    zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0., zoom_range=0., channel_shift_range=0., fill_mode='nearest',
    cval=0., horizontal_flip=True, vertical_flip=False, rescale=None,
    preprocessing_function=None, data_format=None, validation_split=0.2)

train_it = datagen.flow_from_directory('RoverImageSet/train/', target_size = (150,150), class_mode = 'categorical', batch_size = batch_size)
val_it = datagen.flow_from_directory('RoverImageSet/val/', target_size = (150,150), class_mode = 'categorical', batch_size = 2)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape = (150,150, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [learning_rate_scheduler,
             terminate_on_nan]

model.fit_generator(train_it, steps_per_epoch = 10, epochs = epochs, callbacks = callbacks, validation_data = val_it, validation_steps = 5)
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(val_it, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])