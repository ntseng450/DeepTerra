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

batch_size = 16
num_classes = 3
epochs = 120
data_augmentation = True
num_predictions = 3
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_it = datagen.flow_from_directory(
    directory = 'RoverImageSet/train/',
    target_size = (240,240),
    color_mode = 'rgb',
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True,
    seed = 42)

val_it = datagen.flow_from_directory(
    directory = 'RoverImageSet/val/',
    target_size = (240,240),
    color_mode = 'rgb',
    batch_size = 4,
    class_mode = 'categorical',
    shuffle = True,
    seed = 42)

train_steps = train_it.n//train_it.batch_size
val_steps = val_it.n//val_it.batch_size
labels = (train_it.class_indices)
for category_label in labels:
    print(category_label)
input_shape = (240, 240,3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
            verbose = 1)

terminate_on_nan = TerminateOnNaN()

callbacks = [learning_rate_scheduler,
             terminate_on_nan]

model.fit_generator(generator = train_it, steps_per_epoch = train_steps,
                    epochs = epochs, callbacks = callbacks, validation_data = val_it, validation_steps = val_steps)
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