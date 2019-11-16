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

<<<<<<< HEAD
batch_size = 12
=======
batch_size = 4
>>>>>>> 516b48804ddb6660224ae323d6bfad4faf0215b6
num_classes = 3
epochs = 120
data_augmentation = True
num_predictions = 3
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


<<<<<<< HEAD
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_it = datagen.flow_from_directory(
    directory = 'RoverImageSet/train/',
    target_size = (160,160),
    color_mode = 'rgb',
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True,
    seed = 42)

val_it = datagen.flow_from_directory(
    directory = 'RoverImageSet/val/',
    target_size = (160,160),
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
input_shape = (160, 160,3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
=======
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
>>>>>>> 516b48804ddb6660224ae323d6bfad4faf0215b6
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

<<<<<<< HEAD
model.add(Conv2D(64, (3, 3)))
=======
model.add(Conv2D(64, (3, 3), padding='same'))
>>>>>>> 516b48804ddb6660224ae323d6bfad4faf0215b6
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
<<<<<<< HEAD
=======
model.add(Dropout(0.25))
>>>>>>> 516b48804ddb6660224ae323d6bfad4faf0215b6

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
<<<<<<< HEAD
model.add(Dense(2))
=======
model.add(Dense(3))
>>>>>>> 516b48804ddb6660224ae323d6bfad4faf0215b6
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
<<<<<<< HEAD
              optimizer='rmsprop',
=======
              optimizer=adam,
>>>>>>> 516b48804ddb6660224ae323d6bfad4faf0215b6
              metrics=['accuracy'])

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
<<<<<<< HEAD
            verbose = 1)
=======
                                                verbose=1)
>>>>>>> 516b48804ddb6660224ae323d6bfad4faf0215b6

terminate_on_nan = TerminateOnNaN()

callbacks = [learning_rate_scheduler,
             terminate_on_nan]

<<<<<<< HEAD
model.fit_generator(generator = train_it, steps_per_epoch = train_steps,
                    epochs = epochs, callbacks = callbacks, validation_data = val_it, validation_steps = val_steps)
=======
model.fit_generator(train_it, steps_per_epoch = 10, epochs = epochs, callbacks = callbacks, validation_data = val_it, validation_steps = 5)
>>>>>>> 516b48804ddb6660224ae323d6bfad4faf0215b6
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