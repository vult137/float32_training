from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import time


time_start = time.time()

batch_size = 128
epochs = 50
classes = 10
dropout_rate = 0.25
data_augmentation = False
save_dir = os.path.join(os.getcwd(), "..", "saved_models")
save_dir = os.path.abspath(save_dir)
model_name = "regular_128_batch_size.h5"
print(model_name)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="SAME", input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="SAME"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dense(classes))
model.add(Activation("softmax"))

optimizer = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

time_end = time.time()

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

print('time cost for training is', round(time_end - time_start, 2), 's')
