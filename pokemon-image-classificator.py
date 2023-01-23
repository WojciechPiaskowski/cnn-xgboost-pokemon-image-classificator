import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout, Input, LeakyReLU
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

batch_size = 32

# create a train and test gata generator flowing from PokemonData subfolders
generator = ImageDataGenerator(rescale=1 / 255, validation_split=0.2, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='training')
test_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='validation')

label_dict = train_gen.class_indices
label_dict = {value: key for key, value in label_dict.items()}


def sample_image(gen, batch_s=batch_size, model=None, label_d=label_dict):
    # create a sample batch of images and display a random one
    sample_x, sample_y = next(gen)
    # sample_x is 32 x 220 x 220 x 3 -> 32 images, resolution 220x220, 3 colors
    # sample_y is 32 x 150 -> 32 labels one-hot-encoded

    random_idx = np.random.randint(0, batch_s, 1, int)[0]

    if model is not None:
        pred = model.predict(sample_x)
        predicted_label = label_d[np.argmax(pred[random_idx])]
    else:
        predicted_label = 'None'

    plt.imshow(sample_x[random_idx])
    plt.show()
    plt.title('true: %s \n predicted: %s' % (label_d[sample_y[random_idx].argmax()], predicted_label))

    return


# sample_image(gen=train_gen)

# create a CNN classifier
# add batch normalization, stride, dropout

i = Input(shape=(220, 220, 3))

x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation='relu')(i)
# x = MaxPool2D()(x)
# x = BatchNormalization()(x)

x = Conv2D(filters=64, kernel_size=(3, 3), strides=2,  activation='relu')(x)
# x = MaxPool2D()(x)
# x = BatchNormalization()(x)

x = Conv2D(filters=128, kernel_size=(3, 3), strides=2, activation='relu')(x)
# x = MaxPool2D()(x)
# x = BatchNormalization()(x)

x = Conv2D(filters=256, kernel_size=(3, 3), strides=2, activation='relu')(x)
x = MaxPool2D()(x)
# x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(train_gen.class_indices), activation='softmax')(x)  # 150 classes

cnn = Model(i, x)

# summary
cnn.summary()

es = EarlyStopping(monitor='val_loss', patience=3)

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
r = cnn.fit_generator(generator=train_gen, epochs=50, steps_per_epoch=train_gen.samples // batch_size,
                      shuffle=True, validation_data=test_gen, verbose=1, callbacks=[es])

# model training plots

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()

# test examples (create a function out of sampling above)

sample_image(gen=test_gen, model=cnn)

# test the model

yhat = cnn.predict_generator(test_gen, test_gen.samples // batch_size + 1)
yhat = np.argmax(yhat, axis=1)

# print as a heatmap with labels
print(confusion_matrix(test_gen.classes, yhat))

# something is wrong here TODO
print(classification_report(test_gen.classes, yhat))

with open('report.txt', 'a+') as f:
    f.write(
        f""" \n
model name: {input("model name: ")}
val_loss: {np.round(r.history['val_loss'][-1], 4)}
val_accuracy" {np.round(r.history['val_accuracy'][-1], 4)} """)

# visualize filters and feature maps
