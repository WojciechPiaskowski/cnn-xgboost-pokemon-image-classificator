import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Flatten, Dense, Dropout, Input, LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# set the batch size for the generator
batch_size = 32

# create a train and test data generator flowing from PokemonData subfolders
generator = ImageDataGenerator(rescale=1 / 255, validation_split=0.2,
                               width_shift_range=0.12, height_shift_range=0.12, horizontal_flip=True,
                               brightness_range=[0.8, 1.0], zoom_range=[0.97, 1.0], rotation_range=20)

train_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='training',
                                          shuffle=True)
test_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='validation',
                                         shuffle=False)
# label dictionary
label_dict = train_gen.class_indices
label_dict = {value: key for key, value in label_dict.items()}

# create a CNN classifier
i = Input(shape=(220, 220, 3))

x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(i)
# x = MaxPool2D()(x)
# x = BatchNormalization()(x)

x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
# x = MaxPool2D()(x)
# x = BatchNormalization()(x)

x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
# x = MaxPool2D()(x)
# x = BatchNormalization()(x)

x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
# x = MaxPool2D()(x)
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

# early stopping and model checkpoint callbacks
es = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('models/cnn/', save_best_only=True)

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
r = cnn.fit_generator(generator=train_gen, epochs=100, steps_per_epoch=train_gen.samples // batch_size,
                      shuffle=True, validation_data=test_gen, verbose=1, callbacks=[es, checkpoint])

# model training plots
# fix charts

# plt.plot(r.history['loss'], label='loss')
# plt.plot(r.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()
#
# plt.plot(r.history['accuracy'], label='accuracy')
# plt.plot(r.history['val_accuracy'], label='val_accuracy')
# plt.legend()
# plt.show()

joblib.dump(generator, 'other/image_gen.pkl')

