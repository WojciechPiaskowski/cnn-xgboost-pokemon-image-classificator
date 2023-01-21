import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(rescale=1 / 255, validation_split=0.2, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=32, class_mode='categorical', subset='training')

test_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=32,
                                         class_mode='categorical', subset='validation')

label_dict = train_gen.class_indices
label_dict = {value: key for key, value in label_dict.items()}

# create a sample batch of images and display a random one
sample_x, sample_y = next(train_gen)
# sample_x is 32 x 220 x 220 x 3 -> 32 images, resolution 220x220, 3 colors
# sample_y is 32 x 150 -> 32 labels one-hot-encoded

random_idx = np.random.randint(0, 32, 1, int)[0]
plt.imshow(sample_x[random_idx])
plt.show()
plt.title('true: %s \n predicted: %s' % (label_dict[sample_y[random_idx].argmax()], 'none'))
