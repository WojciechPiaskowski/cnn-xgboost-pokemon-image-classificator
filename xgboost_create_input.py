import keras.models
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
import joblib

# load and initiate train and test data generators
generator = joblib.load('image_generator.pkl')
batch_size = 32

train_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='training',
                                          shuffle=True)
test_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='validation',
                                         shuffle=False)

# load CNN model and create an intermediate model (extract features, but do not use final prediction layers)
cnn = keras.models.load_model('models/cnn')
mid_model = Model(cnn.input, cnn.get_layer('dense').output)
mid_model.summary()

# create a training dataset for xgboost out of imagedategenerator batches and CNN feature extraction
n = 2000
for i in range(n):

    if i == 0:
        x_train_xgb, y_train_xgb = train_gen.next()
        y_train_xgb = np.argmax(y_train_xgb, axis=1)
        x_train_xgb = mid_model.predict(x_train_xgb)

    else:
        x_train_xgb2, y_train_xgb2 = train_gen.next()
        y_train_xgb2 = np.argmax(y_train_xgb2, axis=1)
        x_train_xgb2 = mid_model.predict(x_train_xgb2)

        x_train_xgb = np.concatenate((x_train_xgb, x_train_xgb2), axis=0)
        y_train_xgb = np.concatenate((y_train_xgb, y_train_xgb2), axis=0)

    print(f'{i}/{n}')

test_gen.reset()
for i in range(41):

    if i == 0:
        x_test_xgb, y_test_xgb = test_gen.next()
        y_test_xgb = np.argmax(y_test_xgb, axis=1)
        x_test_xgb = mid_model.predict(x_test_xgb)

    else:
        x_test_xgb2, y_test_xgb2 = test_gen.next()
        y_test_xgb2 = np.argmax(y_test_xgb2, axis=1)
        x_test_xgb2 = mid_model.predict(x_test_xgb2)

        x_test_xgb = np.concatenate((x_test_xgb, x_test_xgb2), axis=0)
        y_test_xgb = np.concatenate((y_test_xgb, y_test_xgb2), axis=0)

    print(f'{i}/41')

le = LabelEncoder()
y_train_xgb = le.fit_transform(y_train_xgb)

# save xgb datasets
joblib.dump(x_train_xgb, 'xgb_data/x_train.pkl', compress=True)
joblib.dump(y_train_xgb, 'xgb_data/y_train.pkl', compress=True)
joblib.dump(x_test_xgb, 'xgb_data/x_test.pkl')
joblib.dump(y_test_xgb, 'xgb_data/y_test.pkl')