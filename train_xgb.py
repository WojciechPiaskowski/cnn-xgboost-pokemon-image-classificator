import keras.models
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from xgboost import XGBClassifier
import joblib

# load in image data generators
generator = joblib.load('other/image_gen.pkl')
batch_size = 32

train_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='training',
                                          shuffle=True)
test_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='validation',
                                         shuffle=False)

# load the CNN model
cnn = keras.models.load_model('models/cnn/cnn.h5')

# extract the CNN layers required for feature extraction before passing it through to XGBoostClassifier
mid_model = Model(cnn.input, cnn.get_layer('dense').output)
mid_model.summary()

# I was unable to feed the image generator data directly to CNN->XGB, so I generate the training data 'manually'
# create a training dataset for xgboost out of imagedategenerator batches and CNN feature extraction
n = 1000
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

# label encoder fixes an input issue with XGB classifier (expects Y to be sorted)
le = LabelEncoder()
y_train_xgb = le.fit_transform(y_train_xgb)

# create the XGB model
xgb = XGBClassifier(objective='multi:softprob', num_class=150, eval_metric=['mlogloss', 'merror'],
                    learning_rate=0.2, max_depth=3, n_estimators=300,
                    min_child_weight=50, reg_lambda=5,
                    early_stopping_rounds=6)

# fit the generated dataset to the XGB model
xgb.fit(x_train_xgb, y_train_xgb,
        eval_set=[(x_test_xgb, y_test_xgb)])

# save the model object
joblib.dump(xgb, 'models/xgb/xgb.pkl')