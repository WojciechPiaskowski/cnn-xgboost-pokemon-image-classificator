import time

import keras.models
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss
from keras.models import Model
from xgboost import XGBClassifier
import joblib
import optuna
from optuna.samplers import TPESampler

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
n = 20000
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


# def model_bayesian(trial):
#
#     learning_rate = trial.suggest_float('learning_rate', low=0.1, high=0.45, step=0.05)
#     max_depth = trial.suggest_int('max_depth', low=2, high=10, step=1)
#     n_estimators = trial.suggest_int('n_estimators', low=100, high=1500, step=200)
#     min_child_weight = trial.suggest_int('min_child_weight', low=1, high=301, step=50)
#     # gamma = trial.suggest_float('gamma', low=0, high=10, step=0.1)
#     reg_lambda = trial.suggest_int('reg_lambda', low=1, high=6, step=1)
#     reg_alpha = trial.suggest_int('reg_alpha', low=1, high=6, step=1)
#     subsample = trial.suggest_float('subsample', low=0.6, high=0.9, step=0.1)
#     colsample_bytree = trial.suggest_float('colsample_bytree', low=0.6, high=0.9, step=0.1)
#
#
#     xgb = XGBClassifier(objective='multi:softprob', num_class=150, eval_metric=['mlogloss', 'merror'],
#                         learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
#                         min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
#                         gamma=0.5, subsample=subsample, colsample_bytree=colsample_bytree,
#                         early_stopping_rounds=4)
#
#     # fit the generated dataset to the XGB model
#     xgb.fit(x_train_xgb, y_train_xgb,
#             eval_set=[(x_test_xgb, y_test_xgb)])
#
#     y = test_gen.classes
#     xgb_yhat_proba = xgb.predict_proba(x_test_xgb)
#     logloss = log_loss(y, xgb_yhat_proba)
#
#     return logloss
#
# search = optuna.create_study(sampler=TPESampler(), direction='minimize')
#
# t_start = time.time()
# search.optimize(model_bayesian, n_trials=20)
# t_search = time.time() - t_start
# print('search in minutes: ', t_search / 60)
#
# params_bayesian = [search.best_trial.number, search.best_trial.value, t_search]

xgb = XGBClassifier(objective='multi:softprob', num_class=150, eval_metric=['mlogloss', 'merror'],
                    learning_rate=0.35, max_depth=2, n_estimators=400,
                    min_child_weight=300, reg_lambda=6, reg_alpha=1,
                    gamma=4, subsample=1, colsample_bytree=1,
                    early_stopping_rounds=4,
                    verbosity=1, tree_method='hist')

xgb.fit(x_train_xgb, y_train_xgb, verbose=True,
        eval_set=[(x_train_xgb, y_train_xgb), (x_test_xgb, y_test_xgb)])

# save the model object
joblib.dump(xgb, 'models/xgb/xgb.pkl')
