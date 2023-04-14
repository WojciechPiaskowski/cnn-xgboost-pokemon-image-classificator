# imports
import numpy as np
import joblib
from tensorflow import keras
from keras.models import Model
from sklearn.metrics import accuracy_score, log_loss, f1_score
import matplotlib as mpl

mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

# load image data generator
generator = joblib.load('other/image_gen.pkl')
batch_size = 32

train_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='training',
                                          shuffle=True)
test_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='validation',
                                         shuffle=False)

# class labels dictionary
label_dict = train_gen.class_indices
label_dict = {value: key for key, value in label_dict.items()}

# load CNN and XGBoost models
cnn = keras.models.load_model('models/cnn')
mid_model = Model(cnn.input, cnn.get_layer('dense').output)
xgb = joblib.load('models/xgb/xgb.pkl')


# image sampling function
def sample_image(gen, batch_s=batch_size, label_d=label_dict):
    # create a sample batch of images and display a random one
    sample_x, sample_y = next(gen)
    # sample_x is 32 x 220 x 220 x 3 -> 32 images, resolution 220x220, 3 colors
    # sample_y is 32 x 150 -> 32 labels one-hot-encoded

    random_idx = np.random.randint(0, batch_s, 1, int)[0]

    cnn_pred = cnn.predict(sample_x)
    cnn_predicted_label = label_d[np.argmax(cnn_pred[random_idx])]
    cnn_predicted_score = cnn_pred[random_idx].max()

    xgb_pred = xgb.predict_proba(mid_model.predict(sample_x))
    xgb_predicted_label = label_d[np.argmax(xgb_pred[random_idx])]
    xgb_predicted_score = xgb_pred[random_idx].max()

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    axes[0].title.set_text(
        f'CNN \n true: {label_d[sample_y[random_idx].argmax()]} \n predicted: {cnn_predicted_label} \n probability: {"{:.2f}".format(cnn_predicted_score)}')
    axes[1].title.set_text(
        f'XGBoost-CNN \n true: {label_d[sample_y[random_idx].argmax()]}'
        f' \n predicted: {xgb_predicted_label} \n probability: {"{:.2f}".format(xgb_predicted_score)}')
    axes[0].imshow(sample_x[random_idx])
    axes[1].imshow(sample_x[random_idx])
    fig.tight_layout()
    plt.show()

    return


sample_image(gen=test_gen)


def test_models():

    test_gen.reset()
    cnn_yhat_proba = cnn.predict_generator(test_gen, test_gen.samples // batch_size + 1)
    cnn_yhat = np.argmax(cnn_yhat_proba, axis=1)

    for i in range(41):

        if i == 0:
            x_test_xgb, y_test_xgb = test_gen.next()
            y_test_xgb = np.argmax(y_test_xgb, axis=1)
            x_test_xgb = mid_model.predict(x_test_xgb, verbose=0)

        else:
            x_test_xgb2, y_test_xgb2 = test_gen.next()
            y_test_xgb2 = np.argmax(y_test_xgb2, axis=1)
            x_test_xgb2 = mid_model.predict(x_test_xgb2, verbose=0)

            x_test_xgb = np.concatenate((x_test_xgb, x_test_xgb2), axis=0)
            y_test_xgb = np.concatenate((y_test_xgb, y_test_xgb2), axis=0)


    xgb_yhat_proba = xgb.predict_proba(x_test_xgb)
    xgb_yhat = np.argmax(xgb_yhat_proba, axis=1)

    print(f'CNN accuracy: {np.round(accuracy_score(test_gen.classes, cnn_yhat), 2)}')
    print(f'XGB-CNN accuracy: {np.round(accuracy_score(test_gen.classes, xgb_yhat), 2)}')
    print('')
    print(f'CNN log loss: {np.round(log_loss(test_gen.classes, cnn_yhat_proba), 2)}')
    print(f'XGB-CNN log loss: {np.round(log_loss(test_gen.classes, xgb_yhat_proba), 2)}')
    print('')
    print(f'CNN F1 score: {np.round(f1_score(test_gen.classes, cnn_yhat, average="macro"), 2)}')
    print(f'XGB-CNN F1 score: {np.round(f1_score(test_gen.classes, xgb_yhat, average="macro"), 2)}')

    return


test_models()

# top N predictions correct way of scoring ???
# feature maps