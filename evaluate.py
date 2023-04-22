# imports
import numpy as np
import joblib
from tensorflow import keras
from keras.models import Model
from sklearn.metrics import accuracy_score, log_loss, f1_score, top_k_accuracy_score
import matplotlib as mpl
from download_cnn import download_cnn

mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.ioff()

# load image data generator
generator = joblib.load('other/image_gen.pkl')
batch_size = 32

train_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='training',
                                          shuffle=True)
test_gen = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='validation',
                                         shuffle=False)
test_gen_random = generator.flow_from_directory('PokemonData/', target_size=(220, 220), batch_size=batch_size, class_mode='categorical', subset='validation',
                                                shuffle=True)

# class labels dictionary
label_dict = train_gen.class_indices
label_dict = {value: key for key, value in label_dict.items()}

# load CNN (if it exists, else download it) and XGBoost models
try:
    cnn = keras.models.load_model('models/cnn/cnn.h5')
except OSError:
    download_cnn()
    cnn = keras.models.load_model('models/cnn/cnn.h5')

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
    fig.figure.show()

    return


sample_image(gen=test_gen_random)


def test_models():
    test_gen.reset()
    y = test_gen.classes
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

    print(f'CNN top 1 accuracy: {np.round(accuracy_score(y, cnn_yhat), 2)}')
    print(f'XGB-CNN top 1 accuracy: {np.round(accuracy_score(y, xgb_yhat), 2)}')
    print('')
    print(f'CNN top 3 accuracy: {np.round(top_k_accuracy_score(y, cnn_yhat_proba, k=3), 2)}')
    print(f'XGB-CNN top 3 accuracy: {np.round(top_k_accuracy_score(y, xgb_yhat_proba, k=3), 2)}')
    print('')
    print(f'CNN top 5 accuracy: {np.round(top_k_accuracy_score(y, cnn_yhat_proba, k=5), 2)}')
    print(f'XGB-CNN top 5 accuracy: {np.round(top_k_accuracy_score(y, xgb_yhat_proba, k=5), 2)}')
    print('')
    print(f'CNN log loss: {np.round(log_loss(y, cnn_yhat_proba), 2)}')
    print(f'XGB-CNN log loss: {np.round(log_loss(y, xgb_yhat_proba), 2)}')
    print('')
    print(f'CNN F1 score: {np.round(f1_score(y, cnn_yhat, average="macro"), 2)}')
    print(f'XGB-CNN F1 score: {np.round(f1_score(y, xgb_yhat, average="macro"), 2)}')

    return


test_models()

# summarize feature map shapes
for i in range(len(cnn.layers)):
    layer = cnn.layers[i]
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)


# create a function that visualizes feature maps
def visualize_feature_map(img_x, label, layer_num, rows=16, cols=16):
    plt.ioff()
    cnn_label_fl = (label_dict[np.argmax(cnn.predict(img_x))] == label_y)
    xgb_label_fl = (label_dict[np.argmax(xgb.predict_proba(mid_model.predict(img_x)))] == label_y)

    if layer_num == 0:
        fig = plt.figure(figsize=(10, 10))
        fig = fig.gca().imshow(img_x[0])
        fig.figure.suptitle(f'true label: {label_y}'
                            f'\n CNN prediction correct: {cnn_label_fl}'
                            f'\n XGB-CNN prediction correct: {xgb_label_fl}')
        plt.axis('off')
        fig.figure.tight_layout()
        fig.figure.savefig(f'other/feature_maps/{label}_layer_0_original')

    else:
        cnn_fmap = Model(inputs=cnn.inputs, outputs=cnn.layers[layer_num].output)
        fmap = cnn_fmap.predict(img_x)

        # plot 32 feature maps
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows + 2))
        fig.suptitle(f"{layer_num} layer's feature maps {label} "
                     f"\n CNN prediction correct: {cnn_label_fl}"
                     f"\n XGB-CNN prediction correct: {xgb_label_fl}")

        maps = fmap[0, :, :].shape[2]

        for featmap in range(maps):
            for row in range(rows):
                for col in range(cols):
                    axes[row, col].imshow(fmap[0, :, :, featmap], cmap='gray')
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
        fig.figure.tight_layout()
        fig.figure.savefig(f'other/feature_maps/{label}_layer_{layer_num}')

    return


# get a random test generator image
idx = np.random.randint(0, batch_size + 1, 1)[0]
img_x, img_y = test_gen_random.next()
img_x = np.expand_dims(img_x[idx], axis=0)

# get the label of that image
img_y = np.argmax(img_y[idx])
label_y = label_dict[img_y]


# visualize original image along with all feature maps from all 4 layers
visualize_feature_map(img_x, label_y, layer_num=0, rows=4, cols=8)
visualize_feature_map(img_x, label_y, layer_num=1, rows=4, cols=8)
visualize_feature_map(img_x, label_y, layer_num=2, rows=8, cols=8)
visualize_feature_map(img_x, label_y, layer_num=3, rows=8, cols=16)
visualize_feature_map(img_x, label_y, layer_num=4, rows=16, cols=16)

# tunowanie xgboosta
# add original image cnn label (prob) and xgb label (prob)
# cleaning
# requirements.txt
# readme.md
