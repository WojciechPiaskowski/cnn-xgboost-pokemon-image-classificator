from tensorflow import keras
from matplotlib import pyplot as plt

def sample_image(gen, batch_s=32, model=None, label_d=label_dict):
    # create a sample batch of images and display a random one
    sample_x, sample_y = next(gen)
    # sample_x is 32 x 220 x 220 x 3 -> 32 images, resolution 220x220, 3 colors
    # sample_y is 32 x 150 -> 32 labels one-hot-encoded

    random_idx = np.random.randint(0, batch_s, 1, int)[0]

    if model is not None:
        pred = model.predict(sample_x)
        predicted_label = label_d[np.argmax(pred[random_idx])]
        predicted_score = np.round(pred[random_idx].max(), 2)
    else:
        predicted_label = 'None'
        predicted_score = 'None'

    plt.imshow(sample_x[random_idx])
    plt.show()
    plt.title(f'true: {label_d[sample_y[random_idx].argmax()]} \n predicted: {predicted_label} \n score: {predicted_score}')

    return



model = keras.models.load_model('models/cnn')



