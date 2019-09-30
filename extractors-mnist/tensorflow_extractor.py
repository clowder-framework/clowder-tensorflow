from tensorflow import keras
# import matplotlib.pyplot as plt


def extract(input_file):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Recreate the exact same model, including weights and optimizer.
    saved_model_location = "tensorflow_model_mnist.h5"
    model = keras.models.load_model(saved_model_location)
    model.summary()

    img = keras.preprocessing.image.load_img(path=input_file, color_mode = "grayscale", target_size=(28, 28, 1))
    img = keras.preprocessing.image.img_to_array(img)
    test_img = img.reshape((1, 784))
    img_class = model.predict_classes(test_img)
    prediction = img_class[0]
    classname = img_class[0]
    print("Class: ", classname)

    # img = img.reshape((28, 28))
    # plt.imshow(img)
    # plt.title(classname)
    # plt.show()

    metadata = {
        'label': classes[int(prediction)]
    }
    result = {
        'metadata': metadata
    }
    return result


if __name__ == '__main__':
    extract('/Users/lmarini/data/clowder-demo-files/9xLwvaT.jpg')
    extract("/Users/lmarini/data/clowder-demo-files/IMG_0997.jpg")
