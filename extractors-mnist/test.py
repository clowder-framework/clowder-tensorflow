from tensorflow import keras


def main():
    print('running main')
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
               'Ankle boot']
    print('classes are', classes)

    saved_model_location = "/home/tensorflow_model_mnist.h5"
    model = keras.models.load_model(saved_model_location)
    model.summary()

    input_file = '/home/test_image.png'

    img = keras.preprocessing.image.load_img(path=input_file, color_mode="grayscale", target_size=(28, 28, 1))
    img = keras.preprocessing.image.img_to_array(img)
    test_img = img.reshape((1, 784))
    img_class = model.predict_classes(test_img)
    prediction = img_class[0]
    classname = img_class[0]
    print("Class: ", classname)


if __name__ == '__main__':
    main()
