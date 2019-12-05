# Build-ins import
import logging

# Homebrew
import tensorflow
from keras.layers import Input, Dense
from keras.models import Model

#######################################################
# TODO: clean code and do proper inheritance #
#######################################################
class auto_encoder_shallow:
    """
    Simple 1-width AE.
    """
    def __init__(self, encoding_dim, input_shape):
        # 1) hyperparams
        self.encoding_dim = encoding_dim
        self.input_shape = input_shape

        # 2) setting autoencoder
        self.input_data = Input(shape=self.input_shape)
        # "encoded" is the encoded representation of the input
        self.encoded = Dense(self.encoding_dim, activation='relu')(self.input_data)
        # "decoded" is the lossy reconstruction of the input
        self.decoded = Dense(self.input_shape[0], activation='sigmoid')(self.encoded)
        self.autoencoder = Model(self.input_data, self.decoded)

    def train(self, training_set, test_set, epochs=50, batch_size=10, shuffle=True):
        # 1) start training and extract encoder/decoder
        self.autoencoder.compile(optimizer="adadelta", loss="mean_squared_error")  # binary_crossentropy
        self.autoencoder.fit(training_set, training_set,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(test_set, test_set))

        # 2) Extracting encoding and decoding
        self.encoder = Model(self.input_data, self.encoded)
        # create a placeholder for an encoded input
        self.encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        self.decoder_layer = self.autoencoder.layers[-1]
        # create the decoder modelx
        self.decoder = Model(self.encoded_input, self.decoder_layer(self.encoded_input))

    def test(self, input):
        """
        Method encoding and decoding original input.
        :return:
        """
        encoded_data = self.encoder.predict(input)
        decoded_data = self.decoder.predict(encoded_data)

        print("encoded_data:")
        print(encoded_data)
        print(type(encoded_data))

        print("decoded_data:")
        print(decoded_data)
        print(type(decoded_data))

        # use Matplotlib (don't ask)
        import matplotlib.pyplot as plt

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_data[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def visualise_results(self):
        """
        Visualises the results of the encoding/decoding mechanics.
        :return:
        """
        # encode and decode some digits
        # note that we take them from the *test* set
        encoded_imgs = encoder.predict(x_test)
        decoded_imgs = decoder.predict(encoded_imgs)
        # use Matplotlib (don't ask)
        import matplotlib.pyplot as plt

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def run(self, training_set, test_set, epochs):
        logging.warning("About to train:")
        self.train(training_set=training_set, test_set=test_set, epochs=epochs)
        logging.warning("About to test.")
        self.test(test_set)

class deep_encoder_simple:
    """
    Simple 1-width AE.
    """
    def __init__(self, encoding_dim, input_shape):
        # 1) hyperparams
        self.encoding_dim = encoding_dim
        self.input_shape = input_shape

        # 2) setting autoencoder
        self.input_data = Input(shape=self.input_shape)
        # "encoded" is the encoded representation of the input
        self.encoded = Dense(self.input_shape[0], activation='relu')(self.input_data)
        layer_2 = Dense(self.input_shape[0])
        # "decoded" is the lossy reconstruction of the input
        self.decoded = Dense(self.input_shape[0], activation='sigmoid')(self.encoded)
        self.autoencoder = Model(self.input_data, self.decoded)

    def train(self, training_set, test_set, epochs=50, batch_size=10, shuffle=True):
        # 1) start training and extract encoder/decoder
        self.autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
        self.autoencoder.fit(training_set, training_set,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(test_set, test_set))

        # 2) Extracting encoding and decoding
        self.encoder = Model(self.input_data, self.encoded)
        # create a placeholder for an encoded input
        self.encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        self.decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(self.encoded_input, self.decoder_layer(self.encoded_input))

    def test(self, input):
        """
        Method encoding and decoding original input.
        :return:
        """
        encoded_data = auto_encoder.encoder.predict(input)
        decoded_data = auto_encoder.decoder.predict(encoded_data)

        print("encoded_data:")
        print(encoded_data)
        print(type(encoded_data))

        print("decoded_data:")
        print(decoded_data)
        print(type(decoded_data))

        # use Matplotlib (don't ask)
        import matplotlib.pyplot as plt

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_data[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def run(self):
        logging.warning("About to train:")
        self.train()
        logging.warning("About to test.")
        self.test()


    def visualise_results(self):
        """
        Visualises the results of the encoding/decoding mechanics.
        :return:
        """
        # encode and decode some digits
        # note that we take them from the *test* set
        encoded_imgs = encoder.predict(x_test)
        decoded_imgs = decoder.predict(encoded_imgs)
        # use Matplotlib (don't ask)
        import matplotlib.pyplot as plt

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

class variational_auto_encoder:
    """
    Takes as input sig transformed data in format *****
    """
    def __init__(self):
        raise NotImplementedError

# testing unit: testing auto-encoder1 with mnist
if __name__ == "__main__":
    from keras.datasets import mnist
    import numpy as np

    # Get MNIST data
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    shape_single_data_point = x_train[1].shape



    print(x_train.shape)
    print(type(x_train))
    exit()

    # Instancify obj, train and show results
    auto_encoder = auto_encoder_simple(encoding_dim=15,
                                       input_shape=shape_single_data_point)
    auto_encoder = auto_encoder_simple(encoding_dim=15, input_shape=shape_single_data_point)
    auto_encoder.train(training_set=x_train, test_set=x_test, epochs=1)

    # Visualise end
    auto_encoder.test(x_test)



