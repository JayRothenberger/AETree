import sklearn.preprocessing
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from time import time
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re
import pickle as pk

batch_size = 32
input_shape = (28, 28, 1)  # mnist image shape
loaded = False  # this variable stores whether or not the mnist dataset has been loaded
(x_train, y_train), (x_test, y_test) = (None, None), (None, None)  # dataset is loaded into these globals
(x_train_pure, y_train_pure) = (None, None)  # to store pure training data if needed
threshold = None

results_dir = 'results_0'  # directory to store results pickle and text files


# function that calculates the mean squared error for a sample
def mse(a, b):
    return np.mean(np.square(np.subtract(a, b)), axis=(1, 2))


class Sampling(layers.Layer):  # sampling layer for keras
    def call(self, inputs):
        mean, log_var = inputs
        return tf.keras.backend.random_normal(tf.shape(log_var)) * tf.keras.backend.exp(log_var / 2) + mean


def build_AE(input_shape, activation='selu', encoder_filter_size=(4, 4), init=tf.keras.initializers.LecunNormal(),
             learning_rate=0.001):
    """
    this function builds the autoencoder model
    :param input_shape: shape of the input images (tuple of ints)
    :param activation: activation function for all of the convolutional layers
    :param encoder_filter_size: filter size to use for all of the convolutional layers
    :param init: kernel initializer
    :param learning_rate: learning rate to pass to the optimizer
    :return: returns a triple of model, encoder, decoder where the model is decoder(encoder)
    """
    # model architecture (encoder)
    enc_inputs = layers.Input(input_shape, name='enc_in')
    # 28x28
    x = layers.Conv2D(1, encoder_filter_size, padding='same', activation=activation, kernel_initializer=init)(enc_inputs)
    x = layers.SeparableConv2D(49, encoder_filter_size, (2, 2), padding='same',
                               kernel_initializer=init, activation=activation)(x)
    # 14x14
    x = layers.SeparableConv2D(49, encoder_filter_size, (2, 2), padding='same',
                               kernel_initializer=init, activation=activation)(x)
    # 7x7
    x = layers.Conv2D(49, encoder_filter_size, (2, 2), padding='same',
                               kernel_initializer=init, activation=activation)(x)
    x = layers.GlobalAveragePooling2D()(x)
    # 49 output features
    enc_outputs = x

    # decoder input size should be the same as the number of filters in the last convolutional layer of the encoder
    # decoder architecture
    dec_inputs = layers.Input((49,), name='dec_in')
    # reshape to upscale
    x = layers.Reshape((7, 7, 1))(dec_inputs)
    # 7x7
    x = layers.SeparableConv2D(49, encoder_filter_size, padding='same',
                               kernel_initializer=init, activation=activation)(x)
    x = layers.UpSampling2D(2)(x)
    # 14x14
    x = layers.SeparableConv2D(49, encoder_filter_size, padding='same',
                               kernel_initializer=init, activation=activation)(x)
    x = layers.UpSampling2D(2)(x)
    # 28x28
    x = layers.SeparableConv2D(49, encoder_filter_size, padding='same',
                               kernel_initializer=init, activation=activation)(x)

    x = layers.Conv2D(1, (3, 3), padding='same', activation=activation)(x)

    dec_outputs = x
    # create the models from the layer sequences
    encoder = tf.keras.Model(inputs=[enc_inputs], outputs=[enc_outputs], name='mnist_enc')

    decoder = tf.keras.Model(inputs=[dec_inputs], outputs=[dec_outputs], name='mnist_dec')

    model = tf.keras.Model(inputs=[enc_inputs], outputs=decoder(encoder(enc_inputs)), name='mnist_ae')

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # compile the autoencoder to be trained
    model.compile(loss='mse', optimizer=opt)
    # print the summary
    print(model.summary())
    # return the models
    return model, encoder, decoder


def build_variational_AE(input_shape, activation='selu', encoder_filter_size=(4, 4), init=tf.keras.initializers.LecunNormal(),
             learning_rate=0.001):
    """
    this function builds the autoencoder model
    :param input_shape: shape of the input images (tuple of ints)
    :param activation: activation function for all of the convolutional layers
    :param encoder_filter_size: filter size to use for all of the convolutional layers
    :param init: kernel initializer
    :param learning_rate: learning rate to pass to the optimizer
    :return: returns a triple of model, encoder, decoder where the model is decoder(encoder)
    """
    # model architecture (encoder)
    enc_inputs = layers.Input(input_shape, name='enc_in')
    # 28x28
    x = layers.Conv2D(1, encoder_filter_size, padding='same', activation=activation, kernel_initializer=init)(enc_inputs)

    x = layers.SeparableConv2D(49, encoder_filter_size, (2, 2), padding='same',
                               kernel_initializer=init, activation=activation,
                               use_bias=True, bias_initializer='zeros')(x)
    # 14x14
    x = layers.SeparableConv2D(49, encoder_filter_size, (2, 2), padding='same',
                               kernel_initializer=init, activation=activation,
                               use_bias=True, bias_initializer='zeros')(x)
    # 7x7
    x = layers.Conv2D(49, encoder_filter_size, (2, 2), padding='same',
                               kernel_initializer=init, activation=activation,
                               use_bias=True, bias_initializer='zeros')(x)
    x = layers.GlobalAveragePooling2D()(x)

    m = layers.Dense(49, activation=activation, use_bias=True, bias_initializer='zeros', kernel_initializer=init)(x)
    g = layers.Dense(49, activation=activation, use_bias=True, bias_initializer='zeros', kernel_initializer=init)(x)

    # 49 output features
    enc_outputs = Sampling()([m, g])

    # decoder input size should be the same as the number of filters in the last convolutional layer of the encoder
    # decoder architecture
    dec_inputs = layers.Input((49,), name='dec_in')
    # reshape to upscale
    x = layers.Reshape((7, 7, 1))(dec_inputs)
    # 7x7
    x = layers.SeparableConv2D(49, encoder_filter_size, padding='same',
                               kernel_initializer=init, activation=activation)(x)
    x = layers.UpSampling2D(2)(x)
    # 14x14
    x = layers.SeparableConv2D(49, encoder_filter_size, padding='same',
                               kernel_initializer=init, activation=activation)(x)
    x = layers.UpSampling2D(2)(x)
    # 28x28
    x = layers.SeparableConv2D(49, encoder_filter_size, padding='same',
                               kernel_initializer=init, activation=activation)(x)

    x = layers.Conv2D(1, (3, 3), padding='same', activation=activation)(x)

    dec_outputs = x
    # create the models from the layer sequences
    encoder = tf.keras.Model(inputs=[enc_inputs], outputs=[enc_outputs], name='mnist_enc_var')

    decoder = tf.keras.Model(inputs=[dec_inputs], outputs=[dec_outputs], name='mnist_dec_var')

    model = tf.keras.Model(inputs=[enc_inputs], outputs=decoder(encoder(enc_inputs)), name='mnist_ae_var')

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # compile the autoencoder to be trained
    model.compile(loss='mse', optimizer=opt)
    # print the summary
    print(model.summary())
    # return the models
    return model, encoder, decoder


def train_model(models, args, epochs):
    # different training types
    # with noise, noise with int cast, noisy denoising
    model, encoder, decoder = models
    x_train_noisy = np.clip(np.add(x_train, np.random.normal(mu, sigma, x_train.shape)), 0, 255)
    if args.wnoise:
        X = np.concatenate((x_train, x_train_noisy), axis=0)
        model.fit(X, X, epochs=epochs, batch_size=batch_size)

    elif args.denoise:
        X = x_train_noisy
        Y = x_train
        model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    else:
        model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

    return model, encoder, decoder


def load_recent_model(dirname, filebase):
    """
    this function loads a model for the directory <filebase>_<float> where float is the largest number in dir <dirname>

    :param dirname: directory to load from
    :param filebase: characters in the folder name predeeding the _
    :return: the loaded keras model (or none if cannot find any)
    """
    try:
        files = [f for f in os.listdir(dirname) if re.match(r'%s' % filebase, f)]
        files = sorted(files, key=lambda x: float(x.split('_')[-1]))

        return tf.keras.models.load_model(files[-1])  # return file with latest timestamp

    except IndexError as e:
        print(f'cannot find: {filebase} in {dirname}')
        return None


def create_parser():
    """
    this function creates the argument parser

    :return: returns the argument parser object
    """
    parser = argparse.ArgumentParser(description='MNIST experiment')
    parser.add_argument('--load', action='store_true', help='load a model rather than creating a new one')
    parser.add_argument('--var', action='store_true', help='generate a variational autoencoder')
    parser.add_argument('--tree', action='store_true', help='build a tree and classify on encoded representation')
    parser.add_argument('--forest', action='store_true', help='build a random fores and classify on encoded representation')
    parser.add_argument('--ntrees', type=int, default=250, help='number of trees in random forest')
    parser.add_argument('--maxdepth', type=int, default=9, help='max depth of random forest trees')
    parser.add_argument('--plot', action='store_true', help='show a plot of some test images passed through AE')
    parser.add_argument('--maps', action='store_true', help='generate feature map visualizations for cnn')
    parser.add_argument('--nn', action='store_true', help='train a simple neural net on the encoded input')
    parser.add_argument('--newnn', action='store_true', help='train a new simple neural net on the encoded input')
    parser.add_argument('--noise', action='store_true', help='evaluate trained models on noisy test data')
    parser.add_argument('--mu', type=float, default=0, help='mean for gaussian noise')
    parser.add_argument('--sigma', type=float, default=10, help='standard dev for gaussian noise')
    parser.add_argument('--wnoise', action='store_true', help='train the AE with noisy data as well')
    parser.add_argument('--denoise', action='store_true', help='train the AE to denoise the input data')
    parser.add_argument('--pure', action='store_false', help='do not train the models on pure testing sample')

    return parser


def arg_string(args):
    """
    this function converts the command line arguments to a string for file saving

    :param args: command line arguments from the parser
    :return: a string that corresponds to the command line argument parameters
    """
    return f'var_{args.var}_mu_{args.mu}_sigma_{args.sigma}_ntrees_{args.ntrees}_maxdepth_{args.maxdepth}_pure_{args.pure}'


def save_results(args, results, type):
    """

    :param args:
    :param results:
    :param type:
    :return:
    """
    pk.dump(f'{type}_{arg_string(args)}', results)


def save_results_text(args, results, type):
    """

    :param args:
    :param results:
    :param type:
    :return:
    """
    with open(f'{results_dir}/{type}_{arg_string(args)}.txt', 'a') as rfile:
        rfile.write(f'baseline     - train, test: {t_train_score}, {t_test_score}\n'
                    f'noisy        - train, test: {t_train_score_noisy}, {t_test_score_noisy}\n'
                    f'noisy int    - train, test: {t_train_score_noisy_int}, {t_test_score_noisy_int}\n'
                    f'noisy reject - noise, int : {t_test_score_noisy_reject}, {t_test_score_noisy_reject_int}\n'
                    f'rejected samples: {noisy_rejected}, {noisy_rejected_int}\n')


if __name__ == '__main__':
    # retrieve the command line arguments for the program
    parser = create_parser()
    args = parser.parse_args()

    # these variables will store the loaded autoencoder
    model, encoder, decoder = None, None, None
    threshold = None

    # these variables will store the optional decision tree, neural network, and random forest classifiers
    t, nn, rf = None, None, None

    # if we are supposed to load the models instead of fitting them anew attempt to do so
    if args.load:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True
        if args.var:  # load variational models
            model, encoder, decoder = load_recent_model('.', 'varautoenc'), load_recent_model('.', 'varenc'), load_recent_model('.', 'vardec')
        else:  # load deterministic models
            model, encoder, decoder = load_recent_model('.', 'autoenc'), load_recent_model('.', 'enc'), load_recent_model('.', 'dec')

    # if any of the objects cannot be loaded, then we have to create and fit them
    if model is None or encoder is None or decoder is None:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True
        print(f'one of model ({str(model)}), encoder({str(encoder)}, decoder({str(decoder)} is None')
        if args.var:  # train a variational model
            model, encoder, decoder= train_model(build_variational_AE(input_shape), args, epochs=50)
            # make sure we save the models so we can load them next time
            tf.keras.models.save_model(model, f'varautoenc_{time()}')
            tf.keras.models.save_model(encoder, f'varenc_{time()}')
            tf.keras.models.save_model(decoder, f'vardec_{time()}')
        else:  # train a determinisitic model
            model, encoder, decoder = train_model(build_AE(input_shape), args, epochs=50)
            # make sure we save the models so we can load them next time
            tf.keras.models.save_model(model, f'autoenc_{time()}')
            tf.keras.models.save_model(encoder, f'enc_{time()}')
            tf.keras.models.save_model(decoder, f'dec_{time()}')
    # set the rejection threshold for out of domain examples
    reconstruction = mse(np.squeeze(model.predict(x_train), axis=-1), x_train.astype(float))
    threshold = np.sort(reconstruction)[int(len(reconstruction) * .95)]

    if args.plot:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True
        # indices to load pictures for the plot
        inds = [0, 1, 2, 3, 4, 5, 6]
        ncols = len(inds)
        nrows = 2
        # create the figure
        fig = plt.figure(figsize=(ncols, nrows), dpi=300)
        # populate it with the images and their reconstructions
        for i in inds:
            # first row
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            encoded = encoder.predict(np.array([x_test[inds[i]], ]))
            decoded = decoder.predict(encoded)
            plt.imshow(array_to_img(decoded[0]))
            # second row
            ax = fig.add_subplot(nrows, ncols, i + 1 + ncols)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            plt.imshow(array_to_img(np.expand_dims(x_test[inds[i]], axis=-1)))

        plt.show()
    if args.pure:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True
        reconstruction = mse(np.squeeze(model.predict(x_train), axis=-1), x_train.astype(float))
        # training examples that do not exceed threshold
        x_train_pure, y_train_pure = x_train[reconstruction <= threshold], y_train[reconstruction <= threshold]

    # if we are to train a decision tree
    if args.tree:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True
        # transform the inputs to the correct dimension
        if args.pure:
            x_train_transformed = encoder.predict(x_train_pure)
        else:
            x_train_transformed = encoder.predict(x_train)
        # fit a sklearn decision tree on the training data, and store the tree in the outer scope name
        t = tree.DecisionTreeClassifier(max_depth=16)
        if args.pure:
            t.fit(x_train_transformed, y_train_pure)
        else:
            t.fit(x_train_transformed, y_train)
    # if we are to train a forest
    if args.forest:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True
        # transform the inputs to the correct dimension
        if args.pure:
            x_train_transformed = encoder.predict(x_train_pure)
        else:
            x_train_transformed = encoder.predict(x_train)
        # fit the model and store it in the outer scope name
        rf = RandomForestClassifier(args.ntrees, max_depth=args.maxdepth)
        if args.pure:
            rf.fit(x_train_transformed, y_train_pure)
        else:
            rf.fit(x_train_transformed, y_train)
    # if we are to evaluate with a neural network
    if args.nn:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True
        # if we are not supposed to train a new neural network
        if not args.newnn:
            if args.var:  # if we are using a variational ae, we have to load the variational version
                nn = load_recent_model(".", "varnn")  # try to load one from the directory
            else:  # if we are using a deterministic ae, we have to load the deterministic version
                nn = load_recent_model(".", "nn")  # try to load one from the directory
        # if one is not loaded
        if nn is None:
            # define a one hot encoder for our categorical outputs
            OHE = sklearn.preprocessing.OneHotEncoder()
            # fit it on all of the labels
            OHE.fit(np.concatenate((y_train, y_test)).reshape(-1, 1))
            # transform all of the training and testing data
            if args.pure:
                x_train_transformed = encoder.predict(x_train_pure)
                y_train_transformed = OHE.transform(y_train_pure.reshape(-1, 1)).toarray()
            else:
                x_train_transformed = encoder.predict(x_train)
                y_train_transformed = OHE.transform(y_train.reshape(-1, 1)).toarray()

            # basic neural network architecture
            input = layers.Input(x_train_transformed.shape[1])

            x = layers.Dense(49, kernel_initializer=tf.keras.initializers.LecunNormal(), activation='selu', use_bias=True,
                             bias_initializer='zeros')(input)

            output = layers.Dense(10, kernel_initializer=tf.keras.initializers.LecunNormal(),
                                  activation=tf.keras.activations.softmax)(x)

            opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

            nn = tf.keras.models.Model(inputs=[input], outputs=[output])
            # compile the model
            nn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
            # print the summary
            print(nn.summary())
            # fit the model
            nn.fit(x_train_transformed, y_train_transformed, epochs=10, batch_size=batch_size)
            # save the model so we can load it in the future
            if args.var:  # if we are using a variational ae, save it as variational nn
                tf.keras.models.save_model(nn, f'varnn_{time()}')
            else:  # if we are using a deterministic ae, save it as the deterministic version
                tf.keras.models.save_model(nn, f'nn_{time()}')
    # if we are to display the activation maps for the convolutional layers of the encoder
    if args.maps:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True
        # collect the layer names and the layer outputs for layers that have activation maps
        # (not input layer, not 1 filter layer)
        layer_names = [layer.name for layer in encoder.layers if len(layer.output.shape) == 4][2:]
        layer_outputs = [layer.output for layer in encoder.layers if len(layer.output.shape) == 4][2:]
        # create the model that has outputs all of the activations
        feature_map_model = tf.keras.models.Model(inputs=[encoder.input], outputs=layer_outputs)
        feature_map_model.compile()
        # calculate the average activation across all testing samples
        feature_maps = [np.average(feature_map, axis=0) for feature_map in feature_map_model.predict(x_test)]
        # define the size of the figure to store the images of the average activations
        ncols = 49
        nrows = len(layer_names)
        # create the figure
        fig = plt.figure(figsize=(ncols, nrows))
        row = 0
        # populate the figure with the activation layers
        for layer_name, feature_map in zip(layer_names, feature_maps):
            k = feature_map.shape[-1]
            size = feature_map.shape[1]
            for i in range(k):
                # iterating over a feature map of a particular layer to separate all feature images.
                feature_image = feature_map[:, :, i]
                # normalizing the feature maps to make them visible
                feature_image -= feature_image.mean()
                feature_image /= feature_image.std()
                # transform it to the pixel range 0-255
                feature_image *= 64
                feature_image += 128
                # clip it to the correct range
                feature_image = np.clip(feature_image, 0, 255).astype('uint8')
                # add the figure to the plot
                ax = fig.add_subplot(nrows, ncols, row * ncols + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(feature_image, axis=-1)))

            row += 1
        # display the plot
        plt.show()
    # if we are to perform the noist experiment
    if args.noise:
        if loaded:
            OHE = sklearn.preprocessing.OneHotEncoder()
            # fit the one-hot-encoder
            OHE.fit(np.concatenate((y_train, y_test)).reshape(-1, 1))
            # mean and standard deviation for the gaussian noise
            mu, sigma = args.mu, args.sigma
            # add the noise to the testing and training data
            x_train_noisy = np.clip(np.add(x_train, np.random.normal(mu, sigma, x_train.shape)), 0, 255)
            x_test_noisy = np.clip(np.add(x_test, np.random.normal(mu, sigma, x_test.shape)), 0, 255)
            # perform the integer cast for the integer cast experiment
            x_train_noisy_int = x_train_noisy.astype(int)
            x_test_noisy_int = x_test_noisy.astype(int)
            # predict with the encoder to transform the original examples to the latent space
            x_train_transformed = encoder.predict(x_train)
            x_test_transformed = encoder.predict(x_test)
            # one-hot-encode the labels
            y_train_transformed = OHE.transform(y_train.reshape(-1, 1)).toarray()
            y_test_transformed = OHE.transform(y_test.reshape(-1, 1)).toarray()
            # transform the noisy samples to the latent space
            x_train_transformed_noisy = encoder.predict(x_train_noisy)
            x_test_transformed_noisy = encoder.predict(x_test_noisy)
            x_train_transformed_noisy_int = encoder.predict(x_train_noisy_int)
            x_test_transformed_noisy_int = encoder.predict(x_test_noisy_int)

            # calculate the reconstruction error
            noisy_reconstruction = mse(np.squeeze(model.predict(x_train_noisy), axis=-1), x_train.astype(float))
            noisy_reconstruction_int = mse(np.squeeze(model.predict(x_train_noisy_int), axis=-1), x_train.astype(float))
            reconstruction = mse(np.squeeze(model.predict(x_train), axis=-1), x_train.astype(float))
            # calculate the reconstruction error for the test samples
            noisy_reconstruction_test = mse(np.squeeze(model.predict(x_test_noisy), axis=-1), x_test.astype(float))
            noisy_reconstruction_test_int = mse(np.squeeze(model.predict(x_test_noisy_int), axis=-1), x_test.astype(float))
            # determine which samples will be kept
            x_test_non_reject = x_test_transformed_noisy[noisy_reconstruction_test <= threshold]
            y_test_non_reject = y_test[noisy_reconstruction_test <= threshold]
            y_test_transformed_non_reject = y_test_transformed[noisy_reconstruction_test <= threshold]
            # calculate how many samples were rejected
            noisy_rejected = len(x_test_transformed_noisy[noisy_reconstruction_test > threshold])
            # determine which samples from the integer cast experiment will be kept
            x_test_non_reject_int = x_test_transformed_noisy_int[noisy_reconstruction_test_int <= threshold]
            y_test_non_reject_int = y_test[noisy_reconstruction_test_int <= threshold]
            y_test_transformed_non_reject_int = y_test_transformed[noisy_reconstruction_test_int <= threshold]
            # calculate how many samples were rejected
            noisy_rejected_int = len(x_test_transformed_noisy_int[noisy_reconstruction_test_int > threshold])
            # if we are to evaluate the decision tree
            if t is not None:
                # calculate the accuracy for the decision tree on the training and testing samples
                t_train_score = t.score(x_train_transformed, y_train)
                t_test_score = t.score(x_test_transformed, y_test)
                # and on the noisy samples
                t_train_score_noisy = t.score(x_train_transformed_noisy, y_train)
                t_train_score_noisy_int = t.score(x_train_transformed_noisy_int, y_train)
                # and on the noisy samples with the integer cast
                t_test_score_noisy = t.score(x_test_transformed_noisy, y_test)
                t_test_score_noisy_int = t.score(x_test_transformed_noisy_int, y_test)
                # and on the noisy experiments with the reject option
                t_test_score_noisy_reject = t.score(x_test_non_reject, y_test_non_reject)
                t_test_score_noisy_reject_int = t.score(x_test_non_reject_int, y_test_non_reject_int)
                # store the results in a file
                with open(f'{results_dir}/t_scores_{arg_string(args)}.txt', 'a') as rfile:
                    rfile.write(f'baseline     - train, test: {t_train_score}, {t_test_score}\n'
                                f'noisy        - train, test: {t_train_score_noisy}, {t_test_score_noisy}\n'
                                f'noisy int    - train, test: {t_train_score_noisy_int}, {t_test_score_noisy_int}\n'
                                f'noisy reject - noise, int : {t_test_score_noisy_reject}, {t_test_score_noisy_reject_int}\n'
                                f'rejected samples: {noisy_rejected}, {noisy_rejected_int}\n')
            # if we are to evaluate the neural network
            if nn is not None:
                # calculate the accuracy for the network on the training and testing samples
                nn_train_score = nn.evaluate(x_train_transformed, y_train_transformed)[-1]
                nn_test_score = nn.evaluate(x_test_transformed, y_test_transformed)[-1]
                # and on the noisy samples
                nn_train_score_noisy = nn.evaluate(x_train_transformed_noisy, y_train_transformed)[-1]
                nn_train_score_noisy_int = nn.evaluate(x_train_transformed_noisy_int, y_train_transformed)[-1]
                # and on the noisy samples with the integer cast
                nn_test_score_noisy = nn.evaluate(x_test_transformed_noisy, y_test_transformed)[-1]
                nn_test_score_noisy_int = nn.evaluate(x_test_transformed_noisy_int, y_test_transformed)[-1]
                # and on the noisy experiments with the reject option
                nn_test_score_noisy_reject = nn.evaluate(x_test_non_reject, y_test_transformed_non_reject)[-1]
                nn_test_score_noisy_reject_int = nn.evaluate(x_test_non_reject_int, y_test_transformed_non_reject_int)[-1]
                # store the results in a file
                with open(f'{results_dir}/nn_scores_{arg_string(args)}.txt', 'a') as rfile:
                    rfile.write(f'baseline     - train, test: {nn_train_score}, {nn_test_score}\n'
                                f'noisy        - train, test: {nn_train_score_noisy}, {nn_test_score_noisy}\n'
                                f'noisy int    - train, test: {nn_train_score_noisy_int}, {nn_test_score_noisy_int}\n'
                                f'noisy reject - noise, int : {nn_test_score_noisy_reject}, {nn_test_score_noisy_reject_int}\n'
                                f'rejected samples: {noisy_rejected}, {noisy_rejected_int}\n')
            # if we are to evaluate the random forest
            if rf is not None:
                # calculate the accuracy for the network on the training and testing samples
                rf_train_score = rf.score(x_train_transformed, y_train)
                rf_test_score = rf.score(x_test_transformed, y_test)
                # and on the noisy samples
                rf_train_score_noisy = rf.score(x_train_transformed_noisy, y_train)
                rf_train_score_noisy_int = rf.score(x_train_transformed_noisy_int, y_train)
                # and on the noisy samples with the integer cast
                rf_test_score_noisy = rf.score(x_test_transformed_noisy, y_test)
                rf_test_score_noisy_int = rf.score(x_test_transformed_noisy_int, y_test)
                # and on the noisy experiments with the reject option
                rf_test_score_noisy_reject = rf.score(x_test_non_reject, y_test_non_reject)
                rf_test_score_noisy_reject_int = rf.score(x_test_non_reject_int, y_test_non_reject_int)
                # store the results in a file
                with open(f'{results_dir}/rf_scores_{arg_string(args)}.txt', 'a') as rfile:
                    rfile.write(f'baseline     - train, test: {rf_train_score}, {rf_test_score}\n'
                                f'noisy        - train, test: {rf_train_score_noisy}, {rf_test_score_noisy}\n'
                                f'noisy int    - train, test: {rf_train_score_noisy_int}, {rf_test_score_noisy_int}\n'
                                f'noisy reject - noise, int : {rf_test_score_noisy_reject}, {rf_test_score_noisy_reject_int}\n'
                                f'rejected samples: {noisy_rejected}, {noisy_rejected_int}\n')
            # number of examples to display in each row of the figure that shows best/worst reconstructions
            display_samples = 5
            nrows = 8  # one row for each kind of reconstruction to show
            ncols = display_samples
            # collect the samples to display
            # all rejected samples
            rejected_samples = x_test_noisy[noisy_reconstruction_test > threshold]
            rejected_samples_int = x_test_noisy_int[noisy_reconstruction_test_int > threshold]
            # all reconstruction errors for rejected examples
            reconstruction_err_reject = noisy_reconstruction_test[noisy_reconstruction_test > threshold]
            reconstruction_err_reject_int = noisy_reconstruction_test[noisy_reconstruction_test_int > threshold]
            # original images of the rejected samples
            rejected_original = x_test[noisy_reconstruction_test > threshold]
            rejected_original_int = x_test[noisy_reconstruction_test_int > threshold]
            # indicies of the images to display (worst offenders and just-made-its)
            min_indicies_reject = np.argsort(reconstruction_err_reject)[:display_samples]
            max_indicies_reject = np.argsort(reconstruction_err_reject)[-display_samples:]
            # samples that were kept
            kept_samples = x_test_noisy[noisy_reconstruction_test <= threshold]
            kept_samples_int = x_test_noisy_int[noisy_reconstruction_test_int <= threshold]
            # their reconstruction errors
            reconstruction_err_kept = noisy_reconstruction_test[noisy_reconstruction_test <= threshold]
            reconstruction_err_kept_int = noisy_reconstruction_test[noisy_reconstruction_test_int <= threshold]
            # their original images
            kept_original = x_test[noisy_reconstruction_test <= threshold]
            kept_original_int = x_test[noisy_reconstruction_test_int <= threshold]
            # indicies of the images to display (worst offenders and just-made-its)
            min_indicies_kept = np.argsort(reconstruction_err_kept)[:display_samples]
            max_indicies_kept = np.argsort(reconstruction_err_kept)[-display_samples:]
            # construct the figure
            fig = plt.figure(figsize=(ncols, nrows))
            # ensure images can be directly adjacent for space
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            # populate the figure
            for i in range(display_samples):
                # least offensive rejects
                ax = fig.add_subplot(nrows, ncols, i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(rejected_original[min_indicies_reject[i]], axis=-1)))
                # original counterparts
                ax = fig.add_subplot(nrows, ncols, (1 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(rejected_samples[min_indicies_reject[i]], axis=-1)))
                # most offensive rejects
                ax = fig.add_subplot(nrows, ncols, (2 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(rejected_original[max_indicies_reject[i]], axis=-1)))
                # original image counterparts
                ax = fig.add_subplot(nrows, ncols, (3 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(rejected_samples[max_indicies_reject[i]], axis=-1)))
                # ideal reconstructed samples
                ax = fig.add_subplot(nrows, ncols, (4 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(kept_original[min_indicies_kept[i]], axis=-1)))
                # original counterparts
                ax = fig.add_subplot(nrows, ncols, (5 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(kept_samples[min_indicies_kept[i]], axis=-1)))
                # almost rejected examples
                ax = fig.add_subplot(nrows, ncols, (6 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(kept_original[max_indicies_kept[i]], axis=-1)))
                # original counterparts
                ax = fig.add_subplot(nrows, ncols, (7 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(kept_samples[max_indicies_kept[i]], axis=-1)))

            plt.show()

            fig, axs = plt.subplots(2)
            fig.tight_layout()
            x = range(len(reconstruction))
            x_t = range(len(noisy_reconstruction_test))
            # first figure, shows the MSE of the noisy examples without the cast
            axs[0].set_title('Reconstruction Error of Samples With Gaussian Noise')
            axs[0].plot(x, np.sort(reconstruction))
            axs[0].plot(x, noisy_reconstruction[np.argsort(reconstruction)])
            axs[0].plot(x, np.sort(noisy_reconstruction))
            axs[0].plot(x_t, np.sort(noisy_reconstruction_test))
            axs[0].plot(x, [threshold for i in x])
            axs[0].legend(['no noise (sorted)', 'noise', 'noise (sorted)', 'test sample', 'threshold'])
            axs[0].set_xlabel('sample number')
            axs[0].set_ylabel('sample MSE')
            # second figure, shows the MSE of the noisy examples with the integer cast
            axs[1].set_title('Reconstruction Error of Samples With Gaussian Noise and Int Cast')
            axs[1].plot(x, np.sort(reconstruction))
            axs[1].plot(x, noisy_reconstruction_int[np.argsort(reconstruction)])
            axs[1].plot(x, np.sort(noisy_reconstruction_int))
            axs[1].plot(x_t, np.sort(noisy_reconstruction_test_int))
            axs[1].plot(x, [threshold for i in x])
            axs[1].legend(['no noise (sorted)', 'noise + int cast', 'noise + int cast (sorted)', 'test sample', 'threshold'])
            axs[1].set_xlabel('sample number')
            axs[1].set_ylabel('sample MSE')
            # show the plot
            plt.show()



