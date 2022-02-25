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

batch_size = 32
input_shape = (28, 28, 1)
loaded = False
(x_train, y_train), (x_test, y_test) = (None, None), (None, None)


def build_AE(input_shape, activation='selu', encoder_filter_size=(4, 4), init=tf.keras.initializers.LecunNormal(),
             learning_rate=0.001):

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

    dec_inputs = layers.Input((49,), name='dec_in')

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

    encoder = tf.keras.Model(inputs=[enc_inputs], outputs=[enc_outputs], name='mnist_enc')

    decoder = tf.keras.Model(inputs=[dec_inputs], outputs=[dec_outputs], name='mnist_dec')

    model = tf.keras.Model(inputs=[enc_inputs], outputs=decoder(encoder(enc_inputs)), name='mnist_ae')

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='mse', optimizer=opt)

    print(model.summary())

    return model, encoder, decoder


def load_recent_model(dirname, filebase):
    try:
        files = [f for f in os.listdir(dirname) if re.match(r'%s' % filebase, f)]
        files = sorted(files, key=lambda x: float(x.split('_')[-1]))

        return tf.keras.models.load_model(files[-1])  # return file with latest timestamp

    except IndexError as e:
        return None


def create_parser():
    parser = argparse.ArgumentParser(description='MNIST experiment')
    parser.add_argument('--load', action='store_true', help='load a model rather than creating a new one')
    parser.add_argument('--tree', action='store_true', help='build a tree and classify on encoded representation')
    parser.add_argument('--forest', action='store_true', help='build a random fores and classify on encoded representation')
    parser.add_argument('--ntrees', type=int, default=25, help='number of trees in random forest')
    parser.add_argument('--maxdepth', type=int, default=4, help='max depth of random forest trees')
    parser.add_argument('--plot', action='store_true', help='show a plot of some test images passed through AE')
    parser.add_argument('--maps', action='store_true', help='generate feature map visualizations for cnn')
    parser.add_argument('--nn', action='store_true', help='train a simple neural net on the encoded input')
    parser.add_argument('--newnn', action='store_true', help='train a new simple neural net on the encoded input')
    parser.add_argument('--noise', action='store_true', help='evaluate trained models on noisy test data')
    parser.add_argument('--mu', type=float, default=0, help='mean for gaussian noise')
    parser.add_argument('--sigma', type=float, default=15, help='standard dev for gaussian noise')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    model, encoder, decoder = None, None, None

    t, nn, rf = None, None, None

    if args.load:
        model, encoder, decoder = load_recent_model('.', 'autoenc'), load_recent_model('.', 'enc'), load_recent_model('.', 'dec')
    else:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True

        model, encoder, decoder = build_AE(input_shape)
        model.fit(x_train, x_train, batch_size=batch_size, epochs=25)

        tf.keras.models.save_model(model, f'autoenc_{time()}')
        tf.keras.models.save_model(encoder, f'enc_{time()}')
        tf.keras.models.save_model(decoder, f'dec_{time()}')

    if args.plot:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True

        inds = [0, 1, 2, 3, 4, 5, 6]
        ncols = len(inds)
        nrows = 2

        fig = plt.figure(figsize=(ncols, nrows), dpi=300)

        for i in inds:
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            encoded = encoder.predict(np.array([x_test[inds[i]], ]))
            decoded = decoder.predict(encoded)
            plt.imshow(array_to_img(decoded[0]))
            ax = fig.add_subplot(nrows, ncols, i + 1 + ncols)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            plt.imshow(array_to_img(np.expand_dims(x_test[inds[i]], axis=-1)))

        plt.show()

    if args.tree:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True

        x_train_transformed = encoder.predict(x_train)
        x_test_transformed = encoder.predict(x_test)

        t = tree.DecisionTreeClassifier(max_depth=16)
        t.fit(x_train_transformed, y_train)

    if args.forest:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True

        x_train_transformed = encoder.predict(x_train)
        x_test_transformed = encoder.predict(x_test)

        rf = RandomForestClassifier(args.ntrees, max_depth=args.maxdepth)
        rf.fit(x_train_transformed, y_train)

    if args.nn:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True

        if not args.newnn:
            nn = load_recent_model(".", "nn")

        if nn is None:

            OHE = sklearn.preprocessing.OneHotEncoder()

            OHE.fit(np.concatenate((y_train, y_test)).reshape(-1, 1))

            x_train_transformed = encoder.predict(x_train)
            y_train_transformed = OHE.transform(y_train.reshape(-1, 1)).toarray()
            x_test_transformed = encoder.predict(x_test)
            y_test_transformed = OHE.transform(y_test.reshape(-1, 1)).toarray()

            input = layers.Input(x_train_transformed.shape[1])

            x = layers.Dense(49, kernel_initializer=tf.keras.initializers.LecunNormal(), activation='selu', use_bias=True,
                             bias_initializer='zeros')(input)

            output = layers.Dense(10, kernel_initializer=tf.keras.initializers.LecunNormal(),
                                  activation=tf.keras.activations.softmax)(x)

            opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

            nn = tf.keras.models.Model(inputs=[input], outputs=[output])

            nn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])

            print(nn.summary())

            nn.fit(x_train_transformed, y_train_transformed, epochs=10, batch_size=batch_size)

            eval = nn.evaluate(x_test_transformed, y_test_transformed)

            print(f'score: {eval}')

            tf.keras.models.save_model(nn, f'nn_{time()}')

    if args.maps:
        if not loaded:
            # mnist digits
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            loaded = True

        layer_names = [layer.name for layer in encoder.layers if len(layer.output.shape) == 4][2:]
        layer_outputs = [layer.output for layer in encoder.layers if len(layer.output.shape) == 4][2:]

        feature_map_model = tf.keras.models.Model(inputs=[encoder.input], outputs=layer_outputs)
        feature_map_model.compile()

        feature_maps = [np.average(feature_map, axis=0) for feature_map in feature_map_model.predict(x_test)]
        print([feature_map.shape for feature_map in feature_maps])

        ncols = 49
        nrows = len(layer_names)
        fig = plt.figure(figsize=(ncols, nrows))
        row = 0
        for layer_name, feature_map in zip(layer_names, feature_maps):
            k = feature_map.shape[-1]
            size = feature_map.shape[1]
            for i in range(k):
                # iterating over a feature map of a particular layer to separate all feature images.
                feature_image = feature_map[:, :, i]
                feature_image -= feature_image.mean()
                feature_image /= feature_image.std()
                feature_image *= 64
                feature_image += 128
                feature_image = np.clip(feature_image, 0, 255).astype('uint8')
                ax = fig.add_subplot(nrows, ncols, row * ncols + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(feature_image, axis=-1)))

            row += 1
        plt.show()

    if args.noise:
        if loaded:
            OHE = sklearn.preprocessing.OneHotEncoder()

            OHE.fit(np.concatenate((y_train, y_test)).reshape(-1, 1))

            mu, sigma = args.mu, args.sigma

            x_train_noisy = np.clip(np.add(x_train, np.random.normal(mu, sigma, x_train.shape)), 0, 255)
            x_test_noisy = np.clip(np.add(x_test, np.random.normal(mu, sigma, x_test.shape)), 0, 255)

            x_train_noisy_int = x_train_noisy.astype(int)
            x_test_noisy_int = x_test_noisy.astype(int)

            x_train_transformed = encoder.predict(x_train)
            x_test_transformed = encoder.predict(x_test)

            y_train_transformed = OHE.transform(y_train.reshape(-1, 1)).toarray()
            y_test_transformed = OHE.transform(y_test.reshape(-1, 1)).toarray()

            x_train_transformed_noisy = encoder.predict(x_train_noisy)
            x_test_transformed_noisy = encoder.predict(x_test_noisy)
            x_train_transformed_noisy_int = encoder.predict(x_train_noisy_int)
            x_test_transformed_noisy_int = encoder.predict(x_test_noisy_int)

            def mse(a, b):
                return np.mean(np.square(np.subtract(a, b)), axis=(1, 2))

            noisy_reconstruction = mse(np.squeeze(model.predict(x_train_noisy), axis=-1), x_train.astype(float))
            noisy_reconstruction_int = mse(np.squeeze(model.predict(x_train_noisy_int), axis=-1), x_train.astype(float))
            reconstruction = mse(np.squeeze(model.predict(x_train), axis=-1), x_train.astype(float))

            noisy_reconstruction_test = mse(np.squeeze(model.predict(x_test_noisy), axis=-1), x_test.astype(float))
            noisy_reconstruction_test_int = mse(np.squeeze(model.predict(x_test_noisy_int), axis=-1), x_test.astype(float))

            threshold = np.sort(noisy_reconstruction)[int(len(noisy_reconstruction)*.95)]  # max training reconstruction

            x_test_non_reject = x_test_transformed_noisy[noisy_reconstruction_test <= threshold]
            y_test_non_reject = y_test[noisy_reconstruction_test <= threshold]
            y_test_transformed_non_reject = OHE.transform(y_test_non_reject.reshape(-1, 1)).toarray()

            noisy_rejected = len(x_test_transformed_noisy[noisy_reconstruction_test > threshold])

            x_test_non_reject_int = x_test_transformed_noisy_int[noisy_reconstruction_test_int <= threshold]
            y_test_non_reject_int = y_test[noisy_reconstruction_test_int <= threshold]
            y_test_transformed_non_reject_int = OHE.transform(y_test_non_reject_int.reshape(-1, 1)).toarray()

            noisy_rejected_int = len(x_test_transformed_noisy_int[noisy_reconstruction_test_int > threshold])

            if t is not None:
                t_train_score = t.score(x_train_transformed, y_train)
                t_test_score = t.score(x_test_transformed, y_test)

                t_train_score_noisy = t.score(x_train_transformed_noisy, y_train)
                t_train_score_noisy_int = t.score(x_train_transformed_noisy_int, y_train)

                t_test_score_noisy = t.score(x_test_transformed_noisy, y_test)
                t_test_score_noisy_int = t.score(x_test_transformed_noisy_int, y_test)

                t_test_score_noisy_reject = t.score(x_test_non_reject, y_test_non_reject)
                t_test_score_noisy_reject_int = t.score(x_test_non_reject_int, y_test_non_reject_int)

                with open(f'results_0/t_scores_{mu}_{sigma}.txt', 'a') as rfile:
                    rfile.write(f'baseline     - train, test: {t_train_score}, {t_test_score}\n'
                                f'noisy        - train, test: {t_train_score_noisy}, {t_test_score_noisy}\n'
                                f'noisy int    - train, test: {t_train_score_noisy_int}, {t_test_score_noisy_int}\n'
                                f'noisy reject - noise, int : {t_test_score_noisy_reject}, {t_test_score_noisy_reject_int}\n'
                                f'rejected samples: {noisy_rejected}, {noisy_rejected_int}\n')

            if nn is not None:
                nn_train_score = nn.evaluate(x_train_transformed, y_train_transformed)[-1]
                nn_test_score = nn.evaluate(x_test_transformed, y_test_transformed)[-1]

                nn_train_score_noisy = nn.evaluate(x_train_transformed_noisy, y_train_transformed)[-1]
                nn_train_score_noisy_int = nn.evaluate(x_train_transformed_noisy_int, y_train_transformed)[-1]

                nn_test_score_noisy = nn.evaluate(x_test_transformed_noisy, y_test_transformed)[-1]
                nn_test_score_noisy_int = nn.evaluate(x_test_transformed_noisy_int, y_test_transformed)[-1]

                nn_test_score_noisy_reject = nn.evaluate(x_test_non_reject, y_test_transformed_non_reject)[-1]
                nn_test_score_noisy_reject_int = nn.evaluate(x_test_non_reject_int, y_test_transformed_non_reject_int)[-1]

                with open(f'results_0/nn_scores_{mu}_{sigma}.txt', 'a') as rfile:
                    rfile.write(f'baseline     - train, test: {nn_train_score}, {nn_test_score}\n'
                                f'noisy        - train, test: {nn_train_score_noisy}, {nn_test_score_noisy}\n'
                                f'noisy int    - train, test: {nn_train_score_noisy_int}, {nn_test_score_noisy_int}\n'
                                f'noisy reject - noise, int : {nn_test_score_noisy_reject}, {nn_test_score_noisy_reject_int}\n'
                                f'rejected samples: {noisy_rejected}, {noisy_rejected_int}\n')

            if rf is not None:
                rf_train_score = rf.score(x_train_transformed, y_train)
                rf_test_score = rf.score(x_test_transformed, y_test)
                rf_train_score_noisy = rf.score(x_train_transformed_noisy, y_train)
                rf_train_score_noisy_int = rf.score(x_train_transformed_noisy_int, y_train)

                rf_test_score_noisy = rf.score(x_test_transformed_noisy, y_test)
                rf_test_score_noisy_int = rf.score(x_test_transformed_noisy_int, y_test)

                rf_test_score_noisy_reject = rf.score(x_test_non_reject, y_test_non_reject)
                rf_test_score_noisy_reject_int = rf.score(x_test_non_reject_int, y_test_non_reject_int)

                with open(f'results_0/rf_scores_{mu}_{sigma}.txt', 'a') as rfile:
                    rfile.write(f'baseline     - train, test: {rf_train_score}, {rf_test_score}\n'
                                f'noisy        - train, test: {rf_train_score_noisy}, {rf_test_score_noisy}\n'
                                f'noisy int    - train, test: {rf_train_score_noisy_int}, {rf_test_score_noisy_int}\n'
                                f'noisy reject - noise, int : {rf_test_score_noisy_reject}, {rf_test_score_noisy_reject_int}\n'
                                f'rejected samples: {noisy_rejected}, {noisy_rejected_int}\n')

            display_samples = 5
            nrows = 8
            ncols = display_samples

            rejected_samples = x_test_noisy[noisy_reconstruction_test > threshold]
            rejected_samples_int = x_test_noisy_int[noisy_reconstruction_test_int > threshold]

            reconstruction_err_reject = noisy_reconstruction_test[noisy_reconstruction_test > threshold]
            reconstruction_err_reject_int = noisy_reconstruction_test[noisy_reconstruction_test_int > threshold]

            rejected_original = x_test[noisy_reconstruction_test > threshold]
            rejected_original_int = x_test[noisy_reconstruction_test_int > threshold]

            min_indicies_reject = np.argsort(reconstruction_err_reject)[:display_samples]
            max_indicies_reject = np.argsort(reconstruction_err_reject)[-display_samples:]

            kept_samples = x_test_noisy[noisy_reconstruction_test <= threshold]
            kept_samples_int = x_test_noisy_int[noisy_reconstruction_test_int <= threshold]

            reconstruction_err_kept = noisy_reconstruction_test[noisy_reconstruction_test <= threshold]
            reconstruction_err_kept_int = noisy_reconstruction_test[noisy_reconstruction_test_int <= threshold]

            kept_original = x_test[noisy_reconstruction_test <= threshold]
            kept_original_int = x_test[noisy_reconstruction_test_int <= threshold]

            min_indicies_kept = np.argsort(reconstruction_err_kept)[:display_samples]
            max_indicies_kept = np.argsort(reconstruction_err_kept)[-display_samples:]

            fig = plt.figure(figsize=(ncols, nrows))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            model_names = ['t', 'rf', 'nn']
            for i in range(display_samples):
                ax = fig.add_subplot(nrows, ncols, i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(rejected_original[min_indicies_reject[i]], axis=-1)))

                ax = fig.add_subplot(nrows, ncols, (1 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(rejected_samples[min_indicies_reject[i]], axis=-1)))

                ax = fig.add_subplot(nrows, ncols, (2 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(rejected_original[max_indicies_reject[i]], axis=-1)))

                ax = fig.add_subplot(nrows, ncols, (3 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(rejected_samples[max_indicies_reject[i]], axis=-1)))

                ax = fig.add_subplot(nrows, ncols, (4 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(kept_original[min_indicies_kept[i]], axis=-1)))

                ax = fig.add_subplot(nrows, ncols, (5 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(kept_samples[min_indicies_kept[i]], axis=-1)))

                ax = fig.add_subplot(nrows, ncols, (6 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(kept_original[max_indicies_kept[i]], axis=-1)))

                ax = fig.add_subplot(nrows, ncols, (7 * ncols) + i + 1)
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                plt.imshow(array_to_img(np.expand_dims(kept_samples[max_indicies_kept[i]], axis=-1)))

            plt.show()

            fig, axs = plt.subplots(2)
            fig.tight_layout()
            x = range(len(reconstruction))
            x_t = range(len(noisy_reconstruction_test))

            axs[0].set_title('Reconstruction Error of Samples With Gaussian Noise')
            axs[0].plot(x, np.sort(reconstruction))
            axs[0].plot(x, noisy_reconstruction[np.argsort(reconstruction)])
            axs[0].plot(x, np.sort(noisy_reconstruction))
            axs[0].plot(x_t, np.sort(noisy_reconstruction_test))
            axs[0].plot(x, [threshold for i in x])
            axs[0].legend(['no noise (sorted)', 'noise', 'noise (sorted)', 'test sample', 'threshold'])
            axs[0].set_xlabel('sample number')
            axs[0].set_ylabel('sample MSE')

            axs[1].set_title('Reconstruction Error of Samples With Gaussian Noise and Int Cast')
            axs[1].plot(x, np.sort(reconstruction))
            axs[1].plot(x, noisy_reconstruction_int[np.argsort(reconstruction)])
            axs[1].plot(x, np.sort(noisy_reconstruction_int))
            axs[1].plot(x_t, np.sort(noisy_reconstruction_test_int))
            axs[1].plot(x, [threshold for i in x])
            axs[1].legend(['no noise (sorted)', 'noise + int cast', 'noise + int cast (sorted)', 'test sample', 'threshold'])
            axs[1].set_xlabel('sample number')
            axs[1].set_ylabel('sample MSE')

            plt.show()



