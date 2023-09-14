"""Model architecture and training stages similar to that in:
X. Li et al. (2020). Intelligent cross-machine fault diagnosis approach with deep auto-encoder and domain adaptation.
Neurocomputing, 383, 235–247. https://doi.org/10.1016/j.neucom.2019.12.033


Tensorflow implementation partially following:
https://github.com/wzell/mann/blob/master/models/mann_sentiment_analysis.py.
W. Zellinger et al., "Robust unsupervised domain adaptation
for neural networks via moment alignment," arXiv preprint arXiv:1711.06114, 2017
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from models.mmd import mmd


def default_encoder(dim) -> keras.layers.Dense:
    return keras.layers.Dense(dim, activation='linear')


def default_classifier():
    model = Sequential()
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model


def default_transfer():
    return keras.layers.Dense(10, activation='relu')


def default_decoder(dim) -> keras.layers.Dense:
    model = Sequential()
    model.add(keras.layers.Dense(dim, activation='linear'))
    return model


def transfer_stage_loss(y_true, y_pred):
    """Loss function for transfer stage of network. Does not compare predictions against a ground truth.
    Only compares activations from source against predictions from target, using MDD."""
    transferred_s, transferred_t = tf.split(y_pred, 2, axis=1)
    return mmd(transferred_s, transferred_t)


class Autoencoder:
    def __init__(self, input_dim: int, encoder_dim: int, encoder=None, decoder=None, transfer=None, classifier=None,
                 aux_classifier_weight: float = 1.0, mmd_weight: float = 1.0, target_decode_weight: float = 1.0):
        """Create an autoencoder-based UDA model. Used encoder, decoder and auxiliary classifier in pretraining.
        Then reuses trained encoder, transfer network (minimizes domain discrepancy) and another classifier.
        The auxiliary classifier in the pretrain stage is a clone of the original classifier, if given.
        For a custom transfer network, ensure that the last layer has 'activity_regularizer' as MMD loss."""
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        self.aux_classifier_weight = aux_classifier_weight
        self.mmd_weight = mmd_weight
        self.history_ = None

        self.encoder_model = None
        self.pretrain_model = None
        self.transfer_model = None
        self._build_pretrain_models(encoder, decoder, classifier)
        self._build_transfer_model(transfer, classifier)

    def _build_pretrain_models(self, encoder, decoder, classifier):
        input_s = keras.layers.Input(shape=(self.input_dim,), name='source_input')
        input_t = keras.layers.Input(shape=(self.input_dim,), name='target_input')

        # encoder used for both stages
        if not encoder:
            encoder = default_encoder(self.encoder_dim)
        encoded_s = encoder(input_s)
        encoded_t = encoder(input_t)

        # decoder for pretraining
        if not decoder:
            decoder = default_decoder(self.input_dim)
        decoded_s = decoder(encoded_s)
        decoded_t = decoder(encoded_t)

        # auxiliary classifier for pretraining, independent copy of final classifier if given
        if not classifier:
            aux_classifier = default_classifier()
        else:
            aux_classifier = classifier.copy_model()
        aux_classified_s = aux_classifier(decoded_s)

        self.encoder_model = Model(inputs=[input_s, input_t], outputs=[encoded_s, encoded_t])
        self.pretrain_model = Model(inputs=[input_s, input_t],
                                    outputs=[decoded_s, decoded_t, aux_classified_s])
        self.pretrain_model.compile(loss=['mean_squared_error', 'mean_squared_error', 'binary_crossentropy'],
                                    loss_weights=[1.0, 1.0, self.aux_classifier_weight])

    def _build_transfer_model(self, transfer, classifier):
        # set input layers, which will be encoded data
        input_s = keras.layers.Input(shape=(self.encoder_dim,), name='source_input')
        input_t = keras.layers.Input(shape=(self.encoder_dim,), name='target_input')

        if not transfer:
            transfer = default_transfer()
        transferred_s = transfer(input_s)
        transferred_t = transfer(input_t)

        if not classifier:
            classifier = default_classifier()
        classified_s = classifier(transferred_s)

        transferred_concat = tf.keras.backend.concatenate([transferred_s, transferred_t])
        self.transfer_model = Model(inputs=[input_s, input_t],
                                    outputs=[transferred_concat, classified_s])
        self.transfer_model.compile(loss=[transfer_stage_loss, 'binary_crossentropy'],
                                    loss_weights=[self.mmd_weight, 1.0])

    def fit(self, xs, ys, xt, **fit_params):

        # pretrain autoencoder with auxiliary classifier
        self.pretrain_model.fit([xs, xt], [xs, xt, ys], **fit_params)

        # use fixed encodings for final classification, while minimizing MDD halfway
        enc_s, enc_t = self.encoder_model.predict([xs, xt], verbose=0)
        hist = self.transfer_model.fit([enc_s, enc_t], [ys, ys], **fit_params)
        self.history_ = hist.history
        return self

    def predict(self, x, verbose=0):
        # some redundancy because networks were compiled expecting two data sets
        enc, _, = self.encoder_model.predict([x, x], verbose=verbose)
        _, pred = self.transfer_model.predict([enc, enc], verbose=verbose)
        return pred
