import tensorflow as tf
import tensorflow_addons as tfa
from data import DataSet

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import concatenate
from keras.layers.merge import add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D
from custom_layers import subtract, norm
from grl import GradientReversal
from keras.utils import plot_model

class DuelingTripletNetwork:
    def __init__(self, time_length=1, vars_count=10):
        self.shape = (time_length, vars_count)
        self.model = self.make_model()

    def make_model(self):

        #x, x1=x-n, x2=x+n, x3=x!n
        x = Input(shape=self.shape, name='x')
        x1 = Input(shape=self.shape, name='x1')
        x2 = Input(shape=self.shape, name='x2')
        x3 = Input(shape=self.shape, name='x3')
        fe = self.build_feature_extractor()

        e = fe(x)
        e1 = fe(x1)
        e2 = fe(x2)
        e3 = fe(x3)

        # Get the differences
        d1 = subtract(e, e1)
        d2 = subtract(e, e2)
        d3 = subtract(e2, e3)

        # Normalize the differences
        n1 = norm(d1)
        n2 = norm(d2)
        n3 = norm(d3)

        # Compare
        out = Activation('sigmoid')(subtract(n2, n1))


        domain_classifier = Activation('softmax')(subtract(n3, n2))

#        grl_layer = GradientReversal(0.31)
#        dann_in = grl_layer(domain_classifier)
#        out2 = GradientReversal(domain_classifier)

        return Model(inputs=[x, x1, x2, x3], outputs=[out])


    def build_feature_extractor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding="same",
                use_bias=False, input_shape=self.shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),

            tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding="same",
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same",
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),

            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(256, activation=None), 
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ])
        return model


