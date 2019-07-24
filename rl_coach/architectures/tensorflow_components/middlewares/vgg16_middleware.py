from typing import Union, List

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from rl_coach.architectures.tensorflow_components.middlewares.middleware import Middleware
from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.core_types import Middleware_VGG16_Embedding
from rl_coach.utils import force_list

class VGG16Middleware(Middleware):
    def __init__(self, activation_function=None, #weights: str = "imagenet",
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout_rate: float = 0.0,
                 name="middleware_vgg16_embedder", dense_layer=Dense, is_training=False):
        super().__init__(activation_function=activation_function, batchnorm=batchnorm,
                         dropout_rate=dropout_rate, scheme=scheme, name=name, dense_layer=dense_layer,
                         is_training=is_training)
        self.return_type = Middleware_VGG16_Embedding
        self.layers = []
       # self.weights = weights
    #TODO: 
    #       1. Move last layers to schemes
    #       3. Find out with Atari's images preprocessing(not necessary now)
    #       4. Split original dqn_agent.py and dqn agent on VGG16
    #       6. rovel in shared_running_stats.py   
    #       7. thinking about pre_trained model
    #       8. check embedder parameters
    #       9. ADDED choosing INITIALIZER

    def _build_module(self):
        self.layers.append(self.input)

        self.activation_function = tf.nn.relu
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        window_size = (3, 3)
        self.layers.append(Conv2D(64, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(Conv2D(64, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(MaxPooling2D()(self.layers[-1]))
        self.layers.append(Conv2D(128, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(Conv2D(128, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(MaxPooling2D()(self.layers[-1]))
        self.layers.append(Conv2D(256, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(Conv2D(256, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(Conv2D(256, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))       
        self.layers.append(MaxPooling2D()(self.layers[-1]))
        self.layers.append(Conv2D(512, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(Conv2D(512, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(Conv2D(512, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1])) 
        self.layers.append(MaxPooling2D()(self.layers[-1]))
        self.layers.append(Conv2D(512, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(Conv2D(512, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(Conv2D(512, window_size, padding='same', activation=self.activation_function, kernel_initializer=initializer)(self.layers[-1]))
        self.layers.append(MaxPooling2D()(self.layers[-1]))
        self.layers.append(Flatten()(self.layers[-1]))
      
        for idx, layer_params in enumerate(self.layers_params):
            print(idx, layer_params)
            self.layers.extend(force_list(
                layer_params(self.layers[-1], name='{}_{}'.format(layer_params.__class__.__name__, idx),
                            is_training=self.is_training, kernel_initializer=initializer, activation=self.activation_function)
            ))

        self.output = self.layers[-1]

    @property
    def schemes(self):
        return {
            MiddlewareScheme.Empty:
                [],

            # ppo
            MiddlewareScheme.Shallow:
                [
                    self.dense_layer(64)
                ],

            # dqn
            MiddlewareScheme.Medium:
                [
                    self.dense_layer(512)
                ],

            MiddlewareScheme.Deep: \
                [
                    self.dense_layer(128),
                    self.dense_layer(128),
                    self.dense_layer(128)
                ]
        }
