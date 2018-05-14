"""
change log:
- Version 1: change the out_grads of `backward` function of `ReLU` layer into inputs_grads instead of in_grads
"""

import numpy as np 
from utils.tools import *

class Layer(object):
    """
    
    """
    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, ..., in_features), 
            typically (batch, in_features), or (batch, T, in_features) for sequencical data

        # Returns
            outputs: numpy array with shape (batch, ..., out_features)
        """
        batch = inputs.shape[0]
        b_reshaped = self.bias.reshape((1,)*(inputs.ndim-1)+self.bias.shape)
        outputs = np.dot(inputs, self.weights)+b_reshaped
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, ..., out_features), gradients to outputs
            inputs: numpy array with shape (batch, ..., in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, ..., in_features), gradients to inputs
        """
        dot_axes = np.arange(inputs.ndim-1)
        self.w_grad = np.tensordot(np.nan_to_num(inputs), in_grads, axes=(dot_axes, dot_axes))
        self.b_grad = np.sum(in_grads, axis=tuple(dot_axes))
        out_grads = np.dot(in_grads, self.weights.T)
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class TemporalPooling(Layer):
    """
    Temporal mean-pooling that ignores NaN
    """
    def __init__(self, name='temporal_pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(TemporalPooling, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, time_steps, units)

        # Returns
            outputs: numpy array with shape (batch, units)
        """
        mask = ~np.any(np.isnan(inputs), axis=2)
        outputs = np.sum(np.nan_to_num(inputs), axis=1)
        outputs /= np.sum(mask, axis=1, keepdims=True)
        return outputs
        
    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, units), gradients to outputs
            inputs: numpy array with shape (batch, time_steps, units), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, time_steps, units), gradients to inputs
        """
        batch, time_steps, units = inputs.shape
        mask = ~np.any(np.isnan(inputs), axis=2)
        in_grads = in_grads/np.sum(mask, axis=1, keepdims=True)
        out_grads = np.repeat(in_grads, time_steps, 1).reshape((batch, units, time_steps)).transpose(0, 2, 1)
        out_grads *= ~np.isnan(inputs)
        return out_grads


if __name__ == '__main__':
    import numpy as np
    from utils.tools import rel_error
    from utils.check_grads import check_grads_layer
    import keras
    from keras import layers
    from keras import models
    from keras import optimizers
    from keras import backend as K

    print('Testing Fully Connected Layer...')
    inputs = np.random.uniform(size=(10, 3, 20))
    fclayer = FCLayer(in_features=inputs.shape[-1], out_features=100)
    out = fclayer.forward(inputs)
    keras_model = keras.Sequential()
    keras_layer = layers.Dense(100, input_shape=inputs.shape[1:], use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')
    keras_model.add(keras_layer)
    sgd = optimizers.SGD(lr=0.01)
    keras_model.compile(loss='mean_squared_error', optimizer='sgd')
    keras_layer.set_weights([fclayer.weights, fclayer.bias])
    keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
    print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))
    in_grads = np.random.uniform(size=(10, 3, 100))
    check_grads_layer(fclayer, inputs, in_grads)

    print('Testing TemporalPooling Layer...')
    inputs = np.random.uniform(size=(10, 3, 20))
    pooling_layer = TemporalPooling()
    out = pooling_layer.forward(inputs)
    keras_model = keras.Sequential()
    keras_layer = layers.GlobalAveragePooling1D(input_shape=inputs.shape[1:])
    keras_model.add(keras_layer)
    sgd = optimizers.SGD(lr=0.01)
    keras_model.compile(loss='mean_squared_error', optimizer='sgd')
    keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
    print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))
    in_grads = np.random.uniform(size=(10, 20))
    check_grads_layer(pooling_layer, inputs, in_grads)