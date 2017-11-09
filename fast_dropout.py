from keras import backend as K
from keras import activations
from keras import initializers
from keras.engine.topology import Layer
from keras.engine import InputSpec
import numpy as np
import tensorflow as tf

class FastDropoutDenseLayer(Layer):
    """
    Implements Fast Gaussian approximation of dropout according to "Wang et al.
    2013: Fast dropout training." Works either with ReLU activation or as a linear unit.

    Outputs the mean (and possibly the variance) of Gaussians describing the
    output distribution of each unit in this layer. For dropout=1 (probability
    of keeping an input=1 => no units are dropped) works like normal dense layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: 'ReLU' (default) or 'linear'
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        dropout: float [0;1]. Probability of keeping each of the input features.
            dropout=1 means no dropout.
        has_var_input: bool. True (default) if previous layer outputs also variance.
        has_var_output: bool. True (default) if this layer should also output
            the variance.
    # Input shape
        2D tensor with shape: `(batch_size, input_dim)` or
            `(batch_size, 2* input_dim)` if has_var_input=True

    # Output shape
        2D tensor with shape: `(batch_size, units)` or
            `(batch_size, 2*units)` if has_var_output=True
    """
    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 dropout=0.5,
                 activation='ReLU',
                 has_var_input=True,
                 has_var_output=True,
                 **kwargs):
        assert activation=='ReLU' or activation=='linear', 'Invalid activation \
            parameter. Choose either ReLU or linear'

        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.dropout = K.constant(dropout)
        self.activation = activation
        self.has_var_input = has_var_input
        self.has_var_output = has_var_output
        super(FastDropoutDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        # adjust for variance input
        if self.has_var_input:
            input_dim /= 2
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel')
        # unlike Dense layer, we don't make bias optional
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_shape[-1]})
        super(FastDropoutDenseLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # adjust for input variance
        if self.has_var_input:
            mu_in, s_in = tf.split(x,2, axis=-1)
        else:
            mu_in = x
            s_in = 0.0

        # compute dropout mean and variance
        mu = K.bias_add(self.dropout* K.dot(mu_in,self.kernel), self.bias)
        s = K.dot(self.dropout * (1-self.dropout) * K.square(mu_in) + self.dropout * K.square(s_in), K.square(self.kernel))

        # adjust output mean and variance by activation function
        if self.activation == 'ReLU':
            r = mu/s
            dist = tf.distributions.Normal(0.0,1.0)
            mu_out = dist.cdf(r)*mu + s*dist.prob(r)
            s_out = s
        else:
            mu_out, s_out = mu, s

        # stack mean and variance into output, if desired
        if self.has_var_output:
            return tf.concat([mu_out, s_out], axis=-1)
        else:
            return mu

    def compute_output_shape(self, input_shape):
        if self.has_var_output:
            return (input_shape[0], 2*self.units)
        return (input_shape[0], self.units)

def FastDropoutCrossEntropyLoss(target, output, sample_count=10):
    """
    Attempt to implement Softmax Cross Entropy loss according to "Wang et al.
    2013: Fast dropout training." Does not work though.

    # Arguments
        target: tensor of ground truth labels
        output: tensor of network predictions, assuming the last layer of the
            network is linear (this loss does the softmax internally)
        sample_count: positive int (default 10) for number of MC samples to
            evaluate the expectation.
    """
    # 1. make a distribution
    mu, s = tf.split(output, 2, axis=-1)
    dist = tf.distributions.Normal(mu, s)

    # 2. prepare epsilon
    _epsilon = tf.convert_to_tensor(K.epsilon(), s.dtype.base_dtype)

    # 3. prepare Loss accumulator
    # Hack using sum to get the desired shape...
    # no idea why tf.shape(output)[0] doesn't work
    loss = K.zeros_like(tf.shape(K.sum(output, axis=-1)), dtype=tf.float32)

    # 4. sample from the distribution
    for i in xrange(sample_count):
        s = dist.sample()
        # 5. compute softmax
        sm = K.softmax(s)
        # 6. clip values (like)
        sm = tf.clip_by_value(sm, _epsilon, 1. - _epsilon)
        loss += K.sum(target * K.log(sm), axis=1)
    # 6. compute the expectation
    return loss
