### IMPORTANT COMMENT by Philippe Paradis. I am NOT the other of the code in this file.

## The author is OpenAI. This code was taken verbatim from OpenAI's public Git Repository and is
## being used subject to the MIT licence. All copyrights remain with OpenAI.

## This source file is contains two sections: section (1) from the file at URL::
##  (1) https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py
## Section (2) comes from the file at URL:
##  (2) https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_feature_matching.py

###
### Section (1):
### URL: https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py
###

"""
neural network stuff, intended to be used with Lasagne
"""

import numpy as np
import theano as th
import theano.tensor as T
import lasagne
from lasagne.layers import dnn
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# T.nnet.relu has some stability issues, this is better
def relu(x):
    return T.maximum(x, 0)
    
def lrelu(x, a=0.2):
    return T.maximum(x, a*x)

def centered_softplus(x):
    return T.nnet.softplus(x) - np.cast[th.config.floatX](np.log(2.))

def log_sum_exp(x, axis=1):
    m = T.max(x, axis=axis)
    return m+T.log(T.sum(T.exp(x-m.dimshuffle(0,'x')), axis=axis))

def adam_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    grads = T.grad(cost, params)
    t = th.shared(np.cast[th.config.floatX](1.))
    for p, g in zip(params, grads):
        v = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        mg = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        v_t = mom1*v + (1. - mom1)*g
        mg_t = mom2*mg + (1. - mom2)*T.square(g)
        v_hat = v_t / (1. - mom1 ** t)
        mg_hat = mg_t / (1. - mom2 ** t)
        g_t = v_hat / T.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append((v, v_t))
        updates.append((mg, mg_t))
        updates.append((p, p_t))
    updates.append((t, t+1))
    return updates

class WeightNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), train_g=False, init_stdv=1., nonlinearity=relu, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.init_stdv = init_stdv
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False, trainable=train_g)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

        # scale weights in layer below
        incoming.W_param = incoming.W
        #incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            if isinstance(incoming, Deconv2DLayer):
                W_axes_to_sum = (0,2,3)
                W_dimshuffle_args = ['x',0,'x','x']
            else:
                W_axes_to_sum = (1,2,3)
                W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = incoming.W_param * (self.g/T.sqrt(1e-6 + T.sum(T.square(incoming.W_param),axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(1e-6 + T.sum(T.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True))

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            inv_stdv = self.init_stdv/T.sqrt(T.mean(T.square(input), self.axes_to_sum))
            input *= inv_stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m*inv_stdv), (self.g, self.g*inv_stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(input)

def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, target_shape, filter_size, stride=(2, 2),
                 W=lasagne.init.Normal(0.05), b=lasagne.init.Constant(0.), nonlinearity=relu, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.target_shape = target_shape
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.filter_size = lasagne.layers.dnn.as_tuple(filter_size, 2)
        self.stride = lasagne.layers.dnn.as_tuple(stride, 2)
        self.target_shape = target_shape

        self.W_shape = (incoming.output_shape[1], target_shape[1], filter_size[0], filter_size[1])
        self.W = self.add_param(W, self.W_shape, name="W")
        if b is not None:
            self.b = self.add_param(b, (target_shape[1],), name="b")
        else:
            self.b = None

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=self.target_shape, kshp=self.W_shape, subsample=self.stride, border_mode='half')
        activation = op(self.W, input, self.target_shape[2:])

        if self.b is not None:
            activation += self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return self.target_shape

class MinibatchLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_kernels, dim_per_kernel=5, theta=lasagne.init.Normal(0.05),
                 log_weight_scale=lasagne.init.Constant(0.), b=lasagne.init.Constant(-1.), **kwargs):
        super(MinibatchLayer, self).__init__(incoming, **kwargs)
        self.num_kernels = num_kernels
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_kernels, dim_per_kernel), name="theta")
        self.log_weight_scale = self.add_param(log_weight_scale, (num_kernels, dim_per_kernel), name="log_weight_scale")
        self.W = self.theta * (T.exp(self.log_weight_scale)/T.sqrt(T.sum(T.square(self.theta),axis=0))).dimshuffle('x',0,1)
        self.b = self.add_param(b, (num_kernels,), name="b")
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:])+self.num_kernels)

    def get_output_for(self, input, init=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        activation = T.tensordot(input, self.W, [[1], [0]])
        abs_dif = (T.sum(abs(activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)),axis=2)
                    + 1e6 * T.eye(input.shape[0]).dimshuffle(0,'x',1))

        if init:
            mean_min_abs_dif = 0.5 * T.mean(T.min(abs_dif, axis=2),axis=0)
            abs_dif /= mean_min_abs_dif.dimshuffle('x',0,'x')
            self.init_updates = [(self.log_weight_scale, self.log_weight_scale-T.log(mean_min_abs_dif).dimshuffle(0,'x'))]
        
        f = T.sum(T.exp(-abs_dif),axis=2)

        if init:
            mf = T.mean(f,axis=0)
            f -= mf.dimshuffle('x',0)
            self.init_updates.append((self.b, -mf))
        else:
            f += self.b.dimshuffle('x',0)

        return T.concatenate([input, f], axis=1)

class BatchNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), nonlinearity=relu, **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False)
        self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean", regularizable=False, trainable=False)
        self.avg_batch_var = self.add_param(lasagne.init.Constant(1.), (k,), name="avg_batch_var", regularizable=False, trainable=False)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            norm_features = (input-self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)) / T.sqrt(1e-6 + self.avg_batch_var).dimshuffle(*self.dimshuffle_args)
        else:
            batch_mean = T.mean(input,axis=self.axes_to_sum).flatten()
            centered_input = input-batch_mean.dimshuffle(*self.dimshuffle_args)
            batch_var = T.mean(T.square(centered_input),axis=self.axes_to_sum).flatten()
            batch_stdv = T.sqrt(1e-6 + batch_var)
            norm_features = centered_input / batch_stdv.dimshuffle(*self.dimshuffle_args)

            # BN updates
            new_m = 0.9*self.avg_batch_mean + 0.1*batch_mean
            new_v = 0.9*self.avg_batch_var + T.cast((0.1*input.shape[0])/(input.shape[0]-1),th.config.floatX)*batch_var
            self.bn_updates = [(self.avg_batch_mean, new_m), (self.avg_batch_var, new_v)]

        if hasattr(self, 'g'):
            activation = norm_features*self.g.dimshuffle(*self.dimshuffle_args)
        else:
            activation = norm_features
        if hasattr(self, 'b'):
            activation += self.b.dimshuffle(*self.dimshuffle_args)

        return self.nonlinearity(activation)

def batch_norm(layer, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), **kwargs):
    """
    adapted from https://gist.github.com/f0k/f1a6bd3c8585c400c190
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    else:
        nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, b, g, nonlinearity=nonlinearity, **kwargs)

class GaussianNoiseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, use_last_noise=False, **kwargs):
        if deterministic or self.sigma == 0:
            return input
        else:
            if not use_last_noise:
                self.noise = self._srng.normal(input.shape, avg=0.0, std=self.sigma)
            return input + self.noise


# /////////// older code used for MNIST ////////////

# weight normalization
def l2normalize(layer, train_scale=True):
    W_param = layer.W
    s = W_param.get_value().shape
    if len(s)==4:
        axes_to_sum = (1,2,3)
        dimshuffle_args = [0,'x','x','x']
        k = s[0]
    else:
        axes_to_sum = 0
        dimshuffle_args = ['x',0]
        k = s[1]
    layer.W_scale = layer.add_param(lasagne.init.Constant(1.),
                          (k,), name="W_scale", trainable=train_scale, regularizable=False)
    layer.W = W_param * (layer.W_scale/T.sqrt(1e-6 + T.sum(T.square(W_param),axis=axes_to_sum))).dimshuffle(*dimshuffle_args)
    return layer

# fully connected layer with weight normalization
class DenseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, theta=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.),
                 weight_scale=lasagne.init.Constant(1.), train_scale=False, nonlinearity=relu, **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.num_units = num_units
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_units), name="theta")
        self.weight_scale = self.add_param(weight_scale, (num_units,), name="weight_scale", trainable=train_scale)
        self.W = self.theta * (self.weight_scale/T.sqrt(T.sum(T.square(self.theta),axis=0))).dimshuffle('x',0)
        self.b = self.add_param(b, (num_units,), name="b")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, init=False, deterministic=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)

        if init:
            ma = T.mean(activation, axis=0)
            activation -= ma.dimshuffle('x',0)
            stdv = T.sqrt(T.mean(T.square(activation),axis=0))
            activation /= stdv.dimshuffle('x',0)
            self.init_updates = [(self.weight_scale, self.weight_scale/stdv), (self.b, -ma/stdv)]
        else:
            activation += self.b.dimshuffle('x', 0)

        return self.nonlinearity(activation)


###
### Section (2):
### URL: https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_feature_matching.py
###

import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import plotting
import cifar10_data

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--count', default=400)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='/home/tim/data/cifar-10-python')
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load CIFAR-10
trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
testx, testy = cifar10_data.load(args.data_dir, subset='test')
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(testx.shape[0]/args.batch_size)

# specify generative model
noise_dim = (args.batch_size, 100)
noise = theano_rng.uniform(size=noise_dim)
gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size,512,4,4)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
gen_dat = ll.get_output(gen_layers[-1])

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=10, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
disc_params = ll.get_all_params(disc_layers, trainable=True)

# costs
labels = T.ivector()
x_lab = T.tensor4()
x_unl = T.tensor4()
temp = ll.get_output(gen_layers[-1], deterministic=False, init=True)
temp = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab, deterministic=False)
output_before_softmax_unl = ll.get_output(disc_layers[-1], x_unl, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)

l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
l_unl = nn.log_sum_exp(output_before_softmax_unl)
l_gen = nn.log_sum_exp(output_before_softmax_gen)
loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(l_unl)) + 0.5*T.mean(T.nnet.softplus(l_gen))

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

# test error
output_before_softmax = ll.get_output(disc_layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates) # data based initialization
train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,lr], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates+disc_avg_updates)
test_batch = th.function(inputs=[x_lab,labels], outputs=test_err, givens=disc_avg_givens)
samplefun = th.function(inputs=[],outputs=gen_dat)

# Theano functions for training the gen net
output_unl = ll.get_output(disc_layers[-2], x_unl, deterministic=False)
output_gen = ll.get_output(disc_layers[-2], gen_dat, deterministic=False)
m1 = T.mean(output_unl,axis=0)
m2 = T.mean(output_gen,axis=0)
loss_gen = T.mean(abs(m1-m2)) # feature matching loss
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=None, updates=gen_param_updates)

# select labeled data
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

# //////////// perform training //////////////
for epoch in range(1200):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))

    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]
    
    if epoch==0:
        print(trainx.shape)
        init_param(trainx[:500]) # data based initialization

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    for t in range(nr_batches_train):
        ran_from = t*args.batch_size
        ran_to = (t+1)*args.batch_size
        ll, lu, te = train_batch_disc(trainx[ran_from:ran_to],trainy[ran_from:ran_to],
                                      trainx_unl[ran_from:ran_to],lr)
        loss_lab += ll
        loss_unl += lu
        train_err += te
        
        train_batch_gen(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size],lr)

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train
    
    # test
    test_err = 0.
    for t in range(nr_batches_test):
        test_err += test_batch(testx[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
    test_err /= nr_batches_test

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, train_err, test_err))
    sys.stdout.flush()

    # generate samples from the model
    sample_x = samplefun()
    img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='CIFAR10 samples')
    plotting.plt.savefig("cifar_sample_feature_match.png")

    # save params
    #np.savez('disc_params.npz', *[p.get_value() for p in disc_params])
    #np.savez('gen_params.npz', *[p.get_value() for p in gen_params])
