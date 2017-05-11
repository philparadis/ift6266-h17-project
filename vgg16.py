# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

import numpy as np
from models import BaseModel, Conv_Deconv
import hyper_params
from utils import print_critical, print_error, print_warning, print_info, print_positive, log, logout

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class Lasagne_Conv_Deconv(BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_conv_deconv_hyper_params):
        super(VGG16_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)
        self.lasagne_model = None

    def build(self, batch_size, input_var=None):
        from lasagne.layers import InputLayer
        from lasagne.layers import DenseLayer
        from lasagne.layers import NonlinearityLayer
        from lasagne.layers import DropoutLayer
        from lasagne.layers import Pool2DLayer as PoolLayer
        from lasagne.nonlinearities import softmax
        def build_model(bs, input_var=None):
        net= {}
        net['input'] = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
        net['conv1'] = ConvLayer(net['input'], 64, 5, pad=0)
        net['pool1'] = PoolLayer(net['conv1'], 2)
        net['conv2'] = ConvLayer(net['pool1'], 128, 3, pad=0)
        net['pool2'] = PoolLayer(net['conv2'], 2)
        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=0)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=0)
        net['pool3'] = PoolLayer(net['conv3_2'], 2)
        net['conv4'] = ConvLayer(net['pool3'], 2048, 5, pad=0)
        net['conv4_drop'] = NonlinearityLayer(net['conv4'], nonlinearity=rescale)
        net['conv5'] = ConvLayer(net['conv4_drop'], 2048, 1, pad=0)
        net['conv5_drop'] = NonlinearityLayer(net['conv5'], nonlinearity=rescale)

        net['conv6'] = ConvLayer(net['conv5_drop'], 64, 1, pad=0, nonlinearity=sigmoid)
        net['conv7'] = ConvLayer(net['conv6'], 10, 1, pad=0, nonlinearity=softmax4d)

        return net

    def train(self, dataset):
        pass

def naive_compute_all(batch_size=50):

    input_var = T.tensor4('inputs')

    model = build_model(batch_size, input_var)

    prediction = lasagne.layers.get_output(model['conv6'],deterministic=True)
    prediction = prediction.reshape((batch_size, -1,))

    val_fn = theano.function([input_var], prediction)

    # ---compute feature for a single image---

    # img = load_bgr(sys.argv[3])
    # imgs = make_batch(np.array([img], dtype=np.uint8), batch_size)
    # imgs = imgs.astype(np.float32) / 255
    # feat = val_fn(imgs)
    #
    # feat_part = (feat>0.5).astype(np.uint8)[0]

    # ---compute feature for all imgs---
    t = time.time()
    imgs = dense_sample(img)
    imgs = make_batch(imgs, batch_size)
    imgs = imgs.astype(np.float32) / 255

    feat = np.zeros((imgs.shape[0], 64))
    for i in range(imgs.shape[0] / batch_size):
        ret = val_fn(imgs[i*batch_size:(i+1)*batch_size])
        feat[i*batch_size:(i+1)*batch_size] = ret
    print time.time() - t
    feat_all = (feat>0.5).astype(np.uint8)

    # for i in range(feat_all.shape[0]):
    #     if np.bitwise_xor(feat_all[i], feat_part).sum() <= 3:
    #         print (i / 208) * 2, (i%208) * 2


    w = len(range(0, W-64, 2))
    compact = []
    for i in range(feat_all.shape[0]):
        compact.append((pack(feat_all[i]), (i/w)*2, (i%w)*2))
    compact = np.array(compact, dtype=np.uint64)
    # compact = np.unique(compact)
    # print compact.shape
    return compact


def fc_compute_all(batch_size=1):
    net = build_model(batch_size, None)

    input_var = T.tensor4('inputs')
    output_map = [[], []]
    label_map = [[], []]
    shape_map = [[], []]
    prv = 0
    cur = 1
    output_map[prv] = [input_var]
    label_map[prv] = [("", "")]
    #shape_map[prv] = [(1, 3, H , W)]

    for l in lasagne.layers.get_all_layers(net['conv6']): # stop at conv6
        if isinstance(l, InputLayer):
            continue

        output_map[cur] = []
        label_map[cur] = []
        #shape_map[cur] = []

        if isinstance(l, ConvLayer):
            for j in range(len(output_map[prv])):
                i = output_map[prv][j]
                s,t = label_map[prv][j]
                #sp = shape_map[prv][j]
                o = T.nnet.conv2d(i,
                                  l.W,
                                  #sp,
                                  #l.get_W_shape(),
                                  subsample=(1, 1),
                                  border_mode='valid',
                                  filter_flip=False)
                o = l.nonlinearity(o + l.b.dimshuffle('x', 0, 'x', 'x'))

                output_map[cur].append(o)
                label_map[cur].append((s, t))
                #shape_map[cur].append(l.get_output_shape_for(sp))

        elif isinstance(l, PoolLayer):
            for j in range(len(output_map[prv])):
                i = output_map[prv][j]
                s,t = label_map[prv][j]
                #sp = shape_map[prv][j]

                o = pool_2d(i,
                            ds=(2,2),
                            st=(2,2),
                            ignore_border=l.ignore_border,
                            padding=(0,0),
                            mode='max',
                            )

                output_map[cur].append(o)
                label_map[cur].append(('0'+s, '0'+t))
                #shape_map[cur].append((sp[0], sp[1], sp[2]/2, sp[3]/2))

                output_map[cur].append(o[:,:, 1:, :])
                label_map[cur].append(('1'+s, '0'+t))
                #shape_map[cur].append((sp[0], sp[1], sp[2]/2-1, sp[3]/2))

                output_map[cur].append(o[:,:, :, 1:])
                label_map[cur].append(('0'+s, '1'+t))
                #shape_map[cur].append((sp[0], sp[1], sp[2]/2, sp[3]/2-1))

                output_map[cur].append(o[:,:, 1:, 1:])
                label_map[cur].append(('1'+s, '1'+t))
                #shape_map[cur].append((sp[0], sp[1], sp[2]/2-1, sp[3]/2-1))

        elif isinstance(l, NonlinearityLayer): # rescale for dropout
            for j in range(len(output_map[prv])):
                i = output_map[prv][j]
                s,t = label_map[prv][j]
                #sp = shape_map[prv][j]

                o = rescale(i)
                output_map[cur].append(o)
                label_map[cur].append((s, t))
                #shape_map[cur].append(sp)

        prv = 1-prv
        cur = 1-cur

    # print len(output_map[prv])
    fc_func = theano.function([input_var], output_map[prv])

    img_large = to_bgr(img)
    img_large = np.array([img_large], dtype=np.float32) / 255

    data = img_large
    for i in range(batch_size-1):
        data = np.concatenate((data, img_large), axis=0)


    t = time.time()
    res = fc_func(data)
    print time.time() - t


    feat_all = []
    for r in res:
        s = r.shape
        r = (r>0.5).astype(np.uint8)
        t = np.zeros((s[2], s[3]), dtype=np.uint64)

        for i in range(s[2]):
            for j in range(s[3]):
                t[i][j] = pack(r[0,:,i,j].reshape((s[1],)))

        feat_all.append(t)

    return feat_all,label_map[prv]

    
class VGG16_Model(BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_vgg16_hyper_params):
        super(VGG16_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)
        self.conv_deconv_model = None
        self.vgg16_model = None
        self.lasagne_model = None
        
    def build(self):
        import lasagne
        from lasagne.layers import InputLayer
        from lasagne.layers import DenseLayer
        from lasagne.layers import NonlinearityLayer
        from lasagne.layers import DropoutLayer
        from lasagne.layers import Pool2DLayer as PoolLayer
        from lasagne.nonlinearities import softmax
        import cPickle as pickle

        try:
            from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
        except ImportError as e:
            from lasagne.layers import Conv2DLayer as ConvLayer
            print_warning("Cannot import 'lasagne.layers.dnn.Conv2DDNNLayer' as it requires GPU support and a functional cuDNN installation. Falling back on slower convolution function 'lasagne.layers.Conv2DLayer'.")

            
        log("Building VGG-16 model...")
        net = {}
        net['input'] = InputLayer((None, 3, 224, 224))
        net['conv1_1'] = ConvLayer(
            net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = ConvLayer(
            net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(
            net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = ConvLayer(
            net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(
            net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = ConvLayer(
            net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = ConvLayer(
            net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        net['conv4_1'] = ConvLayer(
            net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = ConvLayer(
            net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = ConvLayer(
            net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        net['conv5_1'] = ConvLayer(
            net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = ConvLayer(
            net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = ConvLayer(
            net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
        net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8'] = DenseLayer(
            net['fc7_dropout'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)
        net_output = net['prob']

        log("Loading VGG16 pre-trained weights from file 'vgg16.pkl'...")
        with open('vgg16.pkl', 'rb') as f:
            params = pickle.load(f)

        #net_output.initialize_layers()
        lasagne.layers.set_all_param_values(net['prob'], params['param values'])

        self.lasagne_model = net

        return net

    def train(self, dataset):
        from lasagne.layers import InputLayer, DenseLayer
        import lasagne
        from lasagne.updates import sgd, total_norm_constraint
        import theano.tensor as T

        x = T.matrix()
        y = T.ivector()
        l_in = InputLayer((5, 10))
        l1 = DenseLayer(l_in, num_units=7, nonlinearity=T.nnet.softmax)
        output = lasagne.layers.get_output(l1, x)
        cost = T.mean(T.nnet.categorical_crossentropy(output, y))
        all_params = lasagne.layers.get_all_params(l1)
        all_grads = T.grad(cost, all_params)
        scaled_grads = total_norm_constraint(all_grads, 5)
        updates = sgd(scaled_grads, all_params, learning_rate=0.1)
