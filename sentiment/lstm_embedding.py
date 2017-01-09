# coding=utf-8
'''
Build a tweet sentiment analyzer
'''

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb
import text_process as text
# from gensim.models import word2vec
from gensim.models import Word2Vec

# datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}
datasets = {'imdb': (text.load_data, imdb.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

wemb = []
word_index = []


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):


    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    # 根据源代码的注释，embedding，claffier和lstm层的参数不是一起初始化的，这里先初始化embedding，claffier的参数

    # 将所有的参数放在一个名为params的OrderedDict中
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])

    # 随机生成embedding矩阵，这里为10000 * 128维的，因为词典大小是10000，也就是说，词的ID范围是1-10000，我们将每个词转换成一个128维的向量，
    # 所以这里生成了一个10000*128的矩阵，每个词转换成它的ID的那一行的128维向量。比如“我”这个词的ID是5，那么“我”这个词就用params['Wemb']矩阵的第5行表示，
    # 第5行就是一个128维的向量，这里用随机生成的矩阵表示，作为示例。（这是下边用到的，这里先给出解释）
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options, params, prefix=options['encoder'])

    # classifier
    # 初始化softmax分类器的参数
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'], options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns
    # layers = {'lstm':(param_init_lstm, lstm_layer)}
    # 我们还记得options['encoder']只能为lstm，这里返回了layers['lstm']的第一项param_init_lstm函数：


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)  # w： (128,128)

    # u： (128,128)
    # s： (128,)
    # v： (128,128)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    # 这里初始化了LSTM的参数，_p()这个函数是连接变量名的，
    # ortho_weight()函数用来生成正交的矩阵，先生成n*n的矩阵，做svd分解，这里不再细述。
    # 将4个矩阵列连接起来是为了方便运算，在计算门的时候，一步到位。
    # 为什么要先生成正交矩阵？为什么要做SVD分解？

    # W: (128, 512)
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)

    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


# (tparams, emb, options,prefix=options['encoder'],mask=mask)
def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    # state_below是输入x和w，b计算后的输入节点。同上，第一维代表step
    # state_below，这是个3D矩阵，[n_Step，BatchSize，Emb_Dim]，这是一个mini_batch
    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        # 如果输入三维的x，那么样本数就是第二维的长度，否则就是只有一个样本
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask非None
    assert mask is not None

    # 切片，计算的时候是几个门一起计算，切片将各个门的值分开
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # 计算T时刻
    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        # 隐藏层计算，i：输入门；f：忘记门；o：输出门；c：cell。
        # h代表隐藏层的输出，h_和c_分别代表上一个时刻的cell和隐藏层输出，m_就是mask，它记录了变换后的x的非零值的位置。下文详细介绍
        # 根据LSTM隐藏层的计算公式，state_below就是input_node
        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        # 这两句的意思是如果某个样本到一定的状态后没有值了，也就是mask矩阵对应位置的值为0了，那么它的c和h就保持上一个时刻的不变。
        # 结合prepare_data函数理解。
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    # 这里相当于计算input_node
    # 这一步把 wx' = wx + b 提到外面做，并行处理，，那么在里面只需做 i = sigmoid(wx') 就可以
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']

    # scan函数，进行迭代的函数是_step，它的输入是m_, x_, h_, c_，迭代的是sequences中的mask，state_below,
    # 每次拿出他们的一行，作为输入的m_和x_，h_和c_的初始值设置为0（outputs_info设置初始值），
    # 每次计算，_step返回的h和c分别赋给下次迭代的h_和c_，迭代次数为nsteps，这样就实现了隐藏层节点的传递，最后函数返回h和c给rval。
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),  # h
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),  # c
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)

    # 整个函数最后返回rval[0]也就是最后的h
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent  随机梯度下降

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def AdaDelta(tparams, grads):
    p = 0.95
    e = 1e-6
    # init
    delta_x2 = [theano.shared(p.get_value() * numpy_floatX(0.)) for k, p in tparams.iteritems()]
    g2 = [theano.shared(p.get_value() * numpy_floatX(0.)) for k, p in tparams.iteritems()]
    # first to update g2
    update_g2 = [(g2, p * g2 + (1 - p) * (g ** 2)) for g2, g in zip(g2, grads)]
    fn_update_1 = theano.function(inputs=[], updates=update_g2)

    # calc delta_x by RMS
    delta_x = [-tensor.sqrt(delta_x2_last + e) / tensor.sqrt(g2_now + e) * g for g, delta_x2_last, g2_now in
               zip(grads, delta_x2, g2)]
    # then to update delta_x2 and param
    update_delta_x2 = [(delta_x2, p * delta_x2 + (1 - p) * (delta_x ** 2)) for delta_x2, delta_x in
                       zip(delta_x2, delta_x)]
    update_param = [(param, param + delta) for param, delta in zip(tparams.values(), delta_x)]
    fn_update_2 = theano.function(inputs=[], updates=update_delta_x2 + update_param)
    # return the update function of theano
    return fn_update_1, fn_update_2


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable  共享变量
        Initial learning rate
    tpramas: Theano SharedVariable  共享变量
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    # for k, p in tparams.items()： k 表示参数名称，p表示变量值
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    # 更新共享变量
    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    # 更新学习率
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    # 随机数生成器
    trng = RandomStreams(SEED)

    # 是否用dropout
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    # 为x,mask,y生成占位符号
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]  # x的行代表steps（经过变换）
    n_samples = x.shape[1]  # x的列代表不同的样本，相当于batch_size

    # 将词用向量表示，原始输入进来的x都是词序号，通过下面这一步将其转换成对应的词向量
    # x.flatten()：将x矩阵展平成向量
    # emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
    #                                             n_samples,
    #                                             options['dim_proj']])

    # 方法一不行
    # xflatten = x.flatten()
    # for xx in xflatten:
    #     print (xx)
    #
    # emb = numpy.array([[wemb[xx]] for xx in xflatten])
    # emb = emb.reshape([n_timesteps, n_samples, options['dim_proj']])
    #

    print (type(wemb))
    emb = wemb[x.flatten()].reshape([n_timesteps, n_samples, options['dim_proj']])

    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    # 计算样本的每个时刻的h的均值，对列求和，列就是样本的step，如果这个状态有值，那么相应的mask值为1，否则就是0，
    # 然后除以mask每列的和，也就是样本一共的step个数，求出平均值
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]

    # 如果dropout，就调用dropout_layer()随机丢弃一些隐藏层
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    print("[proj] ", proj.max, proj.min)

    # 预测就是隐藏层h的均值输入到softmax函数得到的
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    # 将预测输出编译成x和mask的函数
    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    # 损失函数
    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err


def train_lstm(
        dim_proj=128,  # word embedding的维数和隐藏层的维数，用默认值。（word embedding是一种将一个词转成一个向量的过程，这里不去深究）
        patience=10,  # 该参数用于earlystop，如果10轮迭代的误差没有降低，就进行earlystop
        max_epochs=4000,  # 迭代次数（将训练集迭代一轮为一个epoch, 每个迭代周期迭代所有的训练集）
        dispFreq=10,  # 每更新10次显示训练过程，即显示训练、验证和测试误差
        decay_c=0.,  # 参数U的正则权重，U为隐藏层ht到输出层的参数
        lrate=0.0001,  # sgd用的学习率
        n_words=34000,  # 词典大小，用于数据预处理部分，将词用该词在词典中的ID表示，超过10000的用1表示，仅仅用于数据，不做深究
        optimizer=sgd,  # 优化方法，代码提供了sgd,adadelta和rmsprop三种方法，采用了adadelta.
        encoder='lstm',  # 一个标识符，可以去掉，但是必须是lstm
        saveto='model/lstm_model.npz',  # 保存最好模型的文件，保存训练误差，验证误差和测试误差等等
        validFreq=370,  # 验证频率 Compute the validation error after this number of update.
        saveFreq=1110,  # 保存频率 Save the parameters after every saveFreq updates
        maxlen=200,  # 序列的最大长度，超出长度的数据被抛弃，见数据处理部分
        batch_size=16,  # 训练的batch大小.
        valid_batch_size=64,  # 验证集用的*batch大小.
        dataset='imdb',  # 用于数据预处理的参数，全局变量datasets的key'imdb'的value为两个处理数据的函数

        # Parameter for extra option
        noise_std=0.,  # 后边好像没有出现过，
        use_dropout=True,  # 控制dropout，不用dropout的话运行速度较快，但是效果不好，dropout不太好解释，以一定的概率随机丢弃模型中的一些节点，
        # 这样可以综合多种模型的结果，进行投票。需要自行补充deeplearning的知识

        reload_model=None,  # 加载模型参数的文件，用于已训练好的模型，或保存的中间结果
        test_size=-1,  # 测试集大小，如果为正，就只用这么多测试样本
):
    # Model options
    # 首先将当先的函数局部作用于的参数copy到字典model_options中，后面的很多函数就以model_options作为参数进行参数传递。
    model_options = locals().copy()  # 它将函数中所有参数复制，保存为一个词典
    print("model options", model_options)

    # 返回了两个函数：load_data,prepare_data这两个函数定义在imdb.py中
    # 数据已经事先存在了imdb.pkl中，这里用pickle方法load进来，第一项为训练数据，第二项为测试数据；
    # load_data函数将数据集读入，舍弃长度超过maxlen的数据，并按照参数valid_portion的比例将一部分训练集划为验证集。
    # 而第二个函数prepare_data负责数据的转换，在训练和测试的时候先将训练数据和测试数据的横轴和纵轴调换，并使数据维度保持一致，后面详细讲
    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    # train, valid, test = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)
    # train 格式：(x集合，y集合)， 其中x集合 = [word1索引，word2索引，word3索引]
    train, valid = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    # if test_size > 0:
    #     # The test set is sorted by size, but we want to keep random
    #     # size example.  So we must select a random selection of the
    #     # examples.
    #     idx = numpy.arange(len(test[0]))
    #     numpy.random.shuffle(idx)
    #     idx = idx[:test_size]
    #     test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    # 如果我们设置了test_size的大小，这个步骤就是从测试集中随机找test_size个作为测试数据，
    # 如果没有设置test_size,会用所有的测试集数据做测试。
    # 原来的测试数据是根据长度排列的（imdb数据自身特点），这里做了一次打散

    # ydim为标签y的维数，因为是从0开始的，所以后面+1，并将它加入模型参数中
    ydim = numpy.max(train[1]) + 1

    model_options['ydim'] = ydim

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    # 模型建立阶段，首先初始化各种参数，调用了全局作用域的函数init_params()
    params = init_params(model_options)

    global wemb
    global word_index

    wemb, word_index = text.init_embedding()
    wemb = theano.shared(wemb, name="wemb")
    # c = wemb.get_value
    # for w in c:
    #     print (w)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    # 建立模型
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay  # 如果加入正则，损失函数加上L2损失

    # 编译损失函数
    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    # 求导，并编译求导函数
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    # 优化
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)

    print('Optimization')
    # 将验证集和测试机分成batchs：get_minibatchs_idx, 返回batchID和对应的样本序号，省略
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    # kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    # print("%d test examples" % len(test[0]))

    # 记录误差、最好的结果，和bad_count计数
    history_errs = []
    best_p = None
    bad_count = 0

    # 如果未设置验证频率和保存频率，那么就设置为一个epoch，len（train[0]）/ batch_size就是一个epoch
    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            # 得到训练数据的mini_batchs
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1  # 更新次数+1
                use_noise.set_value(1.)  # 设置drop_out

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                # print (x.shape)
                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                # 判断是否到了显示频率
                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    # 判断是否到了保存频率，并保存params
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                # 判断是否到了验证频率，到了就计算各种误差，并更新best_p
                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    # test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    # history_errs.append([valid_err, test_err])
                    history_errs.append(valid_err)

                    # if best_p is None or valid_err <= numpy.array(history_errs)[:, 0].min():
                    if best_p is None or valid_err <= min(history_errs):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    print('Train ', train_err, 'Valid ', valid_err, 'Test ', 'test_err')

                    # 每验证一次，记录下验证误差和测试误差，
                    # 如果当前的验证误差大于前10(patience)次验证误差的最小值（也就是说误差没有降低），bad_counter += 1，
                    # 如果bad_counter>patience，就early stop
                    print("\n")
                    print(len(history_errs))
                    print(patience)
                    print("\n")
                    # if len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience, 0].min():
                    if len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    # test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', 'test_err')
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    # valid_err=valid_err, test_err=test_err,
                    valid_err=valid_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print(('Training took %.1fs' %
           (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err  # , test_err


###########################################################################
###########################################################################
###########################################################################

def predict(
        dim_proj=128,  # word embedding的维数和隐藏层的维数，用默认值。（word embedding是一种将一个词转成一个向量的过程，这里不去深究）
        patience=10,  # 该参数用于earlystop，如果10轮迭代的误差没有降低，就进行earlystop
        max_epochs=4000,  # 迭代次数（将训练集迭代一轮为一个epoch, 每个迭代周期迭代所有的训练集）
        dispFreq=10,  # 每更新10次显示训练过程，即显示训练、验证和测试误差
        decay_c=0.,  # 参数U的正则权重，U为隐藏层ht到输出层的参数
        lrate=0.0001,  # sgd用的学习率
        n_words=34000,  # 词典大小，用于数据预处理部分，将词用该词在词典中的ID表示，超过10000的用1表示，仅仅用于数据，不做深究
        optimizer=adadelta,  # 优化方法，代码提供了sgd,adadelta和rmsprop三种方法，采用了adadelta.
        encoder='lstm',  # 一个标识符，可以去掉，但是必须是lstm
        saveto='model/lstm_model.npz',  # 保存最好模型的文件，保存训练误差，验证误差和测试误差等等
        validFreq=370,  # 验证频率 Compute the validation error after this number of update.
        saveFreq=1110,  # 保存频率 Save the parameters after every saveFreq updates
        maxlen=100,  # 序列的最大长度，超出长度的数据被抛弃，见数据处理部分
        batch_size=16,  # 训练的batch大小.
        valid_batch_size=64,  # 验证集用的*batch大小.
        dataset='imdb',  # 用于数据预处理的参数，全局变量datasets的key'imdb'的value为两个处理数据的函数
        noise_std=0.,  # 后边好像没有出现过，
        use_dropout=True,  # 控制dropout，不用dropout的话运行速度较快，但是效果不好，dropout不太好解释，以一定的概率随机丢弃模型中的一些节点，
        # 这样可以综合多种模型的结果，进行投票。需要自行补充deeplearning的知识

        reload_model=None,  # 加载模型参数的文件，用于已训练好的模型，或保存的中间结果
        test_size=-1,  # 测试集大小，如果为正，就只用这么多测试样本
):
    # Model options
    # 首先将当先的函数局部作用于的参数copy到字典model_options中，后面的很多函数就以model_options作为参数进行参数传递。
    model_options = locals().copy()  # 它将函数中所有参数复制，保存为一个词典
    print("model options", model_options)

    # 返回了两个函数：load_data,prepare_data这两个函数定义在imdb.py中
    # 数据已经事先存在了imdb.pkl中，这里用pickle方法load进来，第一项为训练数据，第二项为测试数据；
    # load_data函数将数据集读入，舍弃长度超过maxlen的数据，并按照参数valid_portion的比例将一部分训练集划为验证集。
    # 而第二个函数prepare_data负责数据的转换，在训练和测试的时候先将训练数据和测试数据的横轴和纵轴调换，并使数据维度保持一致，后面详细讲
    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    # train 格式：(x集合，y集合)， 其中x集合 = [word1索引，word2索引，word3索引]
    train, valid = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    # 如果我们设置了test_size的大小，这个步骤就是从测试集中随机找test_size个作为测试数据，
    # 如果没有设置test_size,会用所有的测试集数据做测试。
    # 原来的测试数据是根据长度排列的（imdb数据自身特点），这里做了一次打散

    # ydim为标签y的维数，因为是从0开始的，所以后面+1，并将它加入模型参数中
    ydim = numpy.max(train[1]) + 1

    model_options['ydim'] = ydim

    print('Building model')
    # 模型建立阶段，首先初始化各种参数，调用了全局作用域的函数init_params()
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    tparams = init_tparams(params)

    # 建立模型
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    start_time = time.time()

    kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

    x = train[0]
    y = train[1]

    x, mask, y = prepare_data(x, y)
    print(x.shape)

    res = pred_probs(f_pred_prob, prepare_data, train, kf, True)
    print("【RES】", res.shape)

    for r in res:
        print("%s + %s = %f" % (r[0], r[1], float(r[0]) + float(r[1])))

    emb = tparams["Wemb"].get_value()
    print(type(emb), emb.shape)
    em = emb[:10]
    print(type(em))

    for e in em:
        print(type(e))
        print(e)

    import text_process
    dic = [d for d in text_process.dict.keys()]
    print(len(x.flatten()))
    # doc = dic[x.flatten()]
    # for d in doc:
    #     print (d)


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    # train_lstm(
    #     max_epochs=100,
    #     test_size=500,
    # )
    train_lstm(max_epochs=100)
    # predict()
