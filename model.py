import numpy as np

import tensorflow as tf
import tensorlayer as tl

# CNN based on specified depth, width and hyper-parameters
def deepnn(x, fsize, ldepth, lwidth, bn=True, act=0, itype=4, tb=True, training=True):
  B = 3
  keep_prob = tf.placeholder(tf.float32)
  network = x

  blockc = (np.floor(ldepth[0]/B)*np.ones(B)).astype('int64')
  for c in range(B-1,0,-1):
    if sum(blockc) < ldepth[0]:
      blockc[c] += 1

  assert(sum(blockc) == ldepth[0])

  for c in range(B):
    sz = network.get_shape().as_list()
    w = lwidth[0]*np.power(2,c)
    network = conv2d(network, [fsize, fsize, sz[3], w], 'conv%d1'%(c+1), bn, act, itype, tb, training)

    for b in range(blockc[c]-1):
      network = conv2d(network, [fsize, fsize, w, w], 'conv%d%d'%(c+1,b+2), bn, act, itype, tb, training)
    network = max_pool_2x2(network, 'pool%d'%(c+1))

  sz = network.get_shape().as_list()
  network = tf.reshape(network, [-1, sz[1]*sz[2]*sz[3]])
  conv_out = network

  for c in range(ldepth[1]):
    sz = network.get_shape().as_list()
    w = (lwidth[1]*np.power(2.0,-c)).astype('int64')
    network = fc(network, [sz[1], w], 'fc%d'%(c+1), bn, act, itype, tb, training)
    network = tf.nn.dropout(network, keep_prob)

  sz = network.get_shape().as_list()
  network = fc(network, [sz[1], 20], 'fc_final', False, -1, itype, tb, training)

  return network, keep_prob, conv_out


# 1D CNN for meta-classification
def cnn1D(x, C=10, bn=True, tb=True, training=True):
  with tf.name_scope('encoder'):
    network = conv1d(x, [5, 1, 8], 'conv11', bn, tb, training)
    network = max_pool_2x1(network, 'pool1')

    network = conv1d(network, [5, 8, 16], 'conv21', bn, tb, training)
    network = max_pool_2x1(network, 'pool2')

    network = conv1d(network, [5, 16, 32], 'conv31', bn, tb, training)
    network = max_pool_2x1(network, 'pool3')

    network = conv1d(network, [5, 32, 64], 'conv41', bn, tb, training)
    network = max_pool_2x1(network, 'pool4')

    network = conv1d(network, [5, 64, 128], 'conv51', bn, tb, training)
    network = max_pool_2x1(network, 'pool5')

    network = conv1d(network, [5, 128, 128], 'conv61', bn, tb, training)
    network = max_pool_2x1(network, 'pool6')

    network = conv1d(network, [5, 128, 128], 'conv71', bn, tb, training)
    network = max_pool_2x1(network, 'pool7')

    network = conv1d(network, [5, 128, 128], 'conv81', bn, tb, training)
    network = max_pool_2x1(network, 'pool8')

    network = conv1d(network, [5, 128, 128], 'conv91', bn, tb, training)
    network = max_pool_2x1(network, 'pool9')

    network = conv1d(network, [5, 128, 128], 'conv101', bn, tb, training)
    network = conv1d(network, [5, 128, 256], 'conv102', bn, tb, training)
    network = max_pool_2x1(network, 'pool10')

    network = conv1d(network, [5, 256, 256], 'conv111', bn, tb, training)
    network = conv1d(network, [5, 256, 256], 'conv112', bn, tb, training)
    network = max_pool_2x1(network, 'pool11')

    network = conv1d(network, [5, 256, 256], 'conv121', bn, tb, training)
    network = conv1d(network, [5, 256, 256], 'conv122', bn, tb, training)
    network = max_pool_2x1(network, 'pool12')

    sz = network.get_shape().as_list()
    print('Final pooled size:')
    print(sz)

    network = tf.reshape(network, [-1, sz[1]*sz[2]])
    print('First FC size:')
    print(network.get_shape().as_list())

    #network = tf.reshape(network, [-1, 5760])
    #print(network.get_shape().as_list())

    keep_prob = tf.placeholder(tf.float32)
    act = 0
    itype = 3

    network = fc(network, [sz[1]*sz[2], 1024], 'fc1', bn, act, itype, tb, training)
    network = tf.nn.dropout(network, keep_prob)
    network = fc(network, [1024, 1024], 'fc2', bn, act, itype, tb, training)
    network = tf.nn.dropout(network, keep_prob)
    network = fc(network, [1024, 1024], 'fc3', bn, act, itype, tb, training)
    network = tf.nn.dropout(network, keep_prob)
    network = fc(network, [1024, 1024], 'fc4', bn, act, itype, tb, training)
    network = tf.nn.dropout(network, keep_prob)
    network1 = fc(network, [1024, 64], 'fc5', bn, act, itype, tb, training)
    network = tf.nn.dropout(network1, keep_prob)

    network = fc(network, [64, C], 'fc_final', False, -1, itype, tb, training)

  return network, network1, keep_prob


# 1D CNN for meta-classification
def cnn1D_small(x, C=10, bn=True, tb=True, training=True):
  with tf.name_scope('encoder'):
    network = conv1d(x, [5, 1, 8], 'conv11', bn, tb, training)
    network = max_pool_2x1(network, 'pool1')

    network = conv1d(network, [5, 8, 16], 'conv21', bn, tb, training)
    network = max_pool_2x1(network, 'pool2')

    network = conv1d(network, [5, 16, 32], 'conv31', bn, tb, training)
    network = conv1d(network, [5, 32, 64], 'conv32', bn, tb, training)
    network = max_pool_2x1(network, 'pool3')

    network = conv1d(network, [5, 64, 128], 'conv41', bn, tb, training)
    network = conv1d(network, [5, 128, 256], 'conv42', bn, tb, training)

    for i in [5,6,7,8,9,10]:
      sz = network.get_shape().as_list()
      if sz[1] > 50:
        print('\tsz = %d, pool%d'%(sz[1], i-1))
        network = max_pool_2x1(network, 'pool%d'%(i-1))
      
      network = conv1d(network, [5, 256, 256], 'conv%d1'%i, bn, tb, training)
    
    network = max_pool_2x1(network, 'pool_final')

    sz = network.get_shape().as_list()
    print('Final pooled size:')
    print(sz)

    network = tf.reshape(network, [-1, sz[1]*sz[2]])
    print('First FC size:')
    print(network.get_shape().as_list())

    keep_prob = tf.placeholder(tf.float32)

    act = 0
    itype = 3

    network = fc(network, [sz[1]*sz[2], 1024], 'fc1', bn, act, itype, tb, training)
    network = tf.nn.dropout(network, keep_prob)
    network = fc(network, [1024, 1024], 'fc2', bn, act, itype, tb, training)
    network = tf.nn.dropout(network, keep_prob)
    network = fc(network, [1024, 1024], 'fc3', bn, act, itype, tb, training)
    network = tf.nn.dropout(network, keep_prob)
    network = fc(network, [1024, 1024], 'fc4', bn, act, itype, tb, training)
    network = tf.nn.dropout(network, keep_prob)
    network1 = fc(network, [1024, 64], 'fc5', bn, act, itype, tb, training)
    network = tf.nn.dropout(network1, keep_prob)

    network = fc(network, [64, C], 'fc_final', False, -1, itype, tb, training)

  return network, network1, keep_prob


# Number of weights of a network
def count_all_vars():
  N = 0
  for v in tf.global_variables():
    N += np.prod(v.shape.as_list())

  return N


# Export weights to file
def print_all_vars(outfile):
  model_vars = [var for var in tf.global_variables() if 'optimizer' not in var.name]
  N = 0

  for v in model_vars:
    N += np.prod(v.shape.as_list())

  arr = np.empty(N, dtype=np.float64)
  p = 0
  for v in model_vars:
    var = v.eval()
    
    sh = v.shape
    if len(sh) > 1:
      for i in range(sh[-1]):
        if len(sh) > 2:
          for j in range(sh[-2]):
            var_sub = var[:,:,j,i]
            var_sub = np.reshape(var_sub, var_sub.size)
            arr[p:p+len(var_sub)] = var_sub
            p += len(var_sub)
        else:
          var_sub = var[:,i]
          arr[p:p+len(var_sub)] = var_sub
          p += len(var_sub)
    else:
      var = np.reshape(var, var.size)

      for j in range(len(var)):
        arr[p] = var[j]
        p += 1

  arr.tofile(outfile)


# Import weights from file
def load_vars(infile, sess):
  if type(infile) == str:
    arr = np.fromfile(infile)
    print("%d weights read from %s.\n"%(arr.size, infile))
  else:
    arr = infile

  model_vars = [var for var in tf.global_variables() if 'optimizer' not in var.name]

  # Assign weights
  p = 0
  for v in model_vars:
    sh = v.shape.as_list()
    var = np.zeros(sh)
    if len(sh) > 1:
      for i in range(sh[-1]):
        if len(sh) > 2:
          for j in range(sh[-2]):
            l = sh[0]*sh[1]
            var[:,:,j,i] = np.reshape(arr[p:p+l], sh[:2])
            p += l
        else:
          var[:,i] = arr[p:p+sh[0]]
          p += sh[0]
    else:
      var = arr[p:p+sh[0]]
      p += sh[0]
      
    sess.run(tf.assign(v, var))

  print('Assigned in total %d weights to model\n'%p)
  
  return arr


#======= Layers =============================================

def conv1d(x, s, nmn, bn=True, tb=True, training=True):
  W = weight_variable('%s_W'%nmn, s, 3, tb)
  b = bias_variable('%s_b'%nmn, [s[2]], 3, tb)
  h = tf.nn.conv1d(x, W, stride=1, padding='SAME') + b

  h = tf.nn.relu(h)

  if bn:
    h = tl.layers.BatchNormLayer(tl.layers.InputLayer(h, name='%s_bn_in'%nmn), is_train=training, name='%s_bn'%nmn).outputs

  return h


def conv2d(x, s, nmn, bn=True, act=0, itype=0, tb=True, training=True):
  W = weight_variable('%s_W'%nmn, s, itype, tb)
  b = bias_variable('%s_b'%nmn, [s[3]], itype, tb)
  h = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b

  if bn:
    h = tl.layers.BatchNormLayer(tl.layers.InputLayer(h, name='%s_bn_in'%nmn), is_train=training, name='%s_bn'%nmn).outputs

  if act==0:
    h = tf.nn.relu(h)
  elif act==1:
    h = tf.nn.elu(h)
  elif act==2:
    h = tf.nn.sigmoid(h)
  elif act==3:
    h = tf.nn.tanh(h)

  return h


def max_pool_2x2(x, nmn):
  with tf.name_scope(nmn):
    h = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  return h


def max_pool_2x1(x, nmn):
  with tf.name_scope(nmn):
    h = tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding='VALID')
  return h


def fc(x, s, nmn, bn=True, act=0, itype=0, tb=True, training=True):
  W = weight_variable('%s_W'%nmn, s, itype, tb)
  b = bias_variable('%s_b'%nmn, [s[1]], itype, tb)

  h = tf.matmul(x, W) + b

  if act==0:
    h = tf.nn.relu(h)
  elif act==1:
    h = tf.nn.elu(h)
  elif act==2:
    h = tf.nn.sigmoid(h)
  elif act==3:
    h = tf.nn.tanh(h)

  if bn:
    h = tl.layers.BatchNormLayer(tl.layers.InputLayer(h, name='%s_bn_in'%nmn), is_train=training, name='%s_bn'%nmn).outputs

  return h


def weight_variable(nmn, shape, itype, tb=True):
  if itype==0:
    v = tf.get_variable(nmn, shape, initializer=tf.constant_initializer(value=0.1), trainable=tb)
  elif itype==1:
    v = tf.get_variable(nmn, shape, initializer=tf.random_normal_initializer, trainable=tb)
  elif itype==2:
    v = tf.get_variable(nmn, initializer=tf.glorot_uniform_initializer()(shape), trainable=tb)
  elif itype==3:
    v = tf.get_variable(nmn, initializer=tf.glorot_normal_initializer()(shape), trainable=tb)

  return v


def bias_variable(nmn, shape, itype, tb=True):
  return tf.get_variable(nmn, shape, initializer=tf.initializers.zeros, trainable=tb)


