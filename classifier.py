import sys, os, shutil, random, time
import numpy as np
import scipy.misc as sm
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import model, util
import csv

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "/mnt/raid/dnn_weights/data", "Path to input data directory")
tf.flags.DEFINE_string("data_nmn", None, "Input data name")
tf.flags.DEFINE_string("log_dir", "/mnt/raid/dnn_weights/output", "Path to output directory")
tf.flags.DEFINE_integer("id", 1, "ID of this training")
tf.flags.DEFINE_float("transf_scaling", 3.0, "Scaling of transformation perturbations")
tf.flags.DEFINE_integer("epochs", 1000, "Maximum number of epochs (early-stopping is used)")
tf.flags.DEFINE_bool("shuffle_epochs", True, "Re-shuffle at each epoch")
tf.flags.DEFINE_float("dropout", 0.5, "Drop-out before final output")
tf.flags.DEFINE_integer("early_stopping", 5, "Early stopping tolerance")
tf.flags.DEFINE_integer("prints", 20, "Number of weight export occasions")
tf.flags.DEFINE_list("params", None, "Specify explicit parameters (not choosen randomly)")
tf.flags.DEFINE_bool("fixed", False, "Fixed architecture")


if FLAGS.id >= 0:
  log_dir = '%s/%04d'%(FLAGS.log_dir, FLAGS.id)
else:
  log_dir = FLAGS.log_dir

if not os.path.exists(log_dir):
  os.makedirs(log_dir)
if not os.path.exists("%s/weights"%log_dir):
  os.makedirs("%s/weights"%log_dir)
if not os.path.exists("%s/weights_tmp"%log_dir):
  os.makedirs("%s/weights_tmp"%log_dir)

print('{:=^80}'.format(' Settings '))
for key, value in tf.app.flags.FLAGS.flag_values_dict().items():
  print('%s%s'%('{:<30}'.format('%s:'%key),value))
print('{:=^80}'.format(''))

# Possible parameter options
Rdata = ['mnist','cifar10',
         'svhn','stl10','fashion']  # Datasets
Rlr = [0.0002,0.005]                # Learning rate range
Rbs = [32,64,128,256]               # Batch sizes
Rbn = [False,True]                  # Batch norm.
Raug = [False,True]                 # Augmentation
Ropt = [0,1,2]                      # Optimizers [Adam, RMSprop, Momentum]
Ract = [0,1,2,3]                    # Activation func. [ReLU, ELU, Sigmoid, TanH]
Ritype = [0,1,2,3]                  # Initialization [Const, Rand norm, Glorot uniform, Glorot norm]

Rfsize = [3,5,7]                    # Conv. filter sizes
Rldepthc = [3,4,5]                  # Conv. layer depth (number of layers)
Rldepthf = [2,3,4]                  # FC depth
Rlwidthc = [4,8,12]                 # Conv. width (channel multiplier)
Rlwidthf = [64,128,192]             # FC width (multiplier)

# Random params from options
if FLAGS.params == None:
  if FLAGS.data_nmn == None:
    data_nmn = Rdata[np.round(np.random.uniform(-0.5,len(Rdata)-0.5)).astype('int64')]
  else:
    data_nmn = FLAGS.data_nmn
  learning_rate = np.random.uniform(Rlr[0],Rlr[1])
  batch_size = Rbs[np.round(np.random.uniform(-0.5,len(Rbs)-0.5)).astype('int64')]
  bn = True #Rbn[np.round(np.random.uniform(-0.5,len(Rbn)-0.5)).astype('int64')]
  augment = Raug[np.round(np.random.uniform(-0.5,len(Raug)-0.5)).astype('int64')]
  opt = Ropt[np.round(np.random.uniform(-0.5,len(Ropt)-0.5)).astype('int64')]
  act = Ract[np.round(np.random.uniform(-0.5,len(Ract)-0.5)).astype('int64')]
  itype = Ritype[np.round(np.random.uniform(-0.5,len(Ritype)-0.5)).astype('int64')]

  fsize = Rfsize[np.round(np.random.uniform(-0.5,len(Rfsize)-0.5)).astype('int64')]
  ldepth = [ Rldepthc[np.round(np.random.uniform(-0.5,len(Rldepthc)-0.5)).astype('int64')], Rldepthf[np.round(np.random.uniform(-0.5,len(Rldepthf)-0.5)).astype('int64')] ]
  lwidth = [ Rlwidthc[np.round(np.random.uniform(-0.5,len(Rlwidthc)-0.5)).astype('int64')], Rlwidthf[np.round(np.random.uniform(-0.5,len(Rlwidthf)-0.5)).astype('int64')] ]

# Specified hyper-parameters
else:
  print("Parsing hyper-parameters from input")
  print(FLAGS.params)

  data_nmn = FLAGS.params[0]
  learning_rate = float(FLAGS.params[1])
  batch_size = int(FLAGS.params[2])
  bn = int(FLAGS.params[3])
  augment = int(FLAGS.params[4])
  opt = int(FLAGS.params[5])
  act = int(FLAGS.params[6])
  itype = int(FLAGS.params[7])

  fsize = int(FLAGS.params[8])
  ldepth = [int(FLAGS.params[9]), int(FLAGS.params[10])]
  lwidth = [int(FLAGS.params[11]), int(FLAGS.params[12])]

# Fixed network trainings, i.e. with same number of layers
if FLAGS.fixed:
  fsize = 5; ldepth = [3,2]; lwidth = [8,128];

transf_geom = augment
transf_int = augment

# Import data
(trainset_x, trainset_y, N, sx, sy, K) = util.load_data(data_nmn, FLAGS.data_dir, 'train')
(testset_x, testset_y, N_test, _, _, _) = util.load_data(data_nmn, FLAGS.data_dir, 'test')

# Train/valid split
perm = [i for i in range(N)]
random.shuffle(perm)
trainset_x = trainset_x[perm,:,:,:]
trainset_y = trainset_y[perm]
split_pos = int(0.9*N)
trainset_x, validset_x = np.split(trainset_x, [split_pos])
trainset_y, validset_y = np.split(trainset_y, [split_pos])

N = trainset_x.shape[0]
N_valid = validset_x.shape[0]

print("====================================")
print("Train set (%d images):"%N)
print(trainset_x.shape)
print("Valid set (%d images):"%N_valid)
print(validset_x.shape)
print("====================================\n\n")

# Sub-sample dataset, if needed for class balance
trainset_X = trainset_x
trainset_Y = trainset_y
(trainset_x, trainset_y) = util.shuffle_data(trainset_X, trainset_Y, K)
N = trainset_x.shape[0]

# Export meta-data
with open('%s/meta.csv'%log_dir, mode='w') as meta:
  meta_writer = csv.writer(meta, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  meta_writer.writerow(['dataset', 'lrate', 'batch_size', 'batch_norm', 
                        'augmentation', 'optimizer', 'activation', 
                        'initialization', 'filter_size', 'depth_conv', 
                        'depth_fc', 'width_conv', 'width_fc'])
  meta_writer.writerow([data_nmn, learning_rate, batch_size, bn, 
                        augment, opt, act, itype, fsize, ldepth[0],
                        ldepth[1], lwidth[0], lwidth[1]])

print('====================================')
print('SETTINGS:\n====================================')
print('\tdataset:          %s'%data_nmn)
print('\tlrate:            %f'%learning_rate)
print('\tbatch_size:       %d'%batch_size)
print('\tbatch_normal:     %d'%bn)
print('\taugmentation:     %d'%augment)
print('\toptimizer:        %d'%opt)
print('\tactivation:       %d'%act)
print('\tinitialization:   %d'%itype)
print('\tfilter_size:      %d'%fsize)
print('\tdepth_conv:       %d'%ldepth[0])
print('\tdepth_fc:         %d'%ldepth[1])
print('\twidth_conv:       %d'%lwidth[0])
print('\twidth_fc:         %d'%lwidth[1])
print('====================================\n\n')

# Input placeholders
# Resize images to 32x32 (used for MNIST)
if sx != 32 or sy != 32:
  sx = 32
  sy = 32
  x = tf.placeholder(tf.float32, [batch_size, 28, 28, 3])
  x_in = tf.image.resize_images(x, [sx,sy], method=tf.image.ResizeMethod.BILINEAR)
else:
  x = tf.placeholder(tf.float32, [batch_size, sx, sy, 3])
  x_in = x

# Geometric augmentations
if transf_geom:
  sc = FLAGS.transf_scaling

  # Random transformation of translation, rotation, zoom, and shearing
  tx = tf.random_uniform(shape=[batch_size,1], minval=-sc, maxval=sc, dtype=tf.float32)
  ty = tf.random_uniform(shape=[batch_size,1], minval=-sc, maxval=sc, dtype=tf.float32)
  r  = tf.random_uniform(shape=[batch_size,1], minval=np.deg2rad(-5.0*sc), maxval=np.deg2rad(5.0*sc), dtype=tf.float32)
  z  = tf.random_uniform(shape=[batch_size,1], minval=1.0-0.033*sc, maxval=1.0+0.033*sc, dtype=tf.float32)
  hx = tf.random_uniform(shape=[batch_size,1], minval=np.deg2rad(-sc), maxval=np.deg2rad(sc), dtype=tf.float32)
  hy = tf.random_uniform(shape=[batch_size,1], minval=np.deg2rad(-sc), maxval=np.deg2rad(sc), dtype=tf.float32)
  a = hx - r
  b = tf.cos(hx)
  c = hy + r
  d = tf.cos(hy)
  m1 = tf.divide(z*tf.cos(a), b)
  m2 = tf.divide(z*tf.sin(a), b)
  m3 = tf.divide(sx*b-sx*z*tf.cos(a)+2*tx*z*tf.cos(a)-sy*z*tf.sin(a)+2*ty*z*tf.sin(a), 2*b)
  m4 = tf.divide(z*tf.sin(c), d)
  m5 = tf.divide(z*tf.cos(c), d)
  m6 = tf.divide(sy*d-sy*z*tf.cos(c)+2*ty*z*tf.cos(c)-sx*z*tf.sin(c)+2*tx*z*tf.sin(c), 2*d)
  m7 = tf.zeros([batch_size,2], 'float32')
  transf = tf.concat([m1, m2, m3, m4, m5, m6, m7], 1)
  x_in = tf.contrib.image.transform(x_in, transf, interpolation='BILINEAR')

# Intensity/color augmentations
if transf_int:
  x_in = tf.image.random_brightness(x_in, max_delta=0.2)
  x_in = tf.image.random_contrast(x_in, 0.5, 1.5)
  x_in = tf.image.random_hue(x_in, max_delta=0.08)
  x_in = tf.image.random_saturation(x_in, 0.3, 1.5)

  std = tf.random_uniform(shape=[1], minval=0.0, maxval=0.03, dtype=tf.float32)
  x_in = tf.add(x_in, tf.random_normal(shape=tf.shape(x_in), mean=0.0, stddev=std, dtype=tf.float32))

x_in = tf.minimum(x_in, 1.0)
x_in = tf.maximum(x_in, 0.0)
y_ = tf.placeholder(tf.int64, [None])

# Network specification based on hyper-parameters
with tf.variable_scope("model"):
  y, keep_prob = model.deepnn(x_in, fsize, ldepth, lwidth, bn, act, itype, True, True)

for v in tf.global_variables():
  print("Name: ", v.name)

print('Model size = %d weights\n'%model.count_all_vars())
sys.stdout.flush()

# Loss
with tf.name_scope('loss'):
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
cross_entropy = tf.reduce_mean(cross_entropy)

# Optimizer specification
with tf.variable_scope("optimizer"):
  extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(extra_update_ops):
    global_step = tf.Variable(0, trainable=False)
    learning_rate_decay = tf.train.exponential_decay(learning_rate, global_step, int(N/batch_size), 0.96, staircase=True)
    
    if opt == 0:
      train_step = tf.train.AdamOptimizer(learning_rate_decay).minimize(cross_entropy)
    elif opt == 1:
      train_step = tf.train.RMSPropOptimizer(learning_rate_decay).minimize(cross_entropy)
    elif opt == 2:
      train_step = tf.train.MomentumOptimizer(learning_rate_decay, 0.95).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  start_time = time.time()
  A = np.zeros([3000,5])
  es, esT, esS, esP = 0, 0, 0, 0
  train_accuracy, a, b, ii, shf = 0.0, 0, 0, 0, 0
  for i in range(int(FLAGS.epochs*N/batch_size) + 1):
    shf += 1

    # Shuffle data at each epoch
    if FLAGS.shuffle_epochs and shf*batch_size > N:
      shf = 0
      (trainset_x, trainset_y) = util.shuffle_data(trainset_X, trainset_Y, K)

    # Batch indices
    bb = i*batch_size
    ind = np.mod(np.arange(bb, bb+batch_size), N)

    x_batch = trainset_x[ind,:,:,:]
    y_batch = trainset_y[ind]

    b += 1
    train_accuracy += accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})

    # Evaluate and export at each epoch
    if i % int(N/batch_size) == 0:
      ii += 1

      # Validation accuracy
      valid_accuracy, cv = 0.0, 0
      for j in range(int(N_valid/batch_size)):
        cv+=1
        bb = j*batch_size
        ind = np.mod(np.arange(bb, bb+batch_size), N_test)
        x_valid = validset_x[ind,:,:,:]
        y_valid = validset_y[ind]
        valid_accuracy += accuracy.eval(feed_dict={x: x_valid, y_: y_valid, keep_prob: 1.0})

      # Test accuracy
      test_accuracy, ct = 0.0, 0
      for j in range(int(N_test/batch_size)):
        ct+=1
        bb = j*batch_size
        ind = np.mod(np.arange(bb, bb+batch_size), N_test)
        x_test = testset_x[ind,:,:,:]
        y_test = testset_y[ind]
        test_accuracy += accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})

      print('%03d: step %05d (epoch %0.4f of %d), train = %0.6f, valid = %0.6f, test = %0.6f' % 
            (ii, i, i*batch_size/N, FLAGS.epochs, train_accuracy/b, valid_accuracy/cv, test_accuracy/ct))
      sys.stdout.flush()

      A[a,0] = i
      A[a,1] = 100.0*(1.0-train_accuracy/b)
      A[a,2] = 100.0*(1.0-valid_accuracy/cv)
      A[a,3] = 100.0*(1.0-test_accuracy/ct)
      a += 1
      b = 0
      train_accuracy = 0.0

      # Export weights
      model.print_all_vars('%s/weights_tmp/%03d.bin'%(log_dir, ii))

      # Early stopping
      if ii>1 and valid_accuracy/cv < curr_acc:
        es += 1
        esT += 1
        esS = 0
      elif ii>1 and np.abs(valid_accuracy/cv-curr_acc)<1e-8:
        esS += 1
      else:
        es = 0
        esS = 0

      curr_acc = valid_accuracy/cv
      if ii==1:
        curr_acc_smooth = curr_acc
      else:
        curr_acc_smooth = 0.5*curr_acc_smooth + 0.5*valid_accuracy/cv
      A[a-1,4] = 100.0*(1.0-curr_acc_smooth)

      if ii>1 and curr_acc < curr_acc_smooth:
        esP += 1

    # Do early stopping if criterias are met
    if (es >= FLAGS.early_stopping or esP >= 30 or esS >= 30) and ii >= 20:
      print('Early stopping!')
      print('\t%d consecutive non-decreasing iterations'%es)
      print('\t%d total non-decreasing iterations'%esT)
      print('\t%d consecutive stationary iterations'%esS)
      print('\t%d total non-decreasing iterations (smoothed)'%esP)
      break

    # Training step
    train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: FLAGS.dropout})

  # Timing
  duration = time.time() - start_time
  print('Finished! Total time = %f min'%(duration/60.0))
  sys.stdout.flush()

  # Sample fixed number of exported weights, delete remaining
  print("Sampling exported weights...")
  s = np.round(np.linspace(1,ii,20)).astype('uint32')
  for k in range(FLAGS.prints):
    print('\t%03d: iteration %03d'%(k, s[k]))
    os.rename('%s/weights_tmp/%03d.bin'%(log_dir, s[k]), '%s/weights/%03d.bin'%(log_dir, k+1))
  shutil.rmtree('%s/weights_tmp'%log_dir)

  # Plot train/valid/test accuracy
  A = A[:a,:]
  fig = plt.figure(figsize=(10, 6))
  plt.plot(A[:,0], A[:,1], 'r--', A[:,0], A[:,2], 'r-', A[:,0], A[:,3], 'g-', A[:,0], A[:,4], 'b-')
  plt.grid(True)
  plt.xlabel('Step')
  plt.ylabel('Error')
  plt.legend(['Training error', 'Validation error', 'Test error', 'Val error (smoothed)'])
  plt.savefig('%s/error.pdf'%log_dir, dpi=100)
  plt.close(fig)

  B = np.array([ii,i,i*batch_size/N,duration])

  A.tofile('%s/error.bin'%log_dir)
  B.tofile('%s/stat.bin'%log_dir)

