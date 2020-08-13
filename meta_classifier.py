import sys, os, random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import model, util

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_train", "data_train.csv", "Path to training data csv")
tf.flags.DEFINE_string("data_test", "data_test.csv", "Path to test data csv")

tf.flags.DEFINE_string("log_dir", "/mnt/raid/dnn_weights/output", "Path to output directory")
tf.flags.DEFINE_string("model_pth", None, "Path to pre-trained model")

tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.flags.DEFINE_integer("epochs", 50, "Epochs")
tf.flags.DEFINE_bool("shuffle_epochs", True, "Re-shuffle at each epoch")
tf.flags.DEFINE_float("dropout", 0.5, "Drop-out before final output")
tf.flags.DEFINE_bool("batchn", True, "Batch normalization")
tf.flags.DEFINE_integer("decay_frequency", 50, "How many times to decay learning rate during training")

tf.flags.DEFINE_integer("prop", 0, "Property to train for")
tf.flags.DEFINE_integer("dnn_part", 0, "Which part of weight vector: 0 - all, 1 - conv, 2 - fc")
tf.flags.DEFINE_integer("slice_length", 5000, "Slice length (size of weight subset)")
tf.flags.DEFINE_list("K", "20,20", "Range of weight snapshots")
tf.flags.DEFINE_integer("P", 100, "Positions")
tf.flags.DEFINE_bool("training", True, "Training")
tf.flags.DEFINE_bool("valid_pos", False, "Test different slice positions")
tf.flags.DEFINE_bool("valid_avg", False, "Print average accuracy over each weight")
tf.flags.DEFINE_bool("load_uniform", True, "Class balancing")


if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

print('{:=^80}'.format(' Settings '))
for key, value in tf.app.flags.FLAGS.flag_values_dict().items():
  print('%s%s'%('{:<30}'.format('%s:'%key),value))
print('{:=^80}'.format(''))


sz = FLAGS.slice_length
K = list(range(int(FLAGS.K[0]),int(FLAGS.K[1])+1,1))
P = FLAGS.P
nci = 13

print('Test data:')
(testset_x, testset_y, err, N_test, C, meta, properties) = util.import_weights(FLAGS.data_test, FLAGS.prop, K, FLAGS.load_uniform)

if FLAGS.training:
  print('Training data:')
  (trainset_x, trainset_y, err_train, N, Cp, meta_train, _) = util.import_weights(FLAGS.data_train, FLAGS.prop, K)

  assert(C==Cp)

  perm = [i for i in range(N)]
  random.shuffle(perm)
  trainset_x = trainset_x[perm]
  trainset_y = trainset_y[perm]
  meta_train = meta_train[perm]
  err_train = err_train[perm,:]

  # Train/valid split
  split_pos = int(0.9*N)
  trainset_x, validset_x = np.split(trainset_x, [split_pos])
  trainset_y, validset_y = np.split(trainset_y, [split_pos])
  meta_train, meta_valid = np.split(meta_train, [split_pos])
  err_train, err_valid = np.split(err_train, [split_pos])

  N = trainset_x.shape[0]
  N_valid = validset_x.shape[0]

  print("====================================")
  print("Train set (%d weight snapshots):"%N)
  print(trainset_x.shape)
  print("Valid set (%d weight snapshots):"%N_valid)
  print(validset_x.shape)
  print("====================================\n\n")

  perm = [i for i in range(N_test)]
  random.shuffle(perm)
  testset_x = testset_x[perm]
  testset_y = testset_y[perm]
  meta = meta[perm]
  err = err[perm,:]

print('Batch size = %d, Epochs = %d, Learning rate = %0.6f'%(FLAGS.batch_size,FLAGS.epochs,FLAGS.learning_rate))
if FLAGS.training:
  print('Training samples = %d (%d dim), Validation samples = %d, Classes = %d\n'%(N, sz, N_test, C))
sys.stdout.flush()

x = tf.placeholder(tf.float32, [FLAGS.batch_size, sz])
x_in = tf.reshape(x, [FLAGS.batch_size, sz, 1])

# More layers if there are large weight vectors
if sz > 40000:
  print("Using 11 convolutional layer model\n")
  y, _, keep_prob = model.cnn1D(x_in, C, FLAGS.batchn, True, FLAGS.training)
else:
  print("Using 9 convolutional layer model\n")
  y, _, keep_prob = model.cnn1D_small(x_in, C, FLAGS.batchn, True, True)

for v in tf.global_variables():
  print("Name: ", v.name)

print('Model size = %d weights\n'%model.count_all_vars())
sys.stdout.flush()

class_probabilities = tf.nn.softmax(y)

# Regression
if type(testset_y[0]) == np.float32:
  print('Using floating point difference loss\n')
  y_ = tf.placeholder(tf.float32, [None])
  loss_func = tf.reduce_mean(tf.abs(y-y_))
  accuracy = loss_func
  _, batchvar = tf.nn.moments(tf.abs(y-y_), 0)

# CLassification
else:
  print('Using cross entropy loss\n')
  y_ = tf.placeholder(tf.int64, [None])
  loss_func = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y))
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  _, batchvar = tf.nn.moments(correct_prediction, 0)

if FLAGS.training:
  extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(extra_update_ops):
    global_step = tf.Variable(0, trainable=False)
    learning_rate_decay = tf.train.exponential_decay(FLAGS.learning_rate, global_step, int(FLAGS.epochs*N/(FLAGS.batch_size*FLAGS.decay_frequency)), 0.95, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate_decay).minimize(loss_func)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

with tf.Session() as sess:
  if FLAGS.model_pth == None:
    sess.run(tf.global_variables_initializer())
  elif not FLAGS.valid_pos:
    print('Loading model from "%s": '%FLAGS.model_pth)
    saver.restore(sess, FLAGS.model_pth)
    sys.stdout.flush()

  if FLAGS.training:
    start_time = time.time()
    A = np.zeros([3000,4])
    train_accuracy, b, shf, p, a = 0.0, 0, 0, 0, 0
    for i in range(int(FLAGS.epochs*N/FLAGS.batch_size) + 1):
      shf += 1

      # Shuffle data
      if FLAGS.shuffle_epochs and shf*FLAGS.batch_size > N:
        shf = 0
        perm = [i for i in range(N)]
        random.shuffle(perm)
        trainset_x = trainset_x[perm]
        trainset_y = trainset_y[perm]
        meta_train = meta_train[perm]

      bb = i*FLAGS.batch_size
      ii = np.mod(np.arange(bb, bb+FLAGS.batch_size), N)

      x_batch = np.empty([FLAGS.batch_size, sz], dtype='float32')
      y_batch = trainset_y[ii]

      for k in range(FLAGS.batch_size):
        if FLAGS.dnn_part == 0:
          xs = random.randint(0, len(trainset_x[ii[k]])-sz)
        elif FLAGS.dnn_part == 1:
          xs = random.randint(0, np.maximum(0, meta_train[ii[k],nci]-sz))
        elif FLAGS.dnn_part == 2:
          xs = random.randint(meta_train[ii[k],nci], len(trainset_x[ii[k]])-sz)

        x_batch[k,:] = trainset_x[ii[k]][xs:xs+sz]

      b += 1
      train_accuracy += accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})

      if i % int(FLAGS.epochs*N/(FLAGS.batch_size*50)) == 0:
        valid_accuracy, cv = 0.0, 0
        for j in range(int(N_valid/FLAGS.batch_size)):
          cv+=1
          bb = j*FLAGS.batch_size
          ii = np.mod(np.arange(bb, bb+FLAGS.batch_size), N_valid)

          x_valid = np.empty([FLAGS.batch_size, sz], dtype='float32')
          y_valid = validset_y[ii]

          for k in range(FLAGS.batch_size):
            if FLAGS.dnn_part == 0:
              xs = random.randint(0, len(validset_x[ii[k]])-sz)
            elif FLAGS.dnn_part == 1:
              xs = random.randint(0, np.maximum(0, meta_valid[ii[k],nci]-sz))
            elif FLAGS.dnn_part == 2:
              xs = random.randint(meta_valid[ii[k],nci], len(validset_x[ii[k]])-sz)
            x_valid[k,:] = validset_x[ii[k]][xs:xs+sz]

          valid_accuracy += accuracy.eval(feed_dict={x: x_valid, y_: y_valid, keep_prob: 1.0})

        test_accuracy, ct = 0.0, 0
        for j in range(int(N_test/FLAGS.batch_size)):
          ct+=1
          bb = j*FLAGS.batch_size
          ii = np.mod(np.arange(bb, bb+FLAGS.batch_size), N_test)

          x_test = np.empty([FLAGS.batch_size, sz], dtype='float32')
          y_test = testset_y[ii]

          for k in range(FLAGS.batch_size):
            if FLAGS.dnn_part == 0:
              xs = random.randint(0, len(testset_x[ii[k]])-sz)
            elif FLAGS.dnn_part == 1:
              xs = random.randint(0, np.maximum(0, meta[ii[k],nci]-sz))
            elif FLAGS.dnn_part == 2:
              xs = random.randint(meta[ii[k],nci], len(testset_x[ii[k]])-sz)
            x_test[k,:] = testset_x[ii[k]][xs:xs+sz]

          test_accuracy += accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})


        print('step %05d (epoch %0.4f of %d), train accuracy = %0.6f, valid accuracy = %0.6f, test accuracy = %0.6f' % (i, i*FLAGS.batch_size/N, FLAGS.epochs, train_accuracy/b, valid_accuracy/cv, test_accuracy/ct))
        sys.stdout.flush()

        A[a,0] = i
        A[a,1] = 100.0*(train_accuracy/b)
        A[a,2] = 100.0*(valid_accuracy/cv)
        A[a,3] = 100.0*(test_accuracy/ct)
        a += 1

        b = 0
        train_accuracy = 0.0

        p += 1
        save_path = saver.save(sess, '%s/model.ckpt'%FLAGS.log_dir, global_step=p)

      train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: FLAGS.dropout})

    duration = time.time() - start_time
    print('Finished! Total time = %f min'%(duration/60.0))
    sys.stdout.flush()

    A = A[:a,:]
    fig = plt.figure(figsize=(10, 6))
    plt.plot(A[:,0], A[:,1], 'r--', A[:,0], A[:,2], 'r-', A[:,0], A[:,3], 'g-')
    plt.grid(True)
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.legend(['Training accuracy', 'Validation accuracy', 'Test accuracy'])
    plt.savefig('%s/accuracy.pdf'%FLAGS.log_dir, dpi=100)
    plt.close(fig)

    A.tofile('%s/accuracy.bin'%FLAGS.log_dir)

  else:
    if FLAGS.valid_avg:
      print("Running per-weight average evaluation...\n")

      for k in range(len(K)):
        print("\n\nStep %02d:\n"%K[k])
        fid_acc = open('valid_avg_%02d_%06d_%02d_%02d.txt'%(FLAGS.prop, sz, K[k], FLAGS.dnn_part), 'w')

        for i in range(N_test):
          x_valid = np.empty([FLAGS.batch_size, sz])
          y_valid = np.ones(FLAGS.batch_size)*testset_y[i]

          samp = (np.round(np.linspace(0, len(testset_x[i])-sz, FLAGS.batch_size))).astype('int64')
          if FLAGS.dnn_part == 1:
            samp = (np.round(np.linspace(0, np.maximum(0, meta[i,nci]-sz), FLAGS.batch_size))).astype('int64')
          elif FLAGS.dnn_part == 2:
            samp = (np.round(np.linspace(meta[i,nci], len(testset_x[i])-sz, FLAGS.batch_size))).astype('int64')

          for j in range(FLAGS.batch_size):
            x_valid[j] = testset_x[i][samp[j]:samp[j]+sz]

          valid_accuracy = accuracy.eval(feed_dict={x: x_valid, y_: y_valid, keep_prob: 1.0})

          fid_acc.write('%s %08f %08f %d %08f '%(meta[i,0], err[i,0], err[i,1], FLAGS.prop, valid_accuracy))
          fid_acc.write('%08f %d %d %d %d %d %d %d %d %d %d\n'%(meta[i,1],meta[i,2],meta[i,3],meta[i,4],meta[i,5],meta[i,6],meta[i,7],meta[i,8],meta[i,9],meta[i,10],meta[i,11]))
          print('sample %03d of %d: accuracy = %f -- %f, %03d, %d, %d, %d, %s'%(i,N_test,valid_accuracy,  meta[i,1],meta[i,2],meta[i,3],meta[i,4],meta[i,5],meta[i,0]))
          
          sys.stdout.flush()

        fid_acc.close()
    
    elif FLAGS.valid_pos:

      print("Running positional evaluation...\n")

      M = range(50,51)
      N_ = int(N_test/FLAGS.batch_size)*FLAGS.batch_size
      G = np.zeros((N_,len(M),P,C), dtype='float32')

      for m in range(len(M)):
        model_pth = '%s/model.ckpt-%d'%(FLAGS.model_pth,M[m])
        print('Loading model from "%s": '%model_pth)
        saver.restore(sess, model_pth)
        sys.stdout.flush()

        for p in range(P):
          valid_accuracy, c = 0.0, 0
          for j in range(int(N_/FLAGS.batch_size)):
            c+=1
            bb = j*FLAGS.batch_size
            ii = np.mod(np.arange(bb, bb+FLAGS.batch_size), N_)

            x_valid = np.empty([FLAGS.batch_size, sz], dtype='float32')
            y_valid = testset_y[ii]

            for k in range(FLAGS.batch_size):
              samp = (np.round(np.linspace(0, len(testset_x[ii[k]])-sz, P))).astype('int64')
              if FLAGS.dnn_part == 1:
                samp = (np.round(np.linspace(0, np.maximum(0, meta[ii[k],nci]-sz), P))).astype('int64')
              elif FLAGS.dnn_part == 2:
                samp = (np.round(np.linspace(meta[ii[k],nci], len(testset_x[ii[k]])-sz, P))).astype('int64')
              xs = samp[p]
              x_valid[k,:] = testset_x[ii[k]][xs:xs+sz]

            cb = class_probabilities.eval(feed_dict={x: x_valid, keep_prob: 1.0})
            G[ii,m,p,:] = cb

          print('model %d, position %04d of %04d' % (M[m], p+1, P))
          sys.stdout.flush()

      G.tofile('%s/step_%02d.bin'%(FLAGS.log_dir, K[0]))

    else:
      test_accuracy, ct = 0.0, 0
      for j in range(int(N_test/FLAGS.batch_size)):
        ct+=1
        bb = j*FLAGS.batch_size
        ii = np.mod(np.arange(bb, bb+FLAGS.batch_size), N_test)

        x_test = np.empty([FLAGS.batch_size, sz], dtype='float32')
        y_test = testset_y[ii]

        for k in range(FLAGS.batch_size):
          if FLAGS.dnn_part == 0:
            xs = random.randint(0, len(testset_x[ii[k]])-sz)
          elif FLAGS.dnn_part == 1:
            xs = random.randint(0, np.maximum(0, meta[ii[k],nci]-sz))
          elif FLAGS.dnn_part == 2:
            xs = random.randint(meta[ii[k],nci], len(testset_x[ii[k]])-sz)
          x_test[k,:] = testset_x[ii[k]][xs:xs+sz]

        test_accuracy += accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})

      print('done (%d batches, %d samples), test accuracy = %0.6f' % (ct,ct*FLAGS.batch_size,test_accuracy/ct))
      sys.stdout.flush()


