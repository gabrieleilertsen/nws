import sys, random
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.misc as sm
import pandas

# Load image dataset
def load_data(data_nmn, data_dir, dset):
  if data_nmn == 'mnist' or data_nmn == 'fashion':
    data_pth = '%s/%s'%(data_dir, data_nmn)

    print('Loading %s data from "%s"...'%(dset, data_pth))
    mnist = input_data.read_data_sets(data_pth)
    K = 10
    sx = 28
    sy = 28

    if dset == 'train':
      N = len(mnist.train.images)
      set_x = np.tile(np.reshape(mnist.train.images, [N, sx, sy, 1]), (1,1,1,3))
      set_y = mnist.train.labels
    else:
      N = len(mnist.test.images)
      set_x = np.tile(np.reshape(mnist.test.images, [N, sx, sy, 1]), (1,1,1,3))
      set_y = mnist.test.labels

  elif data_nmn == 'cifar10' or data_nmn == 'svhn' or data_nmn == 'stl10' or data_nmn == 'pascal':
    data_pth = '%s/%s'%(data_dir, data_nmn)

    # Load data
    print('Loading %s data from "%s"...'%(dset, data_pth))
    meta = np.fromfile('%s/%s_meta.bin'%(data_pth,dset), 'uint32')
    (K,N,sx,sy,sz) = meta[:5]
    set_y = meta[5:]
    set_x = np.fromfile('%s/%s.bin'%(data_pth,dset), 'uint8')
    set_x = np.reshape(set_x, [sz,sx,sy,N])
    set_x = np.transpose(set_x, [3,2,1,0])
    set_x = set_x.astype(np.float32)/255.0
  else:
    print('\nUnknown dataset "%s"!\n\n'%data_nmn)

  print("done! %d %s samples loaded (%d classes)\n"%(N,dset,K))
  sys.stdout.flush()

  return (set_x, set_y, N, sx, sy, K)


# Shuffle and class balancing
def shuffle_data(set_x, set_y, K):
  # Find class with least samples
  mn = int(1e6)
  for k in range(K):
    mn = np.minimum(mn, np.sum(set_y==k))

  (N,sx,sy,sz) = set_x.shape

  # Sub-sample dataset, to get class balance
  if K*mn < N:
    print('Sub-sampling dataset, %d->%d (%d samples from each class)...'%(N,K*mn,mn))
    set_xo = np.zeros((K*mn,sx,sy,sz))
    set_yo = np.zeros((K*mn))
    for k in range(K):
      ind = (set_y == k)
      x = set_x[ind,:,:,:]
      y = set_y[ind]

      perm = [i for i in range(np.sum(ind))]
      random.shuffle(perm)
      set_xo[k*mn:(k+1)*mn,:,:,:] = x[perm[:mn],:,:,:]
      set_yo[k*mn:(k+1)*mn] = y[perm[:mn]]
    
  perm = [i for i in range(set_xo.shape[0])]
  random.shuffle(perm)
  set_xo = set_xo[perm,:,:,:]
  set_yo = set_yo[perm]

  return (set_xo, set_yo)


# Import NWS dataset, i.e. dataset of DNN weights with meta data
def import_weights(pth,prop,steps,un=True):
  # Possible parameter options
  opt = [['mnist', 'cifar10', 'svhn', 'stl10', 'fashion'],
         [0.0002,0.005],
         [32,64,128,256],
         [False,True],
         [False,True],
         [0,1,2],
         [0,1,2,3],
         [0,1,2,3],
         [3,5,7],
         [3,4,5],
         [2,3,4],
         [4,8,12],
         [64,128,192]]
  
  data = pandas.read_csv(pth)
  properties = list(data)

  dt = np.array(data[properties[prop]])
  C = len(opt[prop])
  
  print('Loading data for property "%s"'%properties[prop])

  # Floating point annotations
  if type(opt[prop][0])==float:
    C = 1
    y = np.array(data[properties[prop]], dtype='float32')
    N = len(y)*len(steps)
    X = np.empty(N, dtype=object)
    Y = np.empty(N, dtype='float32')
    err = np.empty([N,2], dtype='float32')
    meta = np.empty([N,len(properties)], dtype=object)
    k = 0

    print('Loading weights...')
    for i in range(len(y)):
      A = np.fromfile('%s/error.bin'%(data['path'][i]))
      B = np.fromfile('%s/stat.bin'%(data['path'][i]))
      A = np.reshape(A, [int(len(A)/5),5])
      A = A[np.round(np.linspace(0,B[0]-1,20)).astype('uint32'),:]
      
      for m in range(len(steps)):
        model = '%s/weights/%03d.bin'%(data['path'][i], steps[m])
        x = np.fromfile(model, 'double')
        assert(len(x)==data['weightn_conv'][i]+data['weightn_fc'][i])
        X[k] = x
        Y[k] = y[i]
        err[k, 0] = A[steps[m]-1,1] # train accuracy
        err[k, 1] = A[steps[m]-1,3] # test accuracy

        for p in range(data.shape[1]):
          meta[k,p] = data[properties[p]][i]

        k += 1

  else:
    K = np.empty(C, dtype='int64')
    k = len(dt)
    for c in range(C):
      K[c] = np.sum(dt==opt[prop][c])
      k = np.minimum(k, K[c])

      if type(opt[prop][c])==str:
        print('\tClass "%s", %d samples'%(opt[prop][c], K[c]))
      elif type(opt[prop][c])==int or type(opt[prop][c])==bool:
        print('\tClass "%d", %d samples'%(opt[prop][c], K[c]))

    if un:
      K = k*np.ones(C, dtype='int64')

    print('Using in total %d of %d training runs'%(np.sum(K),len(dt)))

    N = np.sum(K)*len(steps)
    X = np.empty(N, dtype=object)
    Y = np.empty(N, dtype='int64')
    err = np.empty([N,2], dtype='float32')
    meta = np.empty([N,data.shape[1]], dtype=object)
    Kc = np.zeros(C, dtype='int64')
    k = 0

    print('Loading weights...')

    for n in range(len(dt)):
      A = np.fromfile('%s/error.bin'%(data['path'][n]))
      B = np.fromfile('%s/stat.bin'%(data['path'][n]))
      A = np.reshape(A, [int(len(A)/5),5])
      A = A[np.round(np.linspace(0,B[0]-1,20)).astype('uint32'),:]
      
      for c in range(C):
        if dt[n]==opt[prop][c]:
          break

      Kc[c] += 1
      
      for m in range(len(steps)):

        if Kc[c] <= K[c]:
          model = '%s/weights/%03d.bin'%(data['path'][n], steps[m])
          x = np.fromfile(model, 'double')
          assert(len(x)==data['weightn_conv'][n]+data['weightn_fc'][n])
          X[k] = x
          Y[k] = c
          err[k, 0] = A[steps[m]-1,1] # train accuracy
          err[k, 1] = A[steps[m]-1,3] # test accuracy

          for p in range(data.shape[1]):
            meta[k,p] = data[properties[p]][n]

          k += 1

  print('...done! Loaded %d weights.\n'%k)
  sys.stdout.flush()

  return (X,Y,err,N,C,meta,properties)

