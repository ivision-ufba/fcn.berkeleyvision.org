import caffe
import surgery, score, plotter

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'

# init
# caffe.set_device(int(sys.argv[1]))
# caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/segperson/indices/val.txt', dtype=str)

accs = []
losss = []

for _ in range(75):
    solver.step(4000)
    acc, loss = score.seg_tests(solver, False, val, layer='score')
    accs = np.append(accs, acc)
    losss = np.append(losss, loss)
    plotter.plot(accs, losss)
