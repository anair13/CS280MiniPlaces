import os
# os.environ['GLOG_minloglevel'] = '0' # enable logging
import numpy as np
import caffe
import caffe.io
import sys
import my_pycaffe as mp

# Make sure that caffe is on the python path:
# caffe_root = '../../../'  # this file is expected to be in {caffe_root}/examples
# import sys
# sys.path.insert(0, caffe_root + 'python')

# from __future__ import print_function

def train(gpu, solver_filename):
    ### settings
    start_at = 0
    record_iter = 20
    init_from_ref = False
    ###
    niter = 50000 + 5
    test_interval = 100
    test_iter = 5
    
    # losses will also be stored in the log
    if start_at:
        train_loss = list(np.load(snapshot_folder + "/training_loss.npy"))
        train_acc = list(np.load(snapshot_folder + "/training_acc.npy"))
        test_loss = list(np.load(snapshot_folder + "/test_loss.npy"))
        test_loss = list(np.load(snapshot_folder + "/test_acc.npy"))
    else:
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

    output = np.zeros((niter, 8, 10))
    my_solver = mp.MySolver.from_file(solver_filename, record_iter, gpu)
    debug_file = snapshot_folder + "/statistics.txt"
    solver = my_solver.solver_
    ref = '../bvlc_googlenet.caffemodel'
    # ref = 'snapshots/iter1000.caffemodel'
    if init_from_ref:
        solver.net.copy_from(ref)

    # the main solver loop
    for it in range(start_at, niter):
        my_solver.solve(1)  # step once
        if it % record_iter == 0: # record once in a while
            my_solver.dump_to_file(debug_file)
        
        # store the train loss 
        loss = float(solver.net.blobs['loss'].data)
        train_loss.append(loss)
        acc = float(solver.net.blobs['accuracy'].data)
        train_acc.append(acc)
        
        if it % test_interval == 0:
            tmp_losses = []
            tmp_accs = []
            for j in range(test_iter):
                solver.test_nets[0].forward()
                loss = float(solver.test_nets[0].blobs['loss'].data)
                acc = float(solver.test_nets[0].blobs['accuracy'].data)
                tmp_losses.append(loss)
                tmp_accs.append(acc)
            loss = sum(tmp_losses)/len(tmp_losses)
            acc = sum(tmp_accs)/len(tmp_accs)
            test_loss.append(loss)
            test_acc.append(acc)
            np.save(snapshot_folder + "/training_loss", np.array(train_loss))
            np.save(snapshot_folder + "/test_loss", np.array(test_loss))
            np.save(snapshot_folder + "/test_acc", np.array(test_acc))

        if it % 100 == 0:
            print >>status, 'Iteration', it, 'completed.'
            print >>status, 'Training loss:', train_loss[-1]
            print >>status, 'Training accuracy:', train_acc[-1]
            print >>status, 'Test loss:', test_loss[-1]
            print >>status, 'Test accuracy:', test_acc[-1]

    np.save("training_loss", train_loss)
    np.save("test_loss", test_loss)
    print >>status, "done"

if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    solver_filename = sys.argv[2]
    
    i = solver_filename.index("/") + 1
    j = solver_filename.index(".")
    snapshot_folder = "snapshot_" + solver_filename[i:j]
    status = open(snapshot_folder + '/status.txt', 'w', 0) # own log file
    print >>status, "beginning training"
    
    train(gpu_id, solver_filename)
