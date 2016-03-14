import os
# os.environ['GLOG_minloglevel'] = '0' # enable logging
import numpy as np
import caffe
import caffe.io
import sys
import my_pycaffe as mp
import tempfile
import copy

def to_tempfile(file_content):
    """Serialize a Python protobuf object str(proto), dump to a temporary file,
       and return its filename."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file_content)
        return f.name

def get_split(split):
    filename = './development_kit/data/%s.txt' % split
    if not os.path.exists(filename):
        raise IOError('Split data file not found: %s' % split)
    return filename

deploy_network = "./vgg/deploy_deeper_baseline_multicrop.prototxt"
deploy_weights = "./snapshot_solver_deeper/vgg_net_iter_25000.caffemodel"
OUTPUTS = 25 # outputs per input image
BATCH = 8

# deploy_network = "./models/baseline-deploy.prototxt"
# deploy_weights = "./models/baseline.caffemodel"

def eval_net(split, K=5):
    print 'Running evaluation for split:', split
    filenames = []
    labels = []
    split_file = get_split(split)
    with open(split_file, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            assert 1 <= len(parts) <= 2, 'malformed line'
            filenames.append(parts[0])
            if len(parts) > 1:
                labels.append(int(parts[1]))
    known_labels = (len(labels) > 0)
    print "known", known_labels
    if known_labels:
        assert len(labels) == len(filenames)
    else:
        # create file with 'dummy' labels (all 0s)
        split_file = to_tempfile(''.join('%s 0\n' % name for name in filenames))
        print "split", split_file

    if split == "test":
        net = caffe.Net(deploy_network, deploy_weights, caffe.TEST)
    if split == "val":
        net = caffe.Net(deploy_network, deploy_weights, caffe.TRAIN)

    top_k_predictions = np.zeros((len(filenames), K), dtype=np.int32)
    if known_labels:
        correct_label_probs = np.zeros(len(filenames))
    offset = 0
    while offset < len(filenames):
        probs = copy.deepcopy(net.forward()['probs'])

        for inp in range(0, BATCH * OUTPUTS, OUTPUTS):
            prob = np.mean(probs[inp:inp+OUTPUTS, :], 0)
            # print inp, prob
            top_k_predictions[offset] = (-prob).argsort()[:K]
            # below is more efficient, but sorting 100 should be nothing
            # top_k_ind_unsorted = np.argpartition(prob, -K)[-K:]
            # top_k_predictions[offset] = np.argsort(-prob[top_k_ind_unsorted])
            if known_labels:
                correct_label_probs[offset] = prob[labels[offset]]
            offset += 1
            if offset >= len(filenames):
                break
        print offset
        if offset >= len(filenames):
            break
    if known_labels:
        def accuracy_at_k(preds, labels, k):
            assert len(preds) == len(labels)
            num_correct = sum(l in p[:k] for p, l in zip(preds, labels))
            return float(num_correct) / len(preds)
        for k in [1, K]:
            accuracy = 100 * accuracy_at_k(top_k_predictions, labels, k)
            print '\tAccuracy at %d = %4.2f%%' % (k, accuracy)
        cross_ent_error = -np.log(correct_label_probs).mean()
        print '\tSoftmax cross-entropy error = %.4f' % (cross_ent_error, )
    else:
        print 'Not computing accuracy; ground truth unknown for split:', split
    filename = 'top_%d_predictions.%s.csv' % (K, split)
    with open(filename, 'w') as f:
        f.write(','.join(['image'] + ['label%d' % i for i in range(1, K+1)]))
        f.write('\n')
        f.write(''.join('%s,%s\n' % (image, ','.join(str(p) for p in preds))
                        for image, preds in zip(filenames, top_k_predictions)))
    print 'Predictions for split %s dumped to: %s' % (split, filename)

if __name__ == "__main__":
    caffe.set_mode_gpu()
    caffe.set_device(0)

    print 'Evaluating...\n'

    for split in ('val', 'test'):
        eval_net(split)
        print
    print 'Evaluation complete.'
