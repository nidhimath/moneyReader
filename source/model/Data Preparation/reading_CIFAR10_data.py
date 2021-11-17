import pickle

def load_cfar10_batch():
    with open('/Users/Nidhi/Downloads/machine_learning-master/cifar10-ready-testData.bin', mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    print(len(batch['data']))

    features = batch['data'].reshape((int(len(batch['data'])/3072), 3, 32, 32)).transpose(0, 2, 3, 1)
    print(features)

    labels = batch['labels']

    return features, labels

load_cfar10_batch()