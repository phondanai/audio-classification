import numpy as np
import h5py


def is_zero_value_inside(data):
    """Check numpy array data if contains zero value, will return True.

    :param data: numpy array data
    """
    

    results = False
    for i in data:
        zero_count = np.count_nonzero(i == 0)
        if zero_count > 0:
            results = True

    return results


def load_train_data(dataset):
    """Load data from MATLAB `.dat` file and convert to numpy array.

    :param dataset: name of dataset can be 'train', 'dev' and  'eval'

    Usage::
        train_features = load_train_data('train')
        dev_features = load_train_data('dev')
        eval_features = load_train_data('eval')
    """
    
    # Not using these variable right now.
    #data_dict = {'train': 3014, 'dev': 1710,'eval': 13306}
    #data_size = data_dict[dataset]
    
    f = h5py.File('/home/kasorn/anti_spoof/cqcc_feature.mat', mode='r')
    train_data = f.get('{}FeatureCell'.format(dataset)).value[0]
    
    data = np.array([np.array(f[i]) for i in train_data])

    contain_zero = is_zero_value_inside(data)

    if contain_zero:
        print("Data is contains zero value")
        print("Abort")
        return None

    return data

# Usage example
#train_features = load_train_data('train')

#if isinstance(train_features, np.ndarray):
#    print(train_features[0])
#    save = input('Wanna save?: ')
#    if save == 'yes':
#        print("saving file to ", "/tmp/train_feat.npy")
#        np.save('/tmp/train_feat.npy', train_features)

