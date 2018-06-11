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


def load_data(dataset, avg=False):
    """Load data from MATLAB `.mat` file and convert to numpy array.

    :param dataset: name of dataset can be 'train', 'dev' and  'eval'
    :param avg:     Default is `False` if set to `True` the data will be reshape to (90,) by
                    averaging data                

    Usage::
        train_features = load_data('train')
        train_features_avg = load_data('train', avg=True)
        dev_features = load_data('dev')
        eval_features = load_data('eval')
    """
    
    # Not using these variable right now.
    #data_dict = {'train': 3014, 'dev': 1710,'eval': 13306}
    #data_size = data_dict[dataset]
    
    f = h5py.File('/home/kasorn/anti_spoof/cqcc_feature.mat', mode='r')
    mat_data = f.get('{}FeatureCell'.format(dataset)).value[0]
    if avg: 
        data = np.array([np.mean(np.array(f[i]), axis=0) for i in mat_data])
    else:
        data = np.array([np.array(f[i]) for i in mat_data])

    contain_zero = is_zero_value_inside(data)

    if contain_zero:
        print("Data is contains zero value")
        print("Abort")
        return None

    return data

# Usage example
#train_features = load_data('train', avg=True)

#if isinstance(train_features, np.ndarray):
#    print(train_features[0])
#    print(train_features[0].shape)
#    save = input('Wanna save?: ')
#    if save == 'yes':
#        print("saving file to ", "npy/cqcc_train_feat_2.npy")
#        np.save('npy/cqcc_train_feat_3.npy', train_features)

