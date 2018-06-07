import numpy as np
import h5py


def load_train_data(dataset):
    """
    dataset options: 'train', 'dev', 'eval'
    """
    
    data_dict = {'train': 3014, 'dev': 1710,'eval': 13306}
    
    data_size = data_dict[dataset]
    
    f = h5py.File('/home/kasorn/anti_spoof/cqcc_feature.mat', mode='r')
    train_data = f.get('{}FeatureCell'.format(dataset)).value[0]
    
    da = np.empty((data_size, 90))

    for i in train_data:
        d = np.array(f[i])
        np.append(da, d, axis=0)

    return da
