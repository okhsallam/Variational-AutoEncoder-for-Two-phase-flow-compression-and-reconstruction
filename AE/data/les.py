import os
import numpy as np
import torch
from torch.utils.data import Dataset

ARR_EXTENSIONS = ['.npy']

def is_array_file(filename):
    return any(filename.endswith(extension) for extension in ARR_EXTENSIONS)

def make_dataset(dir, filetype='array'):
    if os.path.isfile(dir):
        samples = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        samples = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if filetype == 'image':
                    raise NotImplementedError
                elif filetype == 'array':
                    if is_array_file(fname):
                        path = os.path.join(root, fname)
                        samples.append(path)
                else:
                    raise ValueError(f'Filetype of "{filetype}" not recognized')           

    return samples

def numpy_loader(path):
    return np.load(path)

def numpy_transforms(arr, data_bounds):
    '''
    Inputs: 
      - arr: Numpy array, where data comes from natural values
      - data_bounds: Tuple containing values for rescaling (umin, umax, vmin, vmax, wmin, wmax, vofmin, vofmax)
    Output: Torch tensor rescaled to [-1,1]
    '''

    # Rescale to [-1,1]
    if len(arr.shape) == 3:  # For 2D data at specific time instance [field, i_coord, j_coord]
        # Then rescale
        umin, umax, vmin, vmax, wmin, wmax, vofmin, vofmax = data_bounds
        arr[0,:,:] = 2*(arr[0,:,:]  - umin)/(umax - umin) - 1 #u - velocity
        arr[1,:,:] = 2*(arr[1,:,:]  - vmin)/(vmax - vmin) - 1 #v - velocity
        arr[2,:,:] = 2*(arr[2,:,:]  - wmin)/(wmax - wmin) - 1 #w - velocity
        arr[3,:,:] = 2*(arr[3,:,:]  - vofmin)/(vofmax - vofmin) - 1 #vof - volume of fluid

    elif len(arr.shape) == 4:  # For 2D data at all time instances [field, i_coord, j_coord, time]
        umin, umax, vmin, vmax, wmin, wmax, vofmin, vofmax = data_bounds
        arr[0,:,:,:] = 2*(arr[0,:,:,:]  - umin)/(umax - umin) - 1 #u - velocity
        arr[1,:,:,:]  = 2*(arr[1,:,:,:]  - vmin)/(vmax - vmin) - 1 #v - velocity
        arr[2,:,:,:]  = 2*(arr[2,:,:,:]  - wmin)/(wmax - wmin) - 1 #w - velocity
        arr[3,:,:,:]  = 2*(arr[3,:,:,:]  - vofmin)/(vofmax - vofmin) - 1 #vof - volume of fluid

    else:
        raise ValueError("Expected tuple containing (umin, umax, vmin, vmax, wmin, wmax, vofmin, vofmax)!")

    # numpy array -> torch tensor with correct dtype
    arr = torch.from_numpy(arr).float()

    return arr
#%%
class LESBase(Dataset):
    def __init__(self, data_root=None, data_bounds=None, image_size=None, apply_transforms=False):
        self.data_root = data_root
        self.data_bounds = data_bounds
        self.image_size = image_size
        self.loader = numpy_loader
        self.tfs = numpy_transforms
        self.apply_transforms = apply_transforms

        self.data = make_dataset(self.data_root, filetype='array')
        # if self.apply_transforms:
        #     self.data = [self.tfs(self.loader(path), self.data_bounds) for path in self.data]
        # else:
        #     self.data = [self.loader(path) for path in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.loader(self.data[index])
        if self.apply_transforms:
            data = self.tfs(data, self.data_bounds)
            # print('Data loaded and transformed on-the-fly.')
        else:
            # print('Data loaded on-the-fly without transformations.')
            pass
        return data

class LEStrain(LESBase):
    NAME = "LES_train"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LESvalidation(LESBase):
    NAME = "LES_validation"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LEStest(LESBase):
    NAME = "LES_test"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
