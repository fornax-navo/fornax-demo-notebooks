import h5py as h5
from torch.utils import data
class galaxydata(data.Dataset):
    
    def __init__(self, archive, transform=None):
        self.archive = h5.File(archive, 'r')
        self.labels = self.archive['train_labels']
        self.data = self.archive['train_img']
        self.transform = transform
    
    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, self.labels[index]
    
    def __len__(self):
        return len(self.labels)
    
    def close(self):
        self.archive.close()