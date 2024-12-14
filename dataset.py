import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from h5_utils import read_assets_from_h5
from utils import load_adata


def create_h5_dataloader(tile_h5_paths, expr_paths, genes, normalize, img_transform,
                         batch_size=8, shuffle=False, num_workers=0):
    tile_dataset = H5Dataset(tile_h5_paths, expr_paths, genes, normalize, shuffle=shuffle,
                             img_transform=img_transform)
    tile_dataloader = torch.utils.data.DataLoader(
        tile_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return tile_dataloader


def preprocess_batch(batch):
    batch['img'] = batch['img'].permute(2, 0, 1)
    batch['mask'] = batch['mask'].permute(2, 0, 1)

    batch['mask'] = torch.clip(batch['mask'], 0, 1)

    return batch


def preprocess_image(img):
    return img.astype('float32') / 255


class H5Dataset(IterableDataset):

    def __init__(self, h5_paths, expr_paths, genes, normalize, shuffle=False,
                 img_transform=None):
        self.h5_paths = h5_paths
        self.expr_paths = expr_paths
        self.genes = genes
        self.normalize = normalize
        self.img_transform = img_transform
        self.n_paths = len(h5_paths)
        self.shuffle = shuffle

        self._get_sizes()

    def _get_sizes(self):
        self.chunk_sizes = []
        for i in range(self.n_paths):
            self._assign_assets(i)
            self.chunk_sizes.append(len(self.assets['barcode']))

    def _assign_assets(self, idx):
        self.assets, _ = read_assets_from_h5(self.h5_paths[idx])
        barcodes = self.assets['barcode'].flatten().astype(str).tolist()
        adata = load_adata(self.expr_paths[idx], genes=self.genes,
                           barcodes=barcodes, normalize=self.normalize)
        self.assets['adata'] = adata.values

    def __len__(self):
        return sum(self.chunk_sizes)

    def _gen(self):
        for i in range(self.n_paths):
            self._assign_assets(i)

            size = self.chunk_sizes[i]
            for j in range(size):
                if self.shuffle:
                    j = np.random.choice(size)

                barcode = self.assets['barcode'][j].item().decode('UTF-8')
                barcode = int(barcode[1:])

                item = {'img': preprocess_image(self.assets['img'][j]),
                        'adata': self.assets['adata'][j],
                        'mask': self.assets['mask'][j] == barcode}
                item = {k: torch.tensor(v, dtype=torch.float32)
                        for (k, v) in item.items()}

                if self.img_transform is not None:
                    item['img'], item['mask'] = self.img_transform(image=item['img'], mask=item['mask'])

                yield preprocess_batch(item)

    def __iter__(self):
        return self._gen()

    def __getitem__(self, idx):
        pass


class DictDataset(Dataset):
    def __init__(self, dat, dtype=None):
        self.dat = dat
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

    def __len__(self):
        return len(list(self.dat.values())[0])

    def __getitem__(self, idx):
        item = {k: v[idx] for (k, v) in self.dat.items()}
        item = {k: torch.tensor(v, dtype=self.dtype)
                for (k, v) in item.items()}
        return item


def create_dict_dataloader(data_dict, dtype=None, batch_size=8, shuffle=False, num_workers=0):
    dict_dataset = DictDataset(data_dict, dtype=dtype)
    dict_dataloader = torch.utils.data.DataLoader(
        dict_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return dict_dataloader
