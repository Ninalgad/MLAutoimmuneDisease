import torch
from torch.utils.data import Dataset, IterableDataset

from h5_utils import read_assets_from_h5
from utils import load_adata


#########
# Torch Dataset & Embeddings
#########


def create_dataloader(tile_h5_paths, expr_paths, genes, normalize, img_transform, size_subset,
                      batch_size=8, training=False, num_workers=0):
    tile_dataset = H5Dataset(tile_h5_paths, expr_paths, genes, normalize, repeat=training,
                             chunk_size=size_subset, img_transform=img_transform)
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


class H5Dataset(torch.utils.data.IterableDataset):

    def __init__(self, h5_paths, expr_paths, genes, normalize, repeat,
                 img_transform=None, chunk_size=1000):
        self.h5_paths = h5_paths
        self.expr_paths = expr_paths
        self.genes = genes
        self.normalize = normalize
        self.img_transform = img_transform
        self.chunk_size = chunk_size
        self.n_paths = len(h5_paths)
        self.repeat = repeat

    def _assign_assets(self, idx):
        self.assets, _ = read_assets_from_h5(self.h5_paths[idx])
        barcodes = self.assets['barcode'].flatten().astype(str).tolist()
        adata = load_adata(self.expr_paths[idx], genes=self.genes,
                           barcodes=barcodes, normalize=self.normalize)
        self.assets['adata'] = adata.values

    def __len__(self):
        return self.n_paths * self.chunk_size

    def _gen(self):
        for i in range(self.n_paths):
            self._assign_assets(i)

            for j in range(self.chunk_size):
                item = {k: torch.tensor(v[j], dtype=torch.float32)
                        for (k, v) in self.assets.items() if k != 'barcode'}

                if self.img_transform is not None:
                    item['img'], item['mask'] = self.img_transform(image=item['img'], mask=item['mask'])

                yield preprocess_batch(item)

    def __iter__(self):
        return self._gen()

    def __getitem__(self, idx):
        pass
