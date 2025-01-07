from typing import List

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import lightning as L
import numpy as np
from data_utils.show_data_utils import SHOWDataset

def dtype_to_dtype(batch, d1, d2):
    if isinstance(batch, np.ndarray):
        return dtype_to_dtype(torch.from_numpy(batch), d1, d2)
    elif isinstance(batch, torch.Tensor) and batch.dtype == d1:
        return batch.to(dtype=d2)
    elif isinstance(batch, list):
        new_batch = [dtype_to_dtype(t, d1, d2) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(dtype_to_dtype(t, d1, d2) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: dtype_to_dtype(t, d1, d2) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def collate_fn(batch):
    new_batch = dtype_to_dtype(batch, torch.float64, torch.float32)
    return default_collate(new_batch)


class SHOWDataModule(L.LightningDataModule):
    def __init__(self, data_root: str, batch_size: int, num_frames: int, speakers: List[int], pkl_name: str, num_workers: int):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = data_root
        self.pkl_name = pkl_name
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.speakers = speakers
        self.num_workers = num_workers

    def get_data(self, split):
        dataset = SHOWDataset(
            self.data_root,
            self.pkl_name,
            split,
            self.speakers,
            limbscaling=False,
            normalization=False,
            norm_method="all",
            split_trans_zero=False,
            num_pre_frames=0,
            num_frames=self.num_frames,
            num_generate_length=self.num_frames,
            aud_feat_win_size=None,
            aud_feat_dim=64,
            feat_method="mfcc",
            context_info=False,
            smplx=True,
            audio_sr=16000,
            convert_to_6d=False,
            expression=True,
        )
        return dataset

    def setup(self, stage):
        self.train_dataset = self.get_data('train')
        self.test_dataset = self.get_data('test')


    def train_dataloader(self):
        return data.DataLoader(self.train_dataset,
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               shuffle=True,
                               drop_last=True,
                               pin_memory=True,
                               collate_fn=collate_fn)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset.all_dataset,
                               batch_size=1,
                               num_workers=0,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=collate_fn)

    def val_dataloader(self):
        return data.DataLoader(self.train_dataset,
                               batch_size=1,
                               num_workers=0,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=collate_fn)

    def predict_dataloader(self):
        return data.DataLoader(self.test_dataset,
                               batch_size=1,
                               num_workers=self.num_workers,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=collate_fn)