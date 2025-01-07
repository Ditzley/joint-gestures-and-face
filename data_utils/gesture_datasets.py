from data_utils.show_data_module import SHOWDataModule
from typing import List, Union

def get_gesture_dataset(dataset: str, data_root: str, batch_size: int, num_frames: int, speakers: List[Union[str, int]], **kwargs):
    if dataset == 'show':
        return SHOWDataModule(
            data_root,
            batch_size,
            num_frames,
            speakers,
            **kwargs
        )

