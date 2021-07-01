import os

import cv2
import numpy as np

from torch.utils.data import Dataset


class WildCamDataset(Dataset):

    def __init__(
            self,
            df,
            num_labels,
            root_path_data,
            stage='train',
            pre_processing=None,
            augmentations=None,
            post_processing=None) -> None:
        super().__init__()
        self.df = df
        self.num_labels = num_labels
        self.root_path_data = os.path.join(root_path_data, stage)
        self.pre_processing = pre_processing
        self.augmentations = augmentations
        self.post_processing = post_processing

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df[self.df.index == index]

        # get image
        image_path = os.path.join(self.root_path_data, sample.image_id)
        image = cv2.imread(image_path)

        # get ta
        target = np.zeros((self.num_labels))
        target[sample.category_id] = 1

        item_data = {'image': image, 'target': target}
        if self.pre_processing is not None:
            item_data = self.pre_processing(**item_data)
        if self.augmentations is not None:
            item_data = self.augmentations(**item_data)
        if self.post_processing is not None:
            item_data = self.post_processing(**item_data)

        return item_data
