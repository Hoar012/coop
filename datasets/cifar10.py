import math
import random
import os.path as osp

from dassl.utils import listdir_nohidden

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
    ]

@DATASET_REGISTRY.register()
class cifar10(DatasetBase):

    dataset_dir = "cifar10"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train_dir = osp.join(self.dataset_dir, "train")
        test_dir = osp.join(self.dataset_dir, "test")

        train_x = self._read_data_train(
            train_dir
        )
        test = self._read_data_test(test_dir)

        super().__init__(train_x=train_x, train_u=None, val=None, test=test)

    def _read_data_train(self, data_dir):
        items = []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, str(label))
            imnames = listdir_nohidden(class_dir)

            for imname in imnames:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=class_name)
                items.append(item)

        return items

    def _read_data_test(self, data_dir):
        items = []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, str(label))
            imnames = listdir_nohidden(class_dir)

            for imname in imnames:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label)
                items.append(item)

        return items
