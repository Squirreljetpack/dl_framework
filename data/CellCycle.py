from lib.data import ImageData, DataConfig
from lib.datasets.image import ImageDataset
from lib.utils import *
from lib.dfs import *


from sklearn.model_selection import train_test_split, StratifiedKFold


class Brightfield(ImageData):
    def __init__(self, c: DataConfig):
        super().__init__()
        self.save_config(c)

        def label_fn(folder):
            classes_to_merge = ["G1", "G2", "S"]  # Classes to merge
            merged_class_name = "G1_G2_S"
            return merged_class_name if folder in classes_to_merge else folder

        data, labels = self._paths_and_labels_from_folder(
            "resources/CellCycle", "*_Ch3.ome.jpg", label_fn
        )

        data, test_data, labels, test_labels = train_test_split(
            data, labels, test_size=0.1, stratify=labels, random_state=42
        )

        self.set_data(data, labels)
        self.set_data(test_data, test_labels, test=True)

        self.set_folds(StratifiedKFold(n_splits=5, shuffle=True))

    def _to_dataset(self, data):
        return ImageDataset(data[0], data[1], transform=self.transform)
