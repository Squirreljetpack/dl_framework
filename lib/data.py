from sklearn.model_selection import KFold, ShuffleSplit, TimeSeriesSplit
import torch
from lib.Utils import *
import polars as pl
import numpy as np
import torch.utils.data as td
import pandas as pd


class DataModule(Base):
    """The base class of td."""

    ## Data Examples section

    # copy this signature when inheriting
    def __init__(self, batch_size=32, num_workers=1):
        super().__init__()
        self.save_parameters()

        # For folds
        self.data_inds = None
        self._folds = None
        self._folder = None

        # Subclasses define self.data = (X, y), or (train_set, val_set) directly
        # self.configure_folds('ShuffleSplit', num_splits = 1, test_size = 0.1)

    def to_tensors(arr, indices=slice(0, None)):
        """Coerce into Tensors for use with TensorDataset.
        Must be a slice if df currently."""

        if isinstance(arr[0], torch.Tensor):
            # Concatenates tensors (into dataset), i.e. batch[-1] is passed as y for loss computation by DataLoader
            tensors = tuple(a[indices] for a in arr)
        elif isinstance(arr[0], np.ndarray):
            tensors = tuple(a[indices] for a in arr)
            tensors = tuple(torch.tensor(a[indices], dtype=torch.float32) for a in arr)
        else:
            start = indices.start or 0
            end = indices.stop or -1
            if isinstance(arr[0], pl.DataFrame):
                tensors = tuple(
                    a.slice(start, end).to_torch(dtype=pl.Float32) for a in arr
                )
            elif isinstance(arr[0], pd.DataFrame):
                tensors = tuple(
                    torch.tensor(a.iloc[start:end, :].values, dtype=torch.float32)
                    for a in arr
                )
            else:
                raise Exception
        return tensors

    def _dataset(self, tensors):
        """Converts tensor tuples to dataset"""
        return td.TensorDataset(*tensors)

    def fit_transforms(self, tensors):
        def transform(t):
            return lambda x: x

        return tuple(transform(t) for t in tensors)

    def transform_val(self, tensors, transforms):
        for i, t in enumerate(tensors):
            transforms[i](t)

    def get_dataloaders(self, train_set=None, val_set=None, val_kwargs=None, **kwargs):
        """returns train_dataloader, val_dataloader
        **kwargs: passed to train DataLoader
        val_kwargs: passed to val DataLoader, **kwargs by default"""

        # If Dataset is directly defined, we just need to pass it to a DataLoader
        train_set = (
            train_set
            if isinstance(train_set, td.Dataset)
            else getattr(self, "train_set", None)
        )
        val_set = (
            val_set
            if isinstance(val_set, td.Dataset)
            else getattr(self, "val_set", None)
        )

        # Create Dataset from self.data(*X, y) by converting to tensors
        # get split from indices
        if train_set is None:
            # initialize indices if not already
            if not self.data_inds:
                if self._folds:
                    self.data_inds = next(self._folds)
                else:
                    # If self._folds is empty (i.e., folds unconfigured), uses unrandomized 80/20 split.
                    split = len(self.data[0]) * 4 // 5
                    self.data_inds = (slice(0, split), slice(split, None))

            train_tensors = DataModule.to_tensors(self.data, self.data_inds[0])
            transforms = self.fit_transforms(train_tensors)

            if not val_set:
                val_tensors = DataModule.to_tensors(self.data, self.data_inds[1])
                self.transform_val(val_tensors, transforms)

            train_set, val_set = (
                self._dataset(train_tensors),
                val_set if val_set is not None else self._dataset(val_tensors),
            )

        # Allow setting some dataloader args on data class itself
        for kwarg in ["sampler", "batch_sampler", "batch_size", "num_workers"]:
            a = getattr(self, kwarg, None)
            if a is not None:
                kwargs[kwarg] = a
        if val_kwargs is None:
            val_kwargs = kwargs

        # shuffle=True shuffles the data after every epoch
        return td.DataLoader(
            train_set, shuffle=getattr(self, "shuffle", True), **kwargs
        ), None if val_set is None else td.DataLoader(
            val_set, shuffle=False, **val_kwargs
        )

    def configure_folds(self, split_method, **kwargs):  #
        """Configure folds

        Args:
            split_method (string): KFold|TimeSeriesSplit|ShuffleSplit
        Kwargs:
            n_splits (int)
            random_state (bool)
            shuffle (bool)
            test_size (float): ShuffleSplit percentage in [0, 1]

        Raises:
            Exception: Unsupported fold method
        """
        if split_method == "KFold":
            self._folder = KFold(**kwargs)
        elif split_method == "TimeSeries":
            self._folder = TimeSeriesSplit(**kwargs)
        elif split_method == "ShuffleSplit":
            self._folder = ShuffleSplit(**kwargs)
        else:
            raise Exception
        self._folds = self._folder.split(self.data[0])

    # todo: shuffle when train_set is given
    def folds(self):
        """Provides an iterator that changes self.data_inds, used by self.get_dataloaders(), using iterator provided by self._folds, to be used to loop over folds.

        >>> t = DataModule()
        >>> t.data = ([1, 2, 3], [1, 2, 3])
        >>> t.configure_folds('KFold', n_splits=3)
        >>> for i in t.folds(): print(t.data_inds)
        (array([1, 2]), array([0]))
        (array([0, 2]), array([1]))
        (array([0, 1]), array([2]))
        """
        if not self._folds:
            logging.debug(f"Reinitializing folds on data of length {len(self.data[0])}")
            self._folds = self._folder.split(self.data[0])
        fold_num = 0
        while True:
            try:
                self.data_inds = next(self._folds)
                fold_num += 1
                yield fold_num
            except StopIteration:
                return

    ## Data Preview section

    def sample_batch(self, train=True):
        train_loader, val_loader = self.get_dataloaders()
        if train:
            return next(iter(train_loader))
        else:
            return next(iter(val_loader))

    def preview_batch(self, batch, num_samples=5):
        # Print shapes for all tensors in the batch
        print("Constituent shapes:")
        for i, tensor in enumerate(batch):
            print(f"batch[{i}]: {tensor.shape}")

        # Print sample values
        num_samples = min(num_samples, len(batch[0]))
        print(f"\nFirst {num_samples} samples:")
        for i in range(num_samples):
            print(f"\nSample {i}: ")
            for j in batch:
                print(f"\n{j[i].squeeze()}")

    def preview(self, num_samples=5):
        """
        Preview the data by showing dimensions and sample rows from both training and validation sets.

        Args:
            num_samples (int): Number of samples to display from each dataset
        """

        train = True
        for name in ["Train", "Validation"]:
            b = self.sample_batch(train=train)
            print(f"\n{name} Data Preview:")
            print("-" * 50)
            self.preview_batch(b, num_samples)
            train = False

    def visualize(self, num_images=8, nrows=1, figsize=(15, 3)):
        """
        Visualize image data in a grid layout.

        Args:
            num_images (int): Number of images to display
            nrows (int): Number of rows in the grid
            figsize (tuple): Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        import torchvision.utils as vutils

        # Get a batch of data
        train_loader, _ = self.get_dataloaders()
        if train_loader is None:
            print("No training loader available")
            return

        batch = next(iter(train_loader))
        images = batch[0]  # Assume first tensor contains images

        # Check if we're dealing with image data
        if len(images.shape) != 4:
            print(
                "Data doesn't appear to be image tensors (expected shape: [B, C, H, W])"
            )
            return

        # Determine if images need normalization
        if images.min() < 0 or images.max() > 1:
            images = (images - images.min()) / (images.max() - images.min())

        # Calculate grid layout
        ncols = (num_images + nrows - 1) // nrows

        # Create figure
        plt.figure(figsize=figsize)

        # Make grid of images
        grid = vutils.make_grid(
            images[:num_images], nrow=ncols, padding=2, normalize=False
        )

        # Display grid
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis("off")

        # Add labels if available
        if len(batch) > 1:
            labels = batch[1][:num_images]
            title = f"Labels: {', '.join(str(label.item()) for label in labels)}"
            plt.title(title)

        plt.tight_layout()
        plt.show()

    def describe(self):
        """
        Provide statistical description of the dataset.
        """
        train_loader, val_loader = self.get_dataloaders()

        def analyze_loader(loader, name):
            if loader is None:
                print(f"No {name} loader available")
                return

            print(f"\n{name} Dataset Statistics:")
            print("-" * 50)

            # Get first batch
            batch = next(iter(loader))

            # For each tensor in the batch
            for i, tensor in enumerate(batch):
                print(f"\nTensor {i}:")
                print(f"Shape: {tensor.shape}")
                print(f"Type: {tensor.dtype}")
                print(f"Min: {tensor.min().item():.4f}")
                print(f"Max: {tensor.max().item():.4f}")
                print(f"Mean: {tensor.mean().item():.4f}")
                print(f"Std: {tensor.std().item():.4f}")

                # If tensor is 1D or 2D, show unique values count
                if tensor.ndim <= 2:
                    unique_count = len(torch.unique(tensor))
                    print(f"Unique values: {unique_count}")

        analyze_loader(train_loader, "Training")
        analyze_loader(val_loader, "Validation")


class SeqDataset(td.Dataset, Base):
    # gap and samples Not Implemented yet
    def __init__(self, data, seq_len: int, y_len=1, gap=1, samples=-1) -> None:
        self.save_attr()

    def __len__(self) -> int:
        return len(self.data[0]) - self.seq_len

    def __getitem__(self, i):
        end = i + self.seq_len
        return torch.FloatTensor(self.data[0][i:end]), torch.FloatTensor(
            self.data[1][end + 1 - self.y_len : end + 1]
        )


class DataModuleFromLoader(DataModule):
    def __init__(self, train_loader, val_loader, batch_size=32, num_workers=1):
        super().__init__()
        self.save_attr()

        # For folds
        self.data_inds = None
        self._folds = None

    def get_dataloaders(self, train_set=None, val_set=None, val_kwargs=None, **kwargs):
        return self.train_loader, self.val_loader
