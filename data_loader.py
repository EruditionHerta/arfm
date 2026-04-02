"""MNIST, CIFAR-10 and CelebA data loading utilities."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image


def load_mnist_from_raw(raw_dir: str = './data/MNIST/raw') -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST from raw idx files, returns (images [N,H,W], labels [N])."""
    import os
    import struct

    raw_dir = Path(raw_dir)

    def read_idx(filename: str) -> np.ndarray:
        """Read idx format file."""
        file_path = raw_dir / filename
        if not file_path.exists():
            gz_path = raw_dir / (filename + '.gz')
            if gz_path.exists():
                import gzip
                with gzip.open(gz_path, 'rb') as f:
                    data = f.read()
                with open(file_path, 'wb') as f:
                    f.write(data)
            else:
                raise FileNotFoundError(f"{file_path} not found")

        with open(file_path, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    train_images = read_idx('train-images-idx3-ubyte')
    train_labels = read_idx('train-labels-idx1-ubyte')
    test_images = read_idx('t10k-images-idx3-ubyte')
    test_labels = read_idx('t10k-labels-idx1-ubyte')

    images = np.concatenate([train_images, test_images], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    return images, labels


class MNISTDataset(Dataset):
    """MNIST dataset with optional normalization to [-1, 1]."""

    def __init__(
        self,
        root: str = './data/MNIST',
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        normalize: bool = True
    ):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.normalize = normalize

        images, labels = load_mnist_from_raw(str(self.root / 'raw'))

        if train:
            self.images = images[:60000]
            self.labels = labels[:60000]
        else:
            self.images = images[60000:]
            self.labels = labels[60000:]

        self.images = self.images.astype(np.float32) / 255.0
        if normalize:
            self.images = (self.images - 0.5) * 2.0

        self.images = self.images[:, np.newaxis, :, :]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[idx])
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_mnist_dataloaders(
    root: str = './data/MNIST',
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Get MNIST train and test data loaders."""
    train_dataset = MNISTDataset(
        root=root,
        train=True,
        normalize=normalize
    )

    test_dataset = MNISTDataset(
        root=root,
        train=False,
        normalize=normalize
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


class UnlabeledMNIST(Dataset):
    """Unlabeled MNIST dataset (for generative models)."""

    def __init__(
        self,
        root: str = './data/MNIST',
        train: bool = True,
        normalize: bool = True
    ):
        self.root = Path(root)
        self.train = train
        self.normalize = normalize

        images, _ = load_mnist_from_raw(str(self.root / 'raw'))

        if train:
            self.images = images[:60000]
        else:
            self.images = images[60000:]

        self.images = self.images.astype(np.float32) / 255.0
        if self.normalize:
            self.images = (self.images - 0.5) * 2.0

        self.images = self.images[:, np.newaxis, :, :]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.images[idx])


def get_unlabeled_dataloaders(
    root: str = './data/MNIST',
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Get unlabeled MNIST data loaders."""
    train_dataset = UnlabeledMNIST(root, train=True, normalize=normalize)
    test_dataset = UnlabeledMNIST(root, train=False, normalize=normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_labeled_dataloaders(
    root: str = './data/MNIST',
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Alias for get_mnist_dataloaders."""
    return get_mnist_dataloaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize
    )


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset with optional augmentation and normalization."""

    def __init__(
        self,
        root: str = './data/CIFAR10',
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        normalize: bool = True,
        use_augmentation: bool = False
    ):
        self.root = Path(root)
        self.train = train
        self.normalize = normalize
        self.use_augmentation = use_augmentation and train

        transform_list = []

        if self.use_augmentation:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        transform_list.append(transforms.ToTensor())

        self.base_transform = transforms.Compose(transform_list)
        self.extra_transform = transform

        self.cifar10 = datasets.CIFAR10(
            root=str(self.root),
            train=train,
            download=True,
            transform=None
        )

        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2470, 0.2435, 0.2616]

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.cifar10[idx]

        if isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            image_tensor = self.base_transform(image)

        if self.normalize:
            image_tensor = (image_tensor - 0.5) * 2.0

        if self.extra_transform is not None:
            image_tensor = self.extra_transform(image_tensor)

        return image_tensor, label


class UnlabeledCIFAR10(Dataset):
    """Unlabeled CIFAR-10 dataset (for generative models)."""

    def __init__(
        self,
        root: str = './data/CIFAR10',
        train: bool = True,
        normalize: bool = True,
        use_augmentation: bool = False
    ):
        self.root = Path(root)
        self.train = train
        self.normalize = normalize
        self.use_augmentation = use_augmentation and train

        transform_list = []

        if self.use_augmentation:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        transform_list.append(transforms.ToTensor())
        self.base_transform = transforms.Compose(transform_list)

        self.cifar10 = datasets.CIFAR10(
            root=str(self.root),
            train=train,
            download=True,
            transform=None
        )

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = self.cifar10[idx]

        if isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            image_tensor = self.base_transform(image)

        if self.normalize:
            image_tensor = (image_tensor - 0.5) * 2.0

        return image_tensor


def get_cifar10_dataloaders(
    root: str = './data/CIFAR10',
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
    use_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test data loaders."""
    train_dataset = CIFAR10Dataset(
        root=root,
        train=True,
        normalize=normalize,
        use_augmentation=use_augmentation
    )

    test_dataset = CIFAR10Dataset(
        root=root,
        train=False,
        normalize=normalize,
        use_augmentation=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_unlabeled_cifar10_dataloaders(
    root: str = './data/CIFAR10',
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
    use_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Get unlabeled CIFAR-10 data loaders."""
    train_dataset = UnlabeledCIFAR10(
        root, train=True, normalize=normalize, use_augmentation=use_augmentation
    )
    test_dataset = UnlabeledCIFAR10(
        root, train=False, normalize=normalize, use_augmentation=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_labeled_cifar10_dataloaders(
    root: str = './data/CIFAR10',
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
    use_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Alias for get_cifar10_dataloaders."""
    return get_cifar10_dataloaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        use_augmentation=use_augmentation
    )


CELEBA_ATTR_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
    'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
    'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
    'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]


class CelebADataset(Dataset):
    """CelebA dataset: center crop 178x178, resize to image_size, 40 binary attributes."""

    def __init__(
        self,
        root: str = './data/celeba',
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        normalize: bool = True,
        use_augmentation: bool = False,
        image_size: int = 64
    ):
        self.root = Path(root)
        self.split = split
        self.normalize = normalize
        self.use_augmentation = use_augmentation and split == 'train'
        self.image_size = image_size
        self.extra_transform = transform

        self.img_dir = self.root / 'img_align_celeba'

        attr_file = self.root / 'list_attr_celeba.txt'
        if not attr_file.exists():
            raise FileNotFoundError(f"Attribute file does not exist: {attr_file}")

        self.attributes, self.attr_names = self._load_attributes(attr_file)

        partition_file = self.root / 'list_eval_partition.txt'
        if not partition_file.exists():
            raise FileNotFoundError(f"Partition file does not exist: {partition_file}")

        self.indices = self._load_partition(partition_file, split)

        transform_list = [
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
        ]

        if self.use_augmentation:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        transform_list.append(transforms.ToTensor())
        self.base_transform = transforms.Compose(transform_list)

    def _load_attributes(self, attr_file: Path):
        """Load attribute file (line1: count, line2: names, rest: filename attrs)."""
        with open(attr_file, 'r') as f:
            lines = f.readlines()

        num_images = int(lines[0].strip())
        attr_names = lines[1].strip().split()

        attributes = {}
        for line in lines[2:]:
            parts = line.strip().split()
            filename = parts[0]
            attrs = [1 if int(x) == 1 else 0 for x in parts[1:]]
            attributes[filename] = torch.tensor(attrs, dtype=torch.long)

        return attributes, attr_names

    def _load_partition(self, partition_file: Path, split: str):
        """Load partition file (line1: count, rest: filename partition_label)."""
        split_map = {'train': 0, 'val': 1, 'test': 2}

        with open(partition_file, 'r') as f:
            lines = f.readlines()

        indices = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filename = parts[0]
            partition = int(parts[1])

            if partition == split_map[split]:
                if filename in self.attributes:
                    indices.append(filename)

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.indices[idx]

        img_path = self.img_dir / filename
        image = Image.open(img_path).convert('RGB')

        image_tensor = self.base_transform(image)

        if self.normalize:
            image_tensor = (image_tensor - 0.5) * 2.0

        attributes = self.attributes[filename]

        if self.extra_transform is not None:
            image_tensor = self.extra_transform(image_tensor)

        return image_tensor, attributes


class UnlabeledCelebA(Dataset):
    """Unlabeled CelebA dataset (for unconditional generation)."""

    def __init__(
        self,
        root: str = './data/celeba',
        split: str = 'train',
        normalize: bool = True,
        use_augmentation: bool = False,
        image_size: int = 64
    ):
        self.root = Path(root)
        self.split = split
        self.normalize = normalize
        self.use_augmentation = use_augmentation and split == 'train'
        self.image_size = image_size

        self.img_dir = self.root / 'img_align_celeba'

        partition_file = self.root / 'list_eval_partition.txt'
        self.indices = self._load_partition(partition_file, split)

        transform_list = [
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
        ]

        if self.use_augmentation:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        transform_list.append(transforms.ToTensor())
        self.base_transform = transforms.Compose(transform_list)

    def _load_partition(self, partition_file: Path, split: str):
        split_map = {'train': 0, 'val': 1, 'test': 2}

        with open(partition_file, 'r') as f:
            lines = f.readlines()

        indices = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filename = parts[0]
            partition = int(parts[1])

            if partition == split_map[split]:
                indices.append(filename)

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        filename = self.indices[idx]
        img_path = self.img_dir / filename
        image = Image.open(img_path).convert('RGB')

        image_tensor = self.base_transform(image)

        if self.normalize:
            image_tensor = (image_tensor - 0.5) * 2.0

        return image_tensor


def get_celeba_dataloaders(
    root: str = './data/celeba',
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
    use_augmentation: bool = False,
    image_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """Get CelebA train and test data loaders."""
    train_dataset = CelebADataset(
        root=root,
        split='train',
        normalize=normalize,
        use_augmentation=use_augmentation,
        image_size=image_size
    )

    test_dataset = CelebADataset(
        root=root,
        split='test',
        normalize=normalize,
        use_augmentation=False,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_unlabeled_celeba_dataloaders(
    root: str = './data/celeba',
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
    use_augmentation: bool = False,
    image_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """Get unlabeled CelebA data loaders."""
    train_dataset = UnlabeledCelebA(
        root, split='train', normalize=normalize,
        use_augmentation=use_augmentation, image_size=image_size
    )
    test_dataset = UnlabeledCelebA(
        root, split='test', normalize=normalize,
        use_augmentation=False, image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_dataloaders(
    dataset: str = 'mnist',
    root: str = None,
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
    labeled: bool = True,
    use_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Unified data loading entry point for mnist/cifar10/celeba."""
    if root is None:
        if dataset.lower() == 'celeba':
            root = './data/celeba'
        else:
            root = f'./data/{dataset.upper()}'

    if dataset.lower() == 'cifar10':
        if labeled:
            return get_cifar10_dataloaders(
                root=root,
                batch_size=batch_size,
                num_workers=num_workers,
                normalize=normalize,
                use_augmentation=use_augmentation
            )
        else:
            return get_unlabeled_cifar10_dataloaders(
                root=root,
                batch_size=batch_size,
                num_workers=num_workers,
                normalize=normalize,
                use_augmentation=use_augmentation
            )
    elif dataset.lower() == 'celeba':
        if labeled:
            return get_celeba_dataloaders(
                root=root,
                batch_size=batch_size,
                num_workers=num_workers,
                normalize=normalize,
                use_augmentation=use_augmentation
            )
        else:
            return get_unlabeled_celeba_dataloaders(
                root=root,
                batch_size=batch_size,
                num_workers=num_workers,
                normalize=normalize,
                use_augmentation=use_augmentation
            )
    else:
        if labeled:
            return get_mnist_dataloaders(
                root=root,
                batch_size=batch_size,
                num_workers=num_workers,
                normalize=normalize
            )
        else:
            return get_unlabeled_dataloaders(
                root=root,
                batch_size=batch_size,
                num_workers=num_workers,
                normalize=normalize
            )
