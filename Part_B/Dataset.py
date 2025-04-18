
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import subprocess
import zipfile
class Nature12KDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=64, image_size=(512, 512), data_aug=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_aug = data_aug

    def prepare_data(self):
        if os.path.exists(self.data_dir):
            print("✅ Dataset already prepared.")
            return

        zip_path = "iNaturalist.zip"
        url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"

        if not os.path.exists(zip_path):
            print("📥 Downloading dataset...")
            subprocess.run(["curl", "-o", zip_path, "-L", url], check=True)
        else:
            print("✅ Zip file already exists.")

        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        os.rename("inaturalist_12K", self.data_dir)
        os.rename(os.path.join(self.data_dir, "val"), os.path.join(self.data_dir, "test"))

    @staticmethod
    def get_transform(image_size, data_aug=False):
        transform_list = [transforms.Resize(image_size)]

        if data_aug:
            transform_list += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ]

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        return transforms.Compose(transform_list)

    def setup(self, stage=None):
        train_transform = self.get_transform(self.image_size, self.data_aug)
        test_transform = self.get_transform(self.image_size, False)

        full_train = datasets.ImageFolder(os.path.join(self.data_dir, "train"), transform=train_transform)
        test_set = datasets.ImageFolder(os.path.join(self.data_dir, "test"), transform=test_transform)

        val_size = int(0.2 * len(full_train))
        train_size = len(full_train) - val_size
        self.train_set, self.val_set = random_split(full_train, [train_size, val_size])
        self.test_set = test_set
        self.class_names = full_train.classes

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
