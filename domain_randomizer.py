import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from glob import glob
from os import path


batch_size: int = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

class DomainRandomizerDataset(Dataset):
    def __init__(self, images_dir: str, backgrounds_dir: str, annotations_dir: str, transform=None):
        self.images: list[str] = glob(path.join(images_dir, "*.png"))
        self.backgrounds: list[str] = glob(path.join(backgrounds_dir, "*"))
        self.annotations: list[str] = glob(path.join(annotations_dir, "*.txt"))
        self.length: int = len(self.images) * len(self.backgrounds)
        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        bg_index: int = index // len(self.images)
        img_index: int = index % len(self.images)

        bg_path: str = self.backgrounds[bg_index]
        img_path: str = self.images[img_index]
        ann_path: str = self.annotations[img_index]

        bg = decode_image(bg_path).to(dtype=torch.float32) / 255.0
        img = decode_image(img_path).to(dtype=torch.float32) / 255.0
        return bg, img

def main() -> None:
    dataset = DomainRandomizerDataset("images", "backgrounds", "annotations")
    dataloader = DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    main()

