import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image, write_jpeg
from torchvision.transforms.v2 import functional as TF
from glob import glob
import os

batch_size: int = 64
num_workers: int = 4

max_angle: int = 30
min_scale: float = 0.2
max_scale: float = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_jpg(img: torch.Tensor, dir: str, path: str, index: int) -> None:
    outpath = os.path.join(dir, f"{os.path.basename(path).split(".")[0]}_{index}.jpg")
    img = (img * 255).to(device="cpu", dtype=torch.uint8)
    write_jpeg(img[:3, :, :], outpath)      # ignore alpha channel if present

def randint(low: int, high: int) -> int:
    return int(torch.randint(low, high, (1,)).item())

def randf(low: float, high: float) -> float:
    num = torch.rand((1,)).item()
    return low + num * (high - low)

def paste_image(bg: torch.Tensor, img: torch.Tensor, pos: tuple[int, int]) -> torch.Tensor:
    """
    Paste image on background at position pos
    """
    bg_h, bg_w = bg.shape[-2:]
    img_h, img_w = img.shape[-2:]

    x0, y0 = pos
    x1 = min(bg_w, x0 + img_w)
    y1 = min(bg_h, y0 + img_h)
    if x0 >= x1 or y0 >= y1:
        return bg       # pasting outside background has no effect

    dx, dy = x1 - x0, y1 - y0
    img_slice = img[:, :dy, :dx]
    if img_slice.shape[0] == 4:         # RGBA
        img_rgb = img_slice[:3, :, :]
        img_alpha = img_slice[3:4, :, :].to(dtype=torch.bool)
    else:                               # RGB
        img_rgb = img_slice
        img_alpha = torch.ones((1, img_rgb.shape[1], img_rgb.shape[2]), dtype=torch.bool, device=img_rgb.device)

    bg_slice = bg[:, y0:y1, x0:x1]
    out_slice = torch.where(img_alpha, img_rgb, bg_slice)
    output = bg.clone()
    output[:, y0:y1, x0:x1] = out_slice
    return output

def transform(bg: torch.Tensor, img: torch.Tensor, label: torch.Tensor, angle: int, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform the image and apply the given background
    Use the same transform on the label using a black background

    Args:
        bg (Tensor): the background of the output image
        img (Tensor): image to transform
        label (Tensor[1, image_height, image_width]): segmentation mask of the image
        angle (int): angle in degrees to rotate the image
        scale (float): scale factor for the image

    Return:
        new_img (Tensor)
        new_label (Tensor)
    """
    from torchvision.transforms import Resize
    img   = TF.rotate(img,   angle, TF.InterpolationMode.BILINEAR, expand=True)
    label = TF.rotate(label, angle, TF.InterpolationMode.BILINEAR, expand=True)

    # === RESIZE ===
    bg_h, bg_w   = bg.shape[-2:]
    bg_min_size  = min(bg_h, bg_w)
    img_h, img_w = img.shape[-2:]
    img_min_size = min(img_h, img_w)
    size = min(int(scale * img_min_size), bg_min_size - 1)

    img   = TF.resize(img,   [size], TF.InterpolationMode.BILINEAR, max_size=bg_min_size)
    label = TF.resize(label, [size], TF.InterpolationMode.BILINEAR, max_size=bg_min_size)

    # choose a random position on the background to apply the image
    img_h, img_w = img.shape[-2:]
    max_y = max(0, bg_h - img_h)
    max_x = max(0, bg_w - img_w)
    paste_x = randint(0, max_x)
    paste_y = randint(0, max_y)

    new_img = paste_image(bg, img, (paste_x, paste_y))

    if label.shape == (1, img_h, img_w):
        black_bg = torch.zeros_like(bg)
        new_label = paste_image(black_bg, label, (paste_x, paste_y))
    else:
        # TODO: add support for a yolo-like annotation?
        # check that the same transform apply correctly to the bbox/keypoints
        # also take a look at `rotate_bounding_box` and KeyPoints class
        new_label = label

    return (new_img, new_label)

class DomainRandomizerDataset(Dataset):
    def __init__(self, backgrounds_dir: str, images_dir: str, labels_dir: str, output_dir: str):
        self.backgrounds: list[str] = glob(os.path.join(backgrounds_dir, "*"))
        self.images: list[str] = sorted(glob(os.path.join(images_dir, "*.png")))
        # TODO: configure textual labels/segmentation mask in a config file
        # self.labels: list[str] = sorted(glob(os.path.join(labels_dir, "*.txt")))
        self.length: int = len(self.backgrounds) * len(self.images)

        self.output_images_dir = os.path.join(output_dir, images_dir)
        self.output_labels_dir = os.path.join(output_dir, labels_dir)
        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        bg_index: int = index // len(self.images)
        img_index: int = index % len(self.images)

        bg_path: str = self.backgrounds[bg_index]
        img_path: str = self.images[img_index]

        bg = decode_image(bg_path).to(dtype=torch.float32) / 255.0
        img = decode_image(img_path).to(dtype=torch.float32) / 255.0

        alpha = img[3:4, :, :]
        label = (alpha > 0).to(dtype=torch.uint8)

        angle = randint(-max_angle, max_angle + 1)
        scale = randf(min_scale, max_scale)
        new_img, new_label = transform(bg, img, label, angle, scale)
        save_jpg(new_img, self.output_images_dir, img_path, index)
        save_jpg(new_label, self.output_labels_dir, img_path, index)

        return bg, img, label

def main() -> None:
    backgrounds_dir = "backgrounds"
    images_dir = "images"
    labels_dir = "labels"
    output_dir = "output"
    dataset = DomainRandomizerDataset(backgrounds_dir, images_dir, labels_dir, output_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    for batch_nr, batch in enumerate(dataloader):
        print("batch", batch_nr, "\tbg", batch[0].shape, "\timg", batch[1].shape, "\tlabel", batch[2].shape)

if __name__ == "__main__":
    print(f"Running on {device}")
    main()

