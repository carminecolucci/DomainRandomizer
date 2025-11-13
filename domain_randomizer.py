import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image, write_jpeg, ImageReadMode
from torchvision.transforms.v2 import functional as TF

import yaml


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_jpg(img: torch.Tensor, dir: str, path: str, index: int) -> None:
    outpath = os.path.join(dir, f"{os.path.basename(path).split(".")[0]}_{index}.jpg")
    if img.dtype != torch.uint8:
        img = (img * 255).to(device="cpu", dtype=torch.uint8)
    write_jpeg(img[:3, :, :], outpath)      # ignore alpha channel if present

def randint(low: int, high: int) -> int:
    return int(torch.randint(low, high, (1,)).item())

def randf(low: float, high: float) -> float:
    num = torch.rand(1).item()
    return low + num * (high - low)

def paste_image(bg: torch.Tensor, img: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    Paste image on background at position pos
    """
    bg_size = torch.tensor([*bg.shape[-2:]], dtype=torch.int32)
    img_size = torch.tensor([*img.shape[-2:]], dtype=torch.int32)

    y1, x1 = torch.min(bg_size, pos + img_size)
    y0, x0 = pos
    if x0 >= x1 or y0 >= y1:
        return bg                       # pasting outside background has no effect

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

def transform(bg: torch.Tensor, img: torch.Tensor, angle: int, scale: float, paste_pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform the image and apply the given background

    Args:
        bg (Tensor): the background of the output image
        img (Tensor): image to transform
        angle (int): angle in degrees to rotate the image
        scale (float): scale factor for the image
        paste_pos (Tensor) optional: coordinates (y, x) to paste the image.
            If None, a random position will be chosen such that the rotated and scaled
            image will fit into the background.
    Return:
        new_img (Tensor)
        paste_pos (Tensor)
    """
    img = TF.rotate(img, angle, TF.InterpolationMode.BILINEAR, expand=True)

    bg_size = torch.tensor([*bg.shape[-2:]], dtype=torch.int32)
    img_size = torch.tensor([*img.shape[-2:]], dtype=torch.int32)

    bg_min_size = torch.min(bg_size).item()
    img_min_size = torch.min(img_size).item()
    size = min(int(scale * img_min_size), bg_min_size - 1)

    img = TF.resize(img, [size], TF.InterpolationMode.BILINEAR, max_size=bg_min_size)

    if paste_pos is None:
        img_size = torch.tensor([*img.shape[-2:]], dtype=torch.int32)
        max_size = torch.max(torch.zeros(2, dtype=torch.int32), bg_size - img_size).tolist()
        paste_y = torch.randint(0, max_size[0], (1,))
        paste_x = torch.randint(0, max_size[1], (1,))
        paste_pos = torch.tensor([paste_y, paste_x])
    new_img = paste_image(bg, img, paste_pos)

    return new_img, paste_pos


class DomainRandomizerDataset(Dataset):
    def __init__(self, config: dict):
        backgrounds_dir: str = config["backgrounds_dir"]
        images_dir: str = config["images_dir"]
        output_dir: str = config.get("output_dir", "output")

        self.backgrounds: list[str] = [os.path.join(backgrounds_dir, fname) for fname in os.listdir(backgrounds_dir)]
        self.images: list[str] = [os.path.join(images_dir, fname) for fname in sorted(os.listdir(images_dir))]
        self.length: int = len(self.backgrounds) * len(self.images)

        self.has_segmasks: bool = "segmasks" in config
        if self.has_segmasks:
            self.segmasks = config["segmasks"]
            masks_dir = self.segmasks["masks_dir"]
            if self.segmasks["mode"] == "file":
                self.masks: list[str] = [os.path.join(masks_dir, fname) for fname in sorted(os.listdir(masks_dir))]
            self.output_masks_dir = os.path.join(output_dir, masks_dir)
            os.makedirs(self.output_masks_dir, exist_ok=True)

        self.output_images_dir = os.path.join(output_dir, images_dir)
        os.makedirs(self.output_images_dir, exist_ok=True)

        self.transform = config["transform"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        bg_index: int = index // len(self.images)
        img_index: int = index % len(self.images)

        bg_path: str = self.backgrounds[bg_index]
        img_path: str = self.images[img_index]

        bg = decode_image(bg_path).to(dtype=torch.float32) / 255.0
        img = decode_image(img_path).to(dtype=torch.float32) / 255.0

        if img.shape[0] == 4:
            alpha = img[3:4]
            img[0:3] *= alpha

        max_angle = self.transform["max_angle"]
        min_scale, max_scale = self.transform["scale"]

        angle = randint(-max_angle, max_angle + 1)
        scale = randf(min_scale, max_scale)
        new_img, paste_pos = transform(bg, img, angle, scale)
        save_jpg(new_img, self.output_images_dir, img_path, index)

        if self.has_segmasks:
            mask = self.load_segmask(img, img_index)
            black_bg = torch.zeros_like(bg)
            new_mask, _ = transform(black_bg, mask, angle, scale, paste_pos)
            new_mask = new_mask.to(dtype=torch.uint8)
            save_jpg(new_mask, self.output_masks_dir, img_path, index)

        return bg, img

    def load_segmask(self, img: torch.Tensor, index: int) -> torch.Tensor:
        """
        Load the segmentation mask associated with img.
        The mask is loaded from file or extracted from the alpha channel of the image, according to segmasks mode.
        Return the mask as a Tensor with dtype uint8.
        """
        mode = self.segmasks["mode"]
        if mode == "alpha":             # extract alpha component from image
            if img.shape[0] == 4:       # RGBA
                alpha = img[3:4, :, :] * 255.0
            else:                       # RGB
                alpha = torch.ones((1, img.shape[1], img.shape[2]), device=img.device)
            mask = (alpha > 0).to(dtype=torch.uint8)
        elif mode == "file":            # load mask from file
            mask = decode_image(self.masks[index], ImageReadMode.RGB)
        else:
            raise ValueError(f"Invalid mode {mode} for segmentation masks")

        return mask


def main() -> None:
    with open("config.yaml") as fp:
        config = yaml.safe_load(fp)

    dataset = DomainRandomizerDataset(config)
    ds_config = config["dataset"]
    dataloader = DataLoader(dataset, batch_size=ds_config["batch_size"], num_workers=ds_config["num_workers"])

    for batch_nr, batch in enumerate(dataloader):
        print("batch", batch_nr, "\tbg", batch[0].shape, "\timg", batch[1].shape)

if __name__ == "__main__":
    print(f"Running on {device}")
    main()

