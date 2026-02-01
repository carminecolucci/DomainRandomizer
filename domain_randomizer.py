import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image, write_jpeg, ImageReadMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF

import yaml


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# NOTE: Photometric transformations only change the pixel intensity of images.
# As such, they are not applied to masks
photometric_transforms = [
    lambda x: x,    # "identity" transform
    T.ColorJitter(0.3, 0, 0.3, 0.1),
    T.GaussianBlur((3, 3)),
    T.GaussianNoise()
]
# TODO: also add geometric transformations

num_transforms = len(photometric_transforms)

def batch_transform(config, bgs: torch.Tensor, imgs: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_angle = config["max_angle"]
    min_scale, max_scale = config["scale"]

    B, C, H, W = bgs.shape
    output_imgs = bgs.repeat_interleave(num_transforms, dim=0)
    output_masks = torch.zeros((B * num_transforms, 1, H, W), device=device)

    for j in range(num_transforms):
        for i in range(B):
            img  = imgs[i].clone()
            mask = masks[i].clone()

            angle = float(torch.randint(-max_angle, max_angle + 1, (1, )))
            scale = float(min_scale + torch.rand(1) * (max_scale - min_scale))

            img = TF.rotate(img, angle, TF.InterpolationMode.BILINEAR, expand=True)
            mask = TF.rotate(mask, angle, TF.InterpolationMode.NEAREST, expand=True)

            # clamp scale to prevent invalid sizes
            bg_min = min(H, W)
            h, w = img.shape[-2:]
            img_min = min(h, w)
            size = int(min(scale * img_min, bg_min - 1))

            img = TF.resize(img, [size], TF.InterpolationMode.BILINEAR, max_size=bg_min, antialias=True)
            mask = TF.resize(mask, [size], TF.InterpolationMode.NEAREST, max_size=bg_min)
            h, w = img.shape[-2:]
            max_h = max(0, H - h)
            max_w = max(0, W - w)

            # paste image at position (y0, x0) on the background
            y0 = int(torch.randint(0, max_h, (1,)))
            x0 = int(torch.randint(0, max_w, (1,)))

            idx = i * num_transforms + j
            output_img = output_imgs[idx]
            roi = torch.where(mask, img[0:3], output_img[:, y0: y0+h, x0: x0+w])
            output_img[:, y0:y0+h, x0:x0+w] = roi

            output_mask = output_masks[idx]
            output_mask[:, y0:y0+h, x0:x0+w] = mask

            # apply transforms on final image
            output_img = photometric_transforms[j](output_img)

            output_imgs[idx] = output_img
            output_masks[idx] = output_mask

    return output_imgs, output_masks

class DomainRandomizerDataset(Dataset):
    def __init__(self, config: dict):
        backgrounds_dir: str = config["backgrounds_dir"]
        images_dir: str = config["images_dir"]

        self.backgrounds: list[str] = [os.path.join(backgrounds_dir, fname) for fname in os.listdir(backgrounds_dir)]
        self.images: list[str] = [os.path.join(images_dir, fname) for fname in sorted(os.listdir(images_dir))]
        self.length: int = len(self.backgrounds) * len(self.images)

        self.segmasks: dict[str, str] = config.get("segmasks", {})
        if self.segmasks["mode"] == "file":
            masks_dir = self.segmasks["masks_dir"]
            self.masks: list[str] = [os.path.join(masks_dir, fname) for fname in sorted(os.listdir(masks_dir))]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bg_index: int = index // len(self.images)
        img_index: int = index % len(self.images)

        bg_path: str = self.backgrounds[bg_index]
        img_path: str = self.images[img_index]

        bg = decode_image(bg_path).float() / 255.0
        img = decode_image(img_path).float() / 255.0

        # NOTE: mask is a boolean tensor
        mode = self.segmasks.get("mode", "alpha")
        if mode == "file":
            mask = decode_image(self.masks[index], ImageReadMode.GRAY).float() / 255.0
            mask = mask > 0.5
        else:   # "alpha"
            if img.shape[0] == 4:       # RGBA
                mask = img[3:4, ...] > 0.5
            else:                       # RGB
                mask = torch.ones((1, img.shape[1], img.shape[2]), dtype=torch.bool)

        return bg, img, mask

def main() -> None:
    print(f"Running on {device}")

    with open("config.yaml") as fp:
        config = yaml.safe_load(fp)

    dataset = DomainRandomizerDataset(config)
    ds_config = config["dataset"]
    dataloader = DataLoader(dataset, batch_size=ds_config["batch_size"], num_workers=ds_config["num_workers"])

    output_dir: str = config.get("output_dir", "output")
    images_dir: str = config["images_dir"]
    output_images_dir: str = os.path.join(output_dir, images_dir)
    os.makedirs(output_images_dir, exist_ok=True)

    save_masks = "segmasks" in config
    if save_masks:
        masks_dir = config["segmasks"]["masks_dir"]
        output_masks_dir: str = os.path.join(output_dir, masks_dir)
        os.makedirs(output_masks_dir, exist_ok=True)

    tf_config = config["transform"]
    for batch_idx, (bgs, imgs, masks) in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

        bgs, imgs, masks = bgs.to(device), imgs.to(device), masks.to(device)
        imgs, masks = batch_transform(tf_config, bgs, imgs, masks)

        for i in range(len(imgs)):
            fname = f"image_{batch_idx}_{i}.jpg"
            write_jpeg((imgs[i] * 255).byte().cpu(), os.path.join(output_images_dir, fname))
            if save_masks:
                write_jpeg((masks[i] * 255).byte().cpu(), os.path.join(output_masks_dir, fname))


if __name__ == "__main__":
    main()

