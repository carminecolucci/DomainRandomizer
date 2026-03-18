import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image, write_jpeg, ImageReadMode
from torchvision.transforms.v2 import functional as TF

import yaml

# debug only
import numpy as np
import cv2


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

def yolo_line(cls: int, center: torch.Tensor, dim: torch.Tensor, kpts: torch.Tensor) -> str:
    """
    Format a single YOLO keypoint label line.
    Args:
        cls:    Class index.
        center: Bounding box center, normalised to [0, 1].
        dim:    Bounding box dimensions, normalised to [0, 1].
        kpts:   (K, 3) tensor of (kx, ky, kv) per keypoint, normalised.

    Returns:
        Space-separated string ready to write to a .txt label file.
    """
    # Header: class and bounding box
    vals = [str(cls), f"{center[0]:.6f}", f"{center[1]:.6f}", f"{dim[0]:.6f}", f"{dim[1]:.6f}"]

    # Keypoints: x, y, visibility
    for (x, y, v) in kpts:
        vals += [f"{x:.6f}", f"{y:.6f}", str(int(v))]

    return " ".join(vals)

def project_points(pts: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Apply a homographic (projective) transformation to a set of 2D points.

    Args:
        pts: (N, 2) tensor of 2D input points.
        M:   (3, 3) homography matrix.

    Returns:
        (N, 2) tensor of transformed 2D points.
    """
    pts_h = torch.hstack([pts, torch.ones(pts.shape[0], 1)])
    warped = (M @ pts_h.T).T
    return warped[:, :2] / warped[:, 2:3]

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

    bg_min_size = int(torch.min(bg_size).item())
    img_min_size = torch.min(img_size).item()
    size = int(min(scale * img_min_size, bg_min_size - 1))

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

        self.has_labels: bool = "labels_dir" in config
        if self.has_labels:
            labels_dir = config["labels_dir"]
            self.labels: list[str] = [os.path.join(labels_dir, fname) for fname in sorted(os.listdir(labels_dir))]
            self.output_labels_dir = os.path.join(output_dir, labels_dir)
            os.makedirs(self.output_labels_dir, exist_ok=True)

            self.debug: bool = "labels_debug_dir" in config
            if self.debug:
                self.output_debug_dir = os.path.join(output_dir, config["labels_debug_dir"])
                os.makedirs(self.output_debug_dir, exist_ok=True)

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

        bg = decode_image(bg_path).float() / 255.0
        img = decode_image(img_path).float() / 255.0

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

        if self.has_labels:
            label = self.load_label(img_index)
            cls, center, dim, kpts = label
            _, h, w = img.shape
            _, H, W = bg.shape

            # 1. Obtain the transformation matrix
            # This must match the rotation/scale/translation logic in your transform()
            M = self._get_transform_matrix(img.shape[-2:], bg.shape[-2:], angle, scale, paste_pos)

            # 2. Warp the Bounding Box Corners
            # Original center (xc, yc) and dimensions (bw, bh) to 4 pixel corners
            bx_c, by_c = center[0] * w, center[1] * h
            bw_h, bh_h = (dim[0] * w) / 2, (dim[1] * h) / 2

            src_box = torch.tensor([
                [bx_c - bw_h, by_c - bh_h],                     # Top-left
                [bx_c + bw_h, by_c - bh_h],                     # Top-right
                [bx_c + bw_h, by_c + bh_h],                     # Bottom-right
                [bx_c - bw_h, by_c + bh_h]                      # Bottom-left
            ])

            # Keypoints
            src_kpts = kpts[:, :2] * torch.tensor([w, h])

            # 3. Warp using your manual function
            warped_box = project_points(src_box, M)
            warped_kpts = project_points(src_kpts, M)

            # 4. Calculate AABB
            mins, _ = warped_box.min(dim=0)                     # x_min, y_min
            maxs, _ = warped_box.max(dim=0)                     # x_max, y_max

            wh = torch.tensor([W, H], dtype=torch.float32)
            new_center = (mins + maxs) / (2 * wh)               # (xc, yc)
            new_dim = (maxs - mins) / wh                        # (bw, bh)

            # Normalize keypoints: (norm_x, norm_y, v)
            vis        = kpts[:, 2:]                            # (K, 1) visibility flags
            norm_xy    = warped_kpts / torch.tensor([W, H])     # (K, 2) normalized coords
            norm_kpts  = torch.cat([norm_xy, vis], dim=1)       # (K, 3) normalized keypoints

            # 5. Save the Label
            label_file = os.path.join(self.output_labels_dir, f"{os.path.basename(img_path).split(".")[0]}_{index}.txt")
            with open(label_file, "w") as f:
                f.write(yolo_line(cls, new_center, new_dim, norm_kpts) + "\n")


            if self.debug:
                # Convert the final composited image (new_img) back to OpenCV format (BGR, 0-255)
                # new_img is [C, H, W] RGB float32 tensor
                dbg = (new_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                dbg = cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR)

                # Draw Bounding Box (AABB)
                x_min, y_min = mins
                x_max, y_max = maxs
                cv2.rectangle(dbg, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 200, 0), 2)

                # Draw Keypoints
                for j, (x, y) in enumerate(warped_kpts):
                    # Draw circle and index
                    cv2.circle(dbg, (int(x), int(y)), 6, (0, 0, 255), -1)
                    cv2.putText(dbg, str(j), (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw Polygon connecting keypoints (if applicable)
                pts = warped_kpts.cpu().numpy().astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(dbg, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Save the debug image
                debug_path = os.path.join(self.output_debug_dir, f"dbg_{index:05d}.jpg")
                cv2.imwrite(debug_path, dbg)

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

    def load_label(self, index: int) -> list:
        """
        Load and parse a YOLO-format label file for a single sample.

        The label file is expected to contain a single line with space-separated
        values in the following format:
            <cls> <xc> <yc> <bw> <bh> [<kx> <ky> <kv>] ...

        Where:
            - cls:       Integer class index.
            - xc, yc:    Bounding box center, normalized to [0, 1].
            - bw, bh:    Bounding box dimensions, normalized to [0, 1].
            - kx, ky:    Keypoint coordinates, normalized to [0, 1].
            - kv:        Keypoint visibility flag (0=hidden, 1=occluded, 2=visible).

        Args:
            index: Dataset index of the sample to load.

        Returns:
            [cls, center, size, keypoints] where:
                - cls:       (int) class index.
                - center:    (2,) float32 tensor of (xc, yc).
                - size:      (2,) float32 tensor of (bw, bh).
                - keypoints: (K, 3) float32 tensor of (kx, ky, kv) per keypoint.
        """
        label_path = self.labels[index]
        with open(label_path, "r") as f:
            line = f.readline().strip()

        vals = torch.tensor(list(map(float, line.split())))

        cls    = int(vals[0])
        center = vals[1:3]                  # (xc, yc)
        size   = vals[3:5]                  # (bw, bh)
        kpts   = vals[5:].reshape(-1, 3)    # (K, 3) — (kx, ky, kv) per keypoint

        return [cls, center, size, kpts]

    def _get_transform_matrix(self, img_shape: tuple[int, int], bg_shape: tuple[int, int], angle: float, scale: float, paste_pos: torch.Tensor) -> torch.Tensor:
        """
        Build a 3x3 homography matrix that replicates the pipeline:
            rotate (expand=True) -> resize -> paste translation.

        Args:
            img_shape:  (H, W) of the source image.
            bg_shape:   (H, W) of the background canvas.
            angle:      Rotation angle in degrees (counter-clockwise, matching torchvision).
            scale:      Scaling factor relative to the smaller side of the rotated image.
            paste_pos:  (y, x) top-left paste position on the background canvas.

        Returns:
            (3, 3) homography matrix M.
        """
        h, w = img_shape

        # 1. Rotate
        cx, cy = w / 2.0, h / 2.0

        angle_rad = torch.deg2rad(torch.tensor(-angle))
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        corners = torch.tensor([[0, 0], [w, 0], [w, h], [0, h]])
        M_rot_only = torch.tensor([
            [cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a],
            [sin_a,  cos_a, cy * (1 - cos_a) - cx * sin_a],
            [    0,      0,                             1]
        ])

        # The 4 corners of the original image after rotation around center
        rotated_corners = project_points(corners, M_rot_only)

        # expand=True shifts origin so all coords are positive
        min_x = rotated_corners[:, 0].min()
        min_y = rotated_corners[:, 1].min()
        new_gw = rotated_corners[:, 0].max() - min_x  # expanded canvas width
        new_gh = rotated_corners[:, 1].max() - min_y  # expanded canvas height

        # Shift matrix to account for expand
        M_rot_only[0, 2] -= min_x
        M_rot_only[1, 2] -= min_y

        # 2. Resize
        # TF.resize operates on the expanded image (new_gh, new_gw)
        img_min_size = min(new_gh, new_gw)
        bg_min_size = min(bg_shape[0], bg_shape[1])
        actual_size = min(int(scale * img_min_size), bg_min_size - 1)
        resize_ratio = actual_size / img_min_size

        M_scale = torch.tensor([
            [resize_ratio, 0, 0],
            [0, resize_ratio, 0],
            [0, 0,            1]
        ])

        # 3. Paste translation
        y, x = paste_pos
        M_translate = torch.tensor([
            [1.0,   0,   x],
            [  0, 1.0,   y],
            [  0,   0, 1.0]
        ])

        M = M_translate @ M_scale @ M_rot_only
        return M

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

