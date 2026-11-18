import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class ForegroundNoiseDataset(Dataset):
    """
    Dataset wrapper implementing:
    1. Aspect-ratio-preserving foreground scale normalization (enlarging object bounding box to span canvas).
    2. Stochastic background replacement (filling background outside mask with dynamic Gaussian or uniform noise,
       or constant neutral values).
    
    Supports the experimental ablation matrix:
    - M2: enlarge=False, noise_bg=False (Original scale, constant neutral background)
    - M3: enlarge=False, noise_bg=True  (Original scale, dynamic stochastic noise)
    - M4: enlarge=True,  noise_bg=False (Enlarged scale, constant neutral background)
    - M5: enlarge=True,  noise_bg=True  (Enlarged scale, dynamic stochastic noise - Proposed Full Model)
    """
    def __init__(self, src_ds, target_size=32, margin=2, enlarge=True, noise_bg=True, noise_type="gaussian", mask_ratio=0.30):
        self.ds = src_ds
        self.target_size = target_size
        self.margin = margin
        self.enlarge = enlarge
        self.noise_bg = noise_bg
        self.noise_type = noise_type
        self.mask_ratio = mask_ratio
        
        self.resize_crop = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size)
        ])
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=target_size, padding=int(target_size * 0.125))
        ])
        self.topil = transforms.ToPILImage()
        
        self.seg_momentum = 0.1
        self.seg_bank = None
        self.init_segbank(8, 8)

    def __len__(self):
        return len(self.ds)

    def init_segbank(self, h, w):
        self.seg_bank = torch.ones([len(self.ds), h, w], dtype=torch.float32) * 0.5
        self.seg_bank.share_memory_()

    def soft_seg(self, idx, seg_mask):
        """Update foreground CAM estimation bank with exponential moving average."""
        if self.seg_bank is None:
            _, h, w = seg_mask.shape
            self.init_segbank(h, w)
        self.seg_bank[idx] = seg_mask * self.seg_momentum + self.seg_bank[idx] * (1.0 - self.seg_momentum)

    def _apply_transformation(self, img_tensor, seg_map):
        C, H, W = img_tensor.shape
        L = self.target_size
        
        # 1. Interpolate CAM activation map to image spatial resolution
        seg_high = F.interpolate(
            seg_map.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )[0, 0]

        # 2. Extract foreground binary mask using top-K percent activation
        flat = seg_high.view(-1)
        k = max(1, int(self.mask_ratio * flat.numel()))
        thresh = torch.topk(flat, k)[0][-1]
        mask = (seg_high >= thresh).float() # (H, W)

        # 3. Compute bounding box coordinates B = (xmin, ymin, xmax, ymax)
        active_idx = torch.nonzero(mask)
        if active_idx.shape[0] < 4:
            # Degenerate mask fallback: use full image bounds
            ymin, ymax, xmin, xmax = 0, H - 1, 0, W - 1
            w, h = W, H
        else:
            ymin = active_idx[:, 0].min().item()
            ymax = active_idx[:, 0].max().item()
            xmin = active_idx[:, 1].min().item()
            xmax = active_idx[:, 1].max().item()
            h = max(2, ymax - ymin + 1)
            w = max(2, xmax - xmin + 1)

        fg_crop = img_tensor[:, ymin:ymax+1, xmin:xmax+1]     # (C, h, w)
        mask_crop = mask[ymin:ymax+1, xmin:xmax+1].unsqueeze(0) # (1, h, w)

        # 4. Aspect-ratio-preserving scale normalization
        if self.enlarge:
            target_dim = L - self.margin
            s = target_dim / max(w, h)
            new_w = max(1, min(L, int(round(s * w))))
            new_h = max(1, min(L, int(round(s * h))))
            
            fg_resized = F.interpolate(
                fg_crop.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )[0]
            mask_resized = F.interpolate(
                mask_crop.unsqueeze(0),
                size=(new_h, new_w),
                mode='nearest'
            )[0]
        else:
            new_w, new_h = w, h
            fg_resized = fg_crop
            mask_resized = mask_crop

        # 5. Background canvas generation
        if self.noise_bg:
            if self.noise_type == "gaussian":
                # N_t ~ N(0, 1) in normalized feature space
                bg = torch.randn(C, L, L, dtype=torch.float32)
            elif self.noise_type in ["balanced_uniform", "balanced_discrete_uniform"]:
                # Balanced discrete uniform noise with complementary (antithetic) pairs
                # Generates complementary intensity pairs (+v, -v) and randomly distributes them
                total = C * L * L
                half = total // 2
                num_levels = 16  # discrete intensity quantization steps
                steps = torch.randint(1, num_levels + 1, (half,))
                scale = 2.0 / num_levels  # scaled to match normalized space [-2.0, 2.0]
                vals = steps.float() * scale
                antithetic = -vals
                all_vals = torch.cat([vals, antithetic], dim=0)
                if all_vals.numel() < total:
                    all_vals = torch.cat([all_vals, torch.zeros(1)], dim=0)
                # Randomly distribute across the background canvas
                perm = torch.randperm(total)
                bg = all_vals[perm].view(C, L, L)
            else:
                # N_t ~ U(-2, 2) – fallback for continuous uniform noise
                bg = torch.rand(C, L, L, dtype=torch.float32) * 4.0 - 2.0
        else:
            # Constant neutral canvas (mean value 0 in normalized space)
            bg = torch.zeros(C, L, L, dtype=torch.float32)

        # 6. Center placement
        y_start = (L - new_h) // 2
        x_start = (L - new_w) // 2
        
        full_fg = torch.zeros(C, L, L, dtype=torch.float32)
        full_mask = torch.zeros(1, L, L, dtype=torch.float32)
        
        full_fg[:, y_start:y_start+new_h, x_start:x_start+new_w] = fg_resized
        full_mask[:, y_start:y_start+new_h, x_start:x_start+new_w] = mask_resized

        # 7. Spatial compositing: I'_t = M' ⊙ F' + (1 - M') ⊙ N_t
        out = full_mask * full_fg + (1.0 - full_mask) * bg
        return out

    def __getitem__(self, idx):
        item = self.ds[idx]
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                img, lb, _ = item
            elif len(item) == 2:
                img, lb = item
            else:
                raise ValueError(f"Unexpected item length {len(item)}")
        else:
            raise ValueError(f"Unexpected item type {type(item)}")

        # Convert to standard normalized tensor
        if not isinstance(img, torch.Tensor):
            img = self.resize_crop(img)
            img = self.totensor(img)
        raw_img = img.detach().clone()

        # Apply transformation
        seg_map = self.seg_bank[idx]
        proc_img = self._apply_transformation(raw_img, seg_map)

        # Apply data augmentation to regularize spatial robustness
        # Convert to [0, 1] range for PIL augmentation, then back to normalized tensor
        f_img = self.topil(torch.clamp(proc_img * 0.25 + 0.5, 0.0, 1.0))
        final_img = self.totensor(self.aug(f_img))

        return raw_img, final_img, lb, idx
