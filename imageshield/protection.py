"""Offline, single-image protection service used by the desktop UI."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as TF

from .resources import resource_path

ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]

EPS_IP2P: float = 8 / 255    # default for IP2P mode
EPS_SD: float   = 16 / 255   # PhotoGuard paper value — default for SD mode

# BlurGuard (SD mode) fixed hyperparameters
_SIGMA_WARMUP_FRAC = 1 / 3   # warmup = round(steps * frac), min 1
_SIGMA_INIT       = 0.0
_SIGMA_LR         = 0.001
_SIGMA_WEIGHT     = 10_000.0
_SLIC_N_SEGMENTS  = 4
_SLIC_COMPACTNESS = 10


class ProtectionCancelled(RuntimeError):
    """Raised when the user stops an active protection job."""


@dataclass(frozen=True)
class ProtectionSettings:
    resolution: int | None = None   # None = keep original image dimensions
    eps: float = EPS_IP2P
    alpha: float = 1 / 255
    steps: int = 100
    seed: int = 33
    beta: float = 0.2
    eot_angle: float = 5.0
    mode: str = "ip2p"              # "ip2p" (EditShield) or "sd" (BlurGuard)

    def validate(self) -> None:
        if self.resolution is not None and self.resolution < 64:
            raise ValueError("Resolution must be at least 64 pixels.")
        if self.steps < 1:
            raise ValueError("Protection steps must be at least 1.")
        if not 0 < self.eps <= 1:
            raise ValueError("Epsilon must be between 0 and 1.")
        if self.alpha <= 0:
            raise ValueError("Alpha must be positive.")
        if self.beta < 0:
            raise ValueError("Beta must be non-negative.")
        if self.mode not in ("ip2p", "sd"):
            raise ValueError("Mode must be 'ip2p' or 'sd'.")


def select_device() -> tuple[torch.device, torch.dtype]:
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float32
    return torch.device("cpu"), torch.float32


def device_summary(device: torch.device) -> str:
    if device.type == "mps":
        return "GPU detected: Apple Silicon (Metal Performance Shaders) — acceleration enabled."
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return f"GPU detected: {name} ({memory_gb:.1f} GB VRAM) — acceleration enabled."
    return (
        "Warning: No GPU detected — running on CPU. "
        "Protection may take 10–60+ minutes per image depending on resolution. "
        "For faster results, select a lower resolution (128 or 256)."
    )


def make_preprocess(resolution: int):
    return transforms.Compose(
        [
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )


def _preprocess_original_size(image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize image to nearest multiple of 8 (VAE requirement) and convert to tensor.

    Returns the tensor [1, 3, H8, W8] and the effective (W8, H8) size.
    """
    w, h = image.size
    w8 = max(8, (w // 8) * 8)
    h8 = max(8, (h // 8) * 8)
    if (w8, h8) != (w, h):
        image = image.resize((w8, h8), Image.BILINEAR)
    return transforms.ToTensor()(image).unsqueeze(0), (w8, h8)


def get_emb(
    image: torch.Tensor,
    vae,
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Stochastic VAE embedding used by IP2P mode — matches notebook get_emb() exactly."""
    image = image.to(device, dtype=dtype)
    dist = vae.encode(image).latent_dist
    latents = dist.sample() * vae.config.scaling_factor
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (bsz,), device=device
    ).long()
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    original_image_embeds = dist.sample()
    return torch.cat([noisy_latents, original_image_embeds], dim=1)


def rotate_batch(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate each image in the batch — differentiable.

    grid_sampler_2d_backward is not yet implemented for MPS, so we perform the
    rotation on CPU and transfer back.  Device transfers are differentiable,
    so gradients flow through the round-trip correctly.
    """
    target_device = x.device
    if target_device.type == "mps":
        x = x.cpu()
    rotated = torch.stack(
        [
            TF.rotate(x[i], angle, interpolation=transforms.InterpolationMode.BILINEAR)
            for i in range(x.shape[0])
        ]
    )
    return rotated.to(target_device)


def derive_image_seed(base_seed: int, image: Image.Image) -> int:
    digest = hashlib.sha256()
    digest.update(str(base_seed).encode("ascii"))
    digest.update(image.tobytes())
    return int.from_bytes(digest.digest()[:4], "big")


# ---------------------------------------------------------------------------
# IP2P mode — EditShield (exact current implementation, unchanged)
# ---------------------------------------------------------------------------

def pgd_protect(
    original: torch.Tensor,
    vae,
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
    settings: ProtectionSettings,
    image_seed: int,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> torch.Tensor:
    original = original.to(device, dtype=dtype)

    torch.manual_seed(image_seed)

    with torch.no_grad():
        tgt_emb = get_emb(original, vae, scheduler, device, dtype).detach()

    delta = (torch.rand(original.shape) * 2.0 - 1.0) * settings.eps
    delta = delta.to(device, dtype=dtype)
    delta = torch.clamp(delta, -settings.eps, settings.eps)
    clipped = torch.clamp(original + delta, 0.0, 1.0)
    delta = (clipped - original).detach()

    for step in range(settings.steps):
        if cancelled and cancelled():
            raise ProtectionCancelled("Protection stopped by the user.")

        perturbed = torch.clamp(original + delta, 0.0, 1.0).detach().requires_grad_(True)

        rotated = rotate_batch(perturbed, settings.eot_angle)
        img_emb = get_emb(rotated, vae, scheduler, device, dtype)

        real_mse = F.mse_loss(img_emb.float(), tgt_emb.float())
        loss_percep = settings.beta * F.mse_loss(perturbed, original)
        total_loss = -real_mse + loss_percep
        total_loss.backward()

        with torch.no_grad():
            delta = delta - settings.alpha * perturbed.grad.sign()
            delta = torch.clamp(delta, -settings.eps, settings.eps)
            clipped = torch.clamp(original + delta, 0.0, 1.0)
            delta = (clipped - original).detach()

        if progress:
            progress(
                (step + 1) / settings.steps,
                f"Protection step {step + 1}/{settings.steps}",
            )

    if cancelled and cancelled():
        raise ProtectionCancelled("Protection stopped by the user.")

    return torch.clamp(original + delta, 0.0, 1.0).cpu()


# ---------------------------------------------------------------------------
# SD mode — BlurGuard helpers (adapted from blurguard.ipynb)
# ---------------------------------------------------------------------------

def _gaussian_blur_bg(x: torch.Tensor, sigma: torch.Tensor | float, width: int = 33) -> torch.Tensor:
    """Differentiable depthwise Gaussian blur. Accepts [B,C,H,W] or [C,H,W]."""
    is_3d = (x.ndim == 3)
    if is_3d:
        x = x.unsqueeze(0)
    width = width + (width + 1) % 2  # ensure odd
    d = torch.arange(-(width // 2), width // 2 + 1, dtype=torch.float, device=x.device)
    k = torch.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    C = x.size(1)
    k_expanded = k[None, None].expand(C, -1, -1, -1)
    x = torch.nn.ReflectionPad2d(width // 2)(x)
    y = F.conv2d(x, k_expanded, groups=C)
    return y[0] if is_3d else y


def _generate_slic_masks(
    image_pil: Image.Image,
    n_segments: int = 4,
    compactness: int = 10,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Segment image into SLIC superpixel regions. Returns {mask1: Tensor[1,1,H,W], ...}."""
    from skimage.segmentation import slic as skimage_slic  # lazy import

    img_np = np.array(image_pil.convert("RGB"))
    labels = skimage_slic(img_np, n_segments=n_segments, compactness=compactness, start_label=0)
    actual_n = int(labels.max()) + 1
    masks_dict: dict[str, torch.Tensor] = {}
    for i in range(actual_n):
        mask = torch.from_numpy((labels == i).astype(np.float32))
        masks_dict[f"mask{i + 1}"] = mask.unsqueeze(0).unsqueeze(0).to(device)
    return masks_dict


def _filter_delta(
    log_sigmas: torch.Tensor,
    delta: torch.Tensor,
    masks_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """BlurGuard output: sum_r  M_r * G(delta, sigma_r)."""
    parts = []
    for i, (_key, mask) in enumerate(masks_dict.items()):
        mask = mask.to(delta.device)
        parts.append(_gaussian_blur_bg(delta, log_sigmas[i].exp()) * mask)
    return sum(parts)  # type: ignore[return-value]


def _fft_power_spectrum(img: torch.Tensor) -> torch.Tensor:
    """2-D FFT power spectrum of [1,C,H,W] or [C,H,W], summed over channels. Returns [H,W]."""
    if img.ndim == 4:
        img = img[0]
    return (torch.abs(torch.fft.fft2(img)) ** 2).sum(dim=0)


def _radial_power_histogram(power_2d: torch.Tensor) -> torch.Tensor:
    """Collapse [H,W] power spectrum to 1-D radial histogram."""
    H, W = power_2d.shape
    device = power_2d.device
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).round().long()
    r_max = r.max().item() + 1
    bins = torch.stack([(r == i).float() for i in range(r_max)])
    return (bins * power_2d).sum(dim=(-1, -2))


def pgd_protect_sd(
    original: torch.Tensor,
    original_pil: Image.Image,
    vae,
    device: torch.device,
    dtype: torch.dtype,
    settings: ProtectionSettings,
    image_seed: int,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> torch.Tensor:
    """SD (BlurGuard) protection — PhotoGuard VAE attack with per-region adaptive blur."""
    original = original.to(device, dtype=dtype)
    torch.manual_seed(image_seed)

    eps = settings.eps
    max_steps = settings.steps
    sigma_warmup = max(1, round(max_steps * _SIGMA_WARMUP_FRAC))

    # VAE encodes expect [-1, 1] input
    with torch.no_grad():
        ref_latent = (
            vae.encode((original * 2 - 1).to(dtype=dtype))
            .latent_dist.mean.detach()
            .float()
            * vae.config.scaling_factor
        )

    # Random Gaussian target in VAE latent space (unpredictable attack direction)
    tgt_latent = torch.randn_like(ref_latent)

    # SLIC segmentation of the source image at the current resolution
    masks_dict = _generate_slic_masks(
        original_pil, n_segments=_SLIC_N_SEGMENTS, compactness=_SLIC_COMPACTNESS, device=device
    )
    n_masks = len(masks_dict)

    log_sigmas = torch.full((n_masks,), _SIGMA_INIT, device=device, requires_grad=True)
    sigma_opt = torch.optim.Adam([log_sigmas], lr=_SIGMA_LR)

    delta = torch.empty_like(original).uniform_(-eps, eps)
    delta = torch.clamp(delta, -eps, eps)
    delta = (torch.clamp(original + delta, 0.0, 1.0) - original).detach()

    stage2_total = max(max_steps - sigma_warmup, 1)

    for step in range(max_steps):
        if cancelled and cancelled():
            raise ProtectionCancelled("Protection stopped by the user.")

        sigma_opt.zero_grad()

        # Frequency-spectrum constraint (always computed; drives sigma warmup in Stage 1)
        rand_n = torch.randn_like(original).clamp(-eps, eps)
        rand_lf = _filter_delta(log_sigmas, rand_n, masks_dict)
        rand_lf = torch.clamp(rand_lf, -eps, eps)
        fps_src = _radial_power_histogram(_fft_power_spectrum(original))
        fps_rob = _radial_power_histogram(
            _fft_power_spectrum(torch.clamp(original + rand_lf, 0.0, 1.0))
        )
        sigma_loss = torch.mean(
            torch.abs(torch.log10(fps_src + 1e-8) - torch.log10(fps_rob + 1e-8))
        )

        if step < sigma_warmup:
            # Stage 1 — only update log_sigmas
            (_SIGMA_WEIGHT * sigma_loss).backward()
            sigma_opt.step()
        else:
            # Stage 2 — PGD on delta, gradient routed through filter_delta → VAE
            delta_var = delta.detach().requires_grad_(True)
            pert_lf = _filter_delta(log_sigmas.detach(), delta_var, masks_dict)
            pert_lf_c = torch.clamp(pert_lf, -eps, eps)
            perturbed = torch.clamp(original + pert_lf_c, 0.0, 1.0)

            adv_latent = (
                vae.encode((perturbed * 2 - 1).to(dtype=dtype))
                .latent_dist.mean.float()
                * vae.config.scaling_factor
            )
            loss_main = F.mse_loss(adv_latent, tgt_latent)
            loss_main.backward()

            with torch.no_grad():
                stage2_step = step - sigma_warmup
                actual_step = 1.0 - (1.0 - 0.01) / stage2_total * stage2_step
                grad = delta_var.grad
                grad_norm = (
                    torch.norm(grad.reshape(grad.size(0), -1), p=2, dim=1) + 1e-10
                )
                grad_normed = grad / grad_norm.view(grad.size(0), 1, 1, 1) * 2
                delta = (delta_var - actual_step * grad_normed).detach()
                delta = torch.clamp(delta, -eps, eps)
                delta = (torch.clamp(original + delta, 0.0, 1.0) - original).detach()

        if progress:
            progress(
                (step + 1) / max_steps,
                f"Protection step {step + 1}/{max_steps}",
            )

    if cancelled and cancelled():
        raise ProtectionCancelled("Protection stopped by the user.")

    # Apply learned per-region blurring to final delta
    with torch.no_grad():
        pert_final = _filter_delta(log_sigmas, delta, masks_dict)
        pert_final = torch.clamp(pert_final, -eps, eps)
        protected = torch.clamp(original + pert_final, 0.0, 1.0)

    return protected.cpu()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


class ProtectionService:
    """Lazily load bundled weights once and serialize protection jobs."""

    def __init__(
        self,
        model_dir: Path | None = None,
        settings: ProtectionSettings | None = None,
    ) -> None:
        self.model_dir = model_dir or resource_path("models", "instruct-pix2pix")
        self.settings = settings or ProtectionSettings()
        self.settings.validate()
        self.device, self.dtype = select_device()
        self._vae = None
        self._scheduler = None
        self._load_lock = threading.Lock()
        self._protection_lock = threading.Lock()
        self._cancel_event = threading.Event()

    @property
    def is_loaded(self) -> bool:
        return self._vae is not None and self._scheduler is not None

    def validate_model_files(self) -> None:
        required = (
            self.model_dir / "vae" / "config.json",
            self.model_dir / "vae" / "diffusion_pytorch_model.safetensors",
            self.model_dir / "scheduler" / "scheduler_config.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "The offline model bundle is incomplete. Missing: " + ", ".join(missing)
            )

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            self.validate_model_files()
            from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            vae = AutoencoderKL.from_pretrained(
                self.model_dir / "vae",
                local_files_only=True,
            ).to(self.device, dtype=self.dtype)
            vae.requires_grad_(False)
            vae.eval()
            scheduler = DDPMScheduler.from_pretrained(
                self.model_dir / "scheduler",
                local_files_only=True,
            )
            self._vae = vae
            self._scheduler = scheduler

    def protect(
        self,
        image: Image.Image,
        settings: ProtectionSettings | None = None,
        progress: ProgressCallback | None = None,
    ) -> Image.Image:
        if image is None:
            raise ValueError("Please upload an image before starting protection.")
        self.load()

        run_settings = settings or self.settings
        run_settings.validate()

        source = ImageOps.exif_transpose(image).convert("RGB")

        if run_settings.resolution is None:
            # Preserve original dimensions, rounded down to nearest multiple of 8
            original_tensor, _ = _preprocess_original_size(source)
            # Reconstruct PIL at the rounded size for SLIC (SD mode)
            source_resized = tensor_to_pil(original_tensor)
        else:
            original_tensor = make_preprocess(run_settings.resolution)(source).unsqueeze(0)
            source_resized = tensor_to_pil(original_tensor)

        image_seed = derive_image_seed(run_settings.seed, source)

        with self._protection_lock:
            self._cancel_event.clear()

            if run_settings.mode == "ip2p":
                protected = pgd_protect(
                    original_tensor,
                    self._vae,
                    self._scheduler,
                    self.device,
                    self.dtype,
                    run_settings,
                    image_seed,
                    progress,
                    self._cancel_event.is_set,
                )
            else:
                protected = pgd_protect_sd(
                    original_tensor,
                    source_resized,
                    self._vae,
                    self.device,
                    self.dtype,
                    run_settings,
                    image_seed,
                    progress,
                    self._cancel_event.is_set,
                )

        if self.device.type == "mps":
            torch.mps.empty_cache()

        return tensor_to_pil(protected)

    def cancel(self) -> None:
        self._cancel_event.set()
