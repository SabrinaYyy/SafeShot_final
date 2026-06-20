from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from imageshield.protection import (
    ProtectionCancelled,
    ProtectionService,
    ProtectionSettings,
    derive_image_seed,
    get_emb,
    make_preprocess,
    pgd_protect,
    tensor_to_pil,
)


class _Distribution:
    def __init__(self, mean):
        self.mean = mean

    def sample(self):
        return self.mean


class _Encoded:
    def __init__(self, mean):
        self.latent_dist = _Distribution(mean)


class MockVAE:
    class _Config:
        scaling_factor = 1.0

    config = _Config()

    def __init__(self):
        self.encode_calls = 0

    def encode(self, image):
        self.encode_calls += 1
        feature = image.mean(dim=1, keepdim=True).expand(-1, 4, -1, -1)
        return _Encoded(F.adaptive_avg_pool2d(feature, (4, 4)))


class MockScheduler:
    class _Config:
        num_train_timesteps = 1000

    config = _Config()

    @staticmethod
    def add_noise(latents, noise, timesteps):
        return 0.7 * latents + 0.3 * noise


@pytest.fixture
def image():
    generator = np.random.default_rng(7)
    pixels = generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def test_settings_reject_unknown_mode():
    with pytest.raises(ValueError, match="Mode"):
        ProtectionSettings(mode="unknown").validate()


def test_settings_allow_original_resolution():
    ProtectionSettings(resolution=None, mode="sd").validate()


def test_image_seed_is_deterministic(image):
    assert derive_image_seed(33, image) == derive_image_seed(33, image.copy())


def test_embedding_encodes_image_once(image):
    vae = MockVAE()
    original = make_preprocess(64)(image).unsqueeze(0)

    embedding = get_emb(
        original,
        vae,
        MockScheduler(),
        torch.device("cpu"),
        torch.float32,
    )

    assert vae.encode_calls == 1
    assert embedding.shape == (1, 8, 4, 4)


def test_pgd_is_deterministic_and_bounded(image):
    settings = ProtectionSettings(
        resolution=64,
        eps=0.05,
        alpha=1 / 255,
        steps=3,
    )
    original = make_preprocess(settings.resolution)(image).unsqueeze(0)
    seed = derive_image_seed(settings.seed, image)
    arguments = (
        original,
        MockVAE(),
        MockScheduler(),
        torch.device("cpu"),
        torch.float32,
        settings,
        seed,
    )

    first = pgd_protect(*arguments)
    second = pgd_protect(*arguments)

    assert torch.equal(first, second)
    assert (first - original).abs().max().item() <= settings.eps + 1e-6
    assert tensor_to_pil(first).size == (64, 64)


def test_pgd_can_be_cancelled(image):
    settings = ProtectionSettings(resolution=64, steps=3)
    original = make_preprocess(settings.resolution)(image).unsqueeze(0)

    with pytest.raises(ProtectionCancelled, match="stopped"):
        pgd_protect(
            original,
            MockVAE(),
            MockScheduler(),
            torch.device("cpu"),
            torch.float32,
            settings,
            derive_image_seed(settings.seed, image),
            cancelled=lambda: True,
        )


def test_service_uses_injected_models_without_loading(image, tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "vae").mkdir(parents=True)
    (model_dir / "scheduler").mkdir()
    (model_dir / "vae" / "config.json").write_text(json.dumps({}))
    (model_dir / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(
        b"test"
    )
    (model_dir / "scheduler" / "scheduler_config.json").write_text(json.dumps({}))

    service = ProtectionService(
        model_dir=model_dir,
        settings=ProtectionSettings(resolution=64, steps=2),
    )
    service.device = torch.device("cpu")
    service._vae = MockVAE()
    service._scheduler = MockScheduler()

    protected = service.protect(image)

    assert protected.mode == "RGB"
    assert protected.size == (64, 64)


def test_offline_bundle_validation_reports_missing_files(tmp_path):
    service = ProtectionService(model_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="offline model bundle"):
        service.validate_model_files()
