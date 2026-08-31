"""Flow-matching helpers used by the stand-alone trainer."""

from __future__ import annotations

import torch

from mantis_alpha.flow_matching import euler_integrate, sample_noise, sample_time_beta


def test_sample_time_in_unit_interval() -> None:
    t = sample_time_beta(256, "cpu", alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
    assert t.shape == (256,)
    assert torch.all(t >= 0.001) and torch.all(t <= 1.0)


def test_euler_integrate_from_noise_to_zero() -> None:
    # x(t) = t * noise, v = noise. Euler from t=1 to t=0 recovers 0.
    noise = sample_noise((4, 8, 7), "cpu")

    def denoise_fn(x_t, time_tensor):
        return noise

    out = euler_integrate(denoise_fn, noise, num_steps=10)
    assert out.shape == noise.shape
    assert torch.allclose(out, torch.zeros_like(noise), atol=1e-5)
