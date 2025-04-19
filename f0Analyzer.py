import torch
import torch.nn as nn

import pyworld as pw

import numpy as np


def norm_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = np.log2(f0 + uv)  # avoid arithmetic error
    f0[uv] = -np.inf
    return f0


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv)
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def resample_align_curve(points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate((curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp


class F0Analyzer(nn.Module):
    def __init__(
            self, 
            sampling_rate: int, 
            f0_extractor : str, 
            hop_size     : int,
            f0_min       : float, 
            f0_max       : float,
    ):
        """  
        Args:
            sampling_rate (int): Sampling rate of the audio signal.
            f0_extractor (str): Type of F0 extractor ('parselmouth' or 'harvest').
            f0_min (float): Minimum F0 in Hz.
            f0_max (float): Maximum F0 in Hz.
            hop_size (int): Hop size in samples.
        """
        super(F0Analyzer, self).__init__()
        self.sampling_rate = sampling_rate
        self.f0_extractor  = f0_extractor
        self.hop_size      = hop_size
        self.f0_min        = f0_min
        self.f0_max        = f0_max

    def forward(
            self, 
            x: torch.Tensor, 
            n_frames: int,
            speed: float = 1.0,
            interp_uv=True
    ) -> torch.Tensor:
        """
        Analyze the given audio signal to extract F0.
        
        Args:
            x (torch.Tensor): Audio signal tensor of shape (t,).
            n_frames (int): Number of frames, equal to the length of mel_spec.
        
        Returns:
            torch.Tensor: Extracted F0 of shape (n_frames,).
        """
        x = x.to('cpu').numpy()

        if self.f0_extractor == 'harvest':
            f0 = self._extract_f0_harvest(x, n_frames, speed)
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")
        
        uv = f0 == 0
        if interp_uv:
            f0, uv = interp_f0(f0, uv)
        return f0, uv

    def _extract_f0_parselmouth(self, x: np.ndarray, n_frames, speed: float):
        new_hop_size = int(np.round(self.hop_size * speed))
        l_pad = int(np.ceil(1.5 / self.f0_min * self.sampling_rate))
        r_pad = new_hop_size * ((len(x) - 1) // new_hop_size + 1) - len(x) + l_pad + 1
        padded_signal = np.pad(x, (l_pad, r_pad))
        
        sound = pm.Sound(padded_signal, self.sampling_rate)
        pitch = sound.to_pitch_ac(
            time_step=new_hop_size / self.sampling_rate, 
            voicing_threshold=0.6,
            pitch_floor=self.f0_min, 
            pitch_ceiling=self.f0_max
        )
        assert np.abs(pitch.t1 - 1.5 / self.f0_min) < 0.001
        f0 = pitch.selected_array['frequency']
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        f0 = f0[:n_frames]

        return f0

    def _extract_f0_harvest(self, x: np.ndarray, n_frames: int, speed: float) -> np.ndarray:
        new_hop_size = int(np.round(self.hop_size * speed))
        f0, _ = pw.harvest(
            x.astype(np.float64),
            self.sampling_rate, 
            f0_floor=self.f0_min, 
            f0_ceil=self.f0_max, 
            frame_period=(1000 * new_hop_size / self.sampling_rate)
        )
        f0 = f0.astype(np.float32)
        if f0.size < n_frames:
            f0 = np.pad(f0, (0, n_frames - f0.size))
        f0 = f0[:n_frames]
        return f0
    
    

if __name__ == '__main__':
    pass
