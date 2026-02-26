# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at

#     https://creativecommons.org/licenses/by-nc/4.0/

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional

import torch
import math
from protenix.model.utils import centre_random_augmentation
from protenix.metrics.rmsd import weighted_rigid_align
from tqdm import tqdm
import random
from protenix.model.utils import expand_at_dim

class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    first_frame_noise: float = 4e-4,
    # --- Hierarchical generation parameters ---
    W_H: int = 1,
    W_G: int = 10,
    coarse_frame_num: int = 20,
    coarse_interval: float = 5,
    fine_frame_num: int = 5,
) -> torch.Tensor:
    """
    Hierarchical trajectory generation via two-stage diffusion sampling.

    Stage 1 (Coarse Forecasting):
        Autoregressively generates coarse_frame_num frames (including x_0)
        with temporal spacing = coarse_interval.  Uses a sliding history
        window of size W_H and generates W_G frames per batch.

    Stage 2 (Fine Interpolation):
        For every pair of consecutive coarse frames, generates
        (fine_frame_num - 1) intermediate frames conditioned on both
        boundary frames.  Temporal spacing = coarse_interval / fine_frame_num.

    Total output frames = 1 + (coarse_frame_num - 1) * fine_frame_num.

    Args:
        denoise_net (Callable): Denoising network.
        input_feature_dict (dict): Input features.
            'time_interval' will be overridden per stage.
        s_inputs  (torch.Tensor): [..., N_tokens, c_s_inputs]
        s_trunk   (torch.Tensor): [..., N_tokens, c_s]
        z_trunk   (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): Decreasing noise levels [N_steps].
        N_sample (int): Number of trajectory samples to draw.
        gamma0, gamma_min, noise_scale_lambda, step_scale_eta:
            Diffusion hyper-parameters (Alg.18 in AF3).
        diffusion_chunk_size (Optional[int]): Chunk over N_sample for memory.
        inplace_safe (bool): Whether inplace ops are safe.
        attn_chunk_size (Optional[int]): Attention chunk size.
        first_frame_noise (float): Small noise level for conditioned frames.
        W_H (int): History window size  (coarse forecasting).
        W_G (int): Generation window size (coarse forecasting).
        coarse_frame_num (int): Number of coarse frames including x_0.
        coarse_interval (float): Temporal spacing between coarse frames.
        fine_frame_num (int): Number of sub-intervals per coarse interval.
            fine_interval = coarse_interval / fine_frame_num.

    Returns:
        torch.Tensor: Full trajectory
            [N_total_frames, N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    device = s_inputs.device
    dtype = s_inputs.dtype
    fine_interval = coarse_interval / fine_frame_num
    coordinate_mask = input_feature_dict["frame0_coordinate_mask"]

    # ================================================================
    #  Per-chunk worker  (handles one slice of N_sample)
    # ================================================================
    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):

        # ------------------------------------------------------------
        #  Core denoising loop (shared by both stages)
        # ------------------------------------------------------------
        def _denoise(x_all, cond_mask, feat_dict):
            """
            Run the full noise_schedule denoising on x_all.

            Args:
                x_all     : [F, S, A, 3]  frames × samples × atoms × 3
                cond_mask : [F] bool tensor  (True → conditioned / frozen)
                feat_dict : shallow copy of input_feature_dict with the
                            correct 'time_interval' already set.
            Returns:
                x_all with generated (non-conditioned) frames denoised.
            """
            gen_idx = torch.where(~cond_mask)[0]
            cond_idx = torch.where(cond_mask)[0]
            t_hat_all = torch.full(
                (x_all.shape[0], chunk_n_sample),
                first_frame_noise, dtype=dtype, device=device,
            )

            for c_tau_last, c_tau in zip(noise_schedule[:-1], noise_schedule[1:]):
                gamma = float(gamma0) if c_tau > gamma_min else 0.0
                t_hat = c_tau_last * (gamma + 1.0)
                d_sigma = torch.sqrt(t_hat ** 2 - c_tau_last ** 2)

                x_noisy = x_all.clone()

                # Stochastic noise on generated frames
                if gen_idx.numel() > 0:
                    x_noisy[gen_idx] += (
                        noise_scale_lambda * d_sigma
                        * torch.randn(
                            gen_idx.numel(), chunk_n_sample, N_atom, 3,
                            device=device, dtype=dtype,
                        )
                    )
                # Small noise on conditioned frames
                if cond_idx.numel() > 0:
                    x_noisy[cond_idx] += (
                        first_frame_noise
                        * torch.randn_like(x_noisy[cond_idx])
                    )

                # Per-frame noise level
                t_hat_all[~cond_mask] = t_hat
                t_hat_all[cond_mask] = first_frame_noise

                x_denoised = denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat_all.clone(),
                    input_feature_dict=feat_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                )

                # Denoising update (only generated frames)
                delta = (x_noisy - x_denoised) / t_hat
                dt = c_tau - t_hat
                x_all[gen_idx] = (x_noisy + step_scale_eta * dt * delta)[gen_idx]

            return x_all

        # ------------------------------------------------------------
        #  Prepare x_0  (initial structure, augmented for S samples)
        # ------------------------------------------------------------
        x_init = torch.zeros(1, N_atom, 3, device=device, dtype=dtype)
        x_init[0] = input_feature_dict["frame0_coordinate"]
        x_init = centre_random_augmentation(
            x_input_coords=x_init,
            N_sample=chunk_n_sample,
            mask=coordinate_mask,
            s_trans=0.,
        ).to(dtype)                                       # [1, S, A, 3]
        x_0 = x_init[0].clone()                           # [S, A, 3]
        x_0[:, coordinate_mask == 0, :] = 0.

        # ============================================================
        #  Stage 1 – Coarse-grained Forecasting
        # ============================================================
        print("=== Stage 1: Coarse Forecasting ===")
        coarse_feat_dict = {**input_feature_dict, "time_interval": coarse_interval}

        X_C = [x_0]                                        # list of [S, A, 3]
        idx = 0

        while idx < coarse_frame_num - 1:
            n_gen = min(W_G, coarse_frame_num - 1 - idx)
            print(f"  coarse: generating frames {idx + 1}..{idx + n_gen}")

            # ---- Select last W_H frames as history context ----
            h0 = max(0, len(X_C) - W_H)
            X_h = torch.stack(X_C[h0:], dim=0)            # [Nh, S, A, 3]
            Nh = X_h.shape[0]

            # Align all history frames to the first history frame
            with torch.no_grad():
                ref = X_h[0]                               # [S, A, 3]
                for fi in range(1, Nh):
                    for si in range(chunk_n_sample):
                        X_h[fi, si] = weighted_rigid_align(
                            x=X_h[fi, si],
                            x_target=ref[si],
                            atom_weight=coordinate_mask,
                            stop_gradient=True,
                        )

            # Initialise generation frames with full noise
            x_gen_noise = noise_schedule[0] * torch.randn(
                n_gen, chunk_n_sample, N_atom, 3,
                device=device, dtype=dtype,
            )

            # Concatenate: [history | generation]
            x_all = torch.cat([X_h, x_gen_noise], dim=0)  # [Nh+ng, S, A, 3]
            cond_mask = torch.zeros(Nh + n_gen, dtype=torch.bool, device=device)
            cond_mask[:Nh] = True

            # Run denoising
            x_all = _denoise(x_all, cond_mask, coarse_feat_dict)

            # Store newly generated coarse frames
            for b in range(n_gen):
                X_C.append(x_all[Nh + b].clone())
            idx += n_gen

        X_C_tensor = torch.stack(X_C, dim=0)              # [Lc, S, A, 3]
        print(f"  coarse trajectory: {X_C_tensor.shape[0]} frames")

        # ============================================================
        #  Stage 2 – Fine-grained Interpolation
        # ============================================================
        print("=== Stage 2: Fine Interpolation ===")
        fine_feat_dict = {**input_feature_dict, "time_interval": fine_interval}
        N_inner = fine_frame_num - 1  # intermediate frames to generate per interval

        X_F = [X_C_tensor[0]]                             # start with x_0

        for i in range(coarse_frame_num - 1):
            x_start = X_C_tensor[i].clone()                # [S, A, 3]
            x_end = X_C_tensor[i + 1].clone()              # [S, A, 3]

            if N_inner > 0:
                print(f"  fine: interpolating interval {i} -> {i + 1}")
                # Initialise intermediate frames with full noise
                x_gen_noise = noise_schedule[0] * torch.randn(
                    N_inner, chunk_n_sample, N_atom, 3,
                    device=device, dtype=dtype,
                )

                # Arrange: [x_start, gen_1, …, gen_K, x_end]
                x_all = torch.cat([
                    x_start.unsqueeze(0),
                    x_gen_noise,
                    x_end.unsqueeze(0),
                ], dim=0)                                  # [Ni+2, S, A, 3]

                cond_mask = torch.zeros(N_inner + 2, dtype=torch.bool, device=device)
                cond_mask[0] = True
                cond_mask[-1] = True

                # Run denoising
                x_all = _denoise(x_all, cond_mask, fine_feat_dict)

                # Collect interpolated inner frames
                for j in range(1, N_inner + 1):
                    X_F.append(x_all[j].clone())

            # Append the coarse end-frame
            X_F.append(x_end)

        X_F_tensor = torch.stack(X_F, dim=0)              # [Ntotal, S, A, 3]
        print(f"  full trajectory: {X_F_tensor.shape[0]} frames")
        return X_F_tensor

    # ================================================================
    #  Optional chunking over N_sample
    # ================================================================
    if diffusion_chunk_size is None:
        return _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)

    parts = []
    n_chunks = math.ceil(N_sample / diffusion_chunk_size)
    for i in tqdm(range(n_chunks)):
        cs = (
            diffusion_chunk_size
            if i < n_chunks - 1
            else N_sample - i * diffusion_chunk_size
        )
        parts.append(
            _chunk_sample_diffusion(cs, inplace_safe=inplace_safe)
        )
    return torch.cat(parts, dim=1)  # concatenate along sample dimension


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        N_sample (int): number of training samples
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    # breakpoint()
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    traj_len = label_dict["traj_len"]
    traj = []
    for i in range(traj_len):
        traj.append(label_dict[f"coordinate_{i}"])

    x_original = torch.stack(traj, dim=0)

    with torch.no_grad():
        # Use first frame as reference
        ref_frame = x_original[0]  # [N_atom, 3]
        
        # Align all frames to the reference frame
        x_aligned = torch.zeros_like(x_original)
        x_aligned[0] = ref_frame
        for i in range(1, x_original.shape[0]):
            x_aligned[i] = weighted_rigid_align(
                x=x_original[i],
                x_target=ref_frame,
                atom_weight=label_dict["coordinate_mask"].float(),
                stop_gradient=True
            )

    # Create N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=x_aligned,
        N_sample=N_sample,
        mask=None,
        s_trans=0.,
    ).to(dtype)  # [N_frame, N_sample, N_atom, 3]

    N_frame = x_gt_augment.shape[0]


    # =============== for training MD=======================
    if random.random() <= 1/2:
        # Add independent noise to each frame
        sigma = noise_sampler(size=(N_frame, N_sample), device=device).to(dtype)
    else:
        # Add identical noise to each frame
        sigma = noise_sampler(size=(N_sample,), device=device).to(dtype)
        sigma = expand_at_dim(sigma, dim=0, n=N_frame).clone()

    # Set first and last frame noise to zero, => forecasting and interpolation
    for idx in range(N_sample):
        if random.random() <= 1/2:
            sigma[0, idx] = 4e-4 # s_min for inference_noise_scheduler
            if random.random() <= 1/2:
                sigma[-1, idx] = 4e-4 # s_min for inference_noise_scheduler
    
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]

    # Get denoising outputs [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_gt_augment + noise, # [N_frame, N_sample, N_atom, 3]
            t_hat_noise_level=sigma, # [N_frame, N_sample]
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs, # [N_atom, c_s_inputs]
            s_trunk=s_trunk, # [N_atom, c_s]
            z_trunk=z_trunk, # [N_atom, N_atom, c_z]
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
            )
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigma