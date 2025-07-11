# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new latent tokens from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from latent_token_models import DiT_models
import argparse
import numpy as np


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    model = DiT_models[args.model](
        num_classes=args.num_classes,
        in_channels=args.token_dim,
        num_tokens=args.num_tokens,
    
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    
    all_samples = []
    for i in range(args.num_k_samples):
        print(f"num_k_samples {i} out of {args.num_k_samples}")
        
        # Labels to condition the model with (feel free to change):
        class_labels = [0]*64#[207, 360, 387, 974, 88, 979, 417, 279]

        # Create sampling noise:
        n = len(class_labels)
        # z = torch.randn(n, 4, latent_size, latent_size, device=device)
        z = torch.randn(n, args.token_dim, args.num_tokens, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample latent tokens:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)
    
    # save npz file
    np_samples = np.asarray(all_samples.cpu()).transpose(0, 2, 1)
    print('samples shape', np_samples.shape)
    print(f'sample mean {np_samples.mean()} std {np_samples.std()}')
    np.savez(args.outfile, np_samples/args.normalization)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/8")
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--normalization", type=float, default=200.)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-k-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--outfile", type=str, default='latent_token_samples.npz',
                        help="Optional path to output file.")
    args = parser.parse_args()
    main(args)
