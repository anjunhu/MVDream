import os
import sys
import random
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch 
import torch.nn.functional as F
from tqdm import tqdm
from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from torchvision.transforms.functional import to_tensor

FORMAT_INTERLEAVED = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1,
        ddim_inversion=False, start_time_step=35):
    if type(prompt)!=list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        neutral_pose_prompt = "a bunny standing, ears relaxed, empty background"
        c0 = model.get_learned_conditioning(neutral_pose_prompt).to(device)
        c1 = model.get_learned_conditioning(neutral_pose_prompt).to(device)
        c_ = {"context": torch.cat([c0, c1]).repeat(batch_size//2,1,1)}
        ##### for vf, context in enumerate(c_['context']):  print(vf, context.shape, context[4:16])
        # uc = model.get_learned_conditioning("legs bent").to(device)
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames
        shape = [4, image_size // 8, image_size // 8] # [4, 32, 32]

        x_T = None
        start_time_step = None

        samples_ddim, intermediates = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=x_T,
                                        start_time_step=start_time_step,)
                                        # cached_trajectory="assets/saved_ddim_trajectories/x_inter_a_bird_resting_on_a_branch.torch")
        prompt = neutral_pose_prompt.replace(' ', '_').replace(',', '')
        torch.save(intermediates['x_inter'], f'assets/saved_ddim_trajectories/x_inter_{prompt}.torch')
        # for t, x_t in enumerate(intermediates['pred_x0']):
        #     x_sample = model.decode_first_stage(x_t)
        #     x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        #     x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
        #     x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
        #     Image.fromarray(x_sample).save(f"forward_cache_artefacts/pred_x0_t={t}.png")
        # for t, x_t in enumerate(intermediates['x_inter']):
        #     x_sample = model.decode_first_stage(x_t)
        #     x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        #     x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
        #     x_sample = np.concatenate(list(x_sample.astype(np.uint8)), 1)
        #     Image.fromarray(x_sample).save(f"forward_cache_artefacts/x_inter_t={t}.png")
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="a cow standing still legs straight")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--start_time_step", type=int, default=35, help="DDIM inversion start time step")
    parser.add_argument("--ddim_inversion", action="store_true")
    parser.add_argument("--num_frames", type=int, default=8, help="num of frames (views) to generate")
    parser.add_argument("--num_rows", type=int, default=1, help="number of rows to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = args.num_frames

    print("load t2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model.device = device
    model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning( [""] ).to(device)
    print("loaded t2i model. ")

    # pre-compute camera matrices
    if args.use_camera:
        camera = get_camera(args.num_frames//2, elevation=args.camera_elev, 
                azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
        if FORMAT_INTERLEAVED:
            camera = camera.repeat_interleave(2*batch_size//args.num_frames,dim=0).to(device)    # AABBCCDD
        else:
            camera = camera.repeat(2*batch_size//args.num_frames,1).to(device)                   # ABCDABCD
        print(camera.shape)
    else:
        camera = None
    
    t = args.text + args.suffix
    set_seed(args.seed)
    images = []
    for j in range(args.num_rows):
        img = t2i(model, args.size, t, uc, sampler, step=args.step, scale=10, batch_size=batch_size, ddim_eta=0.0, 
                dtype=dtype, device=device, camera=camera, num_frames=args.num_frames, ddim_inversion=args.ddim_inversion, start_time_step=args.start_time_step)
        for i, im in enumerate(img):
            Image.fromarray(im).save(f"sample_{i}.png")
        img = np.concatenate(img, 1)
        images.append(img)
    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f"sample.png")