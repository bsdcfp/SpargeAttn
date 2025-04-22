######################################################################
#
# Copyright (c) 2025 Shopee Inc. All Rights Reserved.
#
######################################################################

"""
@FileName: flux_fill_example.py
@Author: fuping.chu
@Email: fuping.chu@shopee.com
@Date:   2025-04-07 04:25:47
@Desc: 
"""
import torch
# from diffusers import FluxPipeline
from diffusers import FluxFillPipeline
from diffusers import FluxTransformer2DModel
from diffusers.utils import load_image

import torch, argparse
from modify_model.modify_flux import set_spas_sage_attn_flux
import os, gc
from spas_sage_attn.autotune import (
    extract_sparse_attention_state_dict,
    load_sparse_attention_state_dict,
)
import time
from datetime import datetime

file_path = "evaluate/datasets/video/prompts.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Flux-fill Evaluation")

    parser.add_argument("--use_spas_sage_attn", action="store_true", help="Use Sage Attention")
    parser.add_argument("--tune", action="store_true", help="tuning hyperpamameters")
    parser.add_argument('--parallel_tune', action='store_true', help='enable prallel tuning')
    parser.add_argument('--l1', type=float, default=0.06, help='l1 bound for qk sparse')
    parser.add_argument('--pv_l1', type=float, default=0.065, help='l1 bound for pv sparse')
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument(
        "--out_path",
        type=str,
        default="evaluate/datasets/image/flux_fill_sparge",
        help="out_path",
    )
    parser.add_argument(
        "--model_out_path",
        type=str,
        default="evaluate/models_dict/flux_fill_saved_state_dict.pt",
        help="model_out_path",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=50, help="The sampling steps.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(  
        "--image",  
        type=str,  
        default=None,  
        help="Path to the input image file for generation or editing."  
    )  
    parser.add_argument(  
        "--image-mask",  
        type=str,  
        default=None,  
        help="Optional path to a mask image file. Specifies regions in the input image to modify or inpaint."  
    )  
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # with open(file_path, "r", encoding="utf-8") as file:
    #     prompts = file.readlines()
    prompts = ["a white paper cup"]
    image_path="./assets/cup.png"
    mask_path = "./assets/cup_mask.png"

    image = load_image(image_path)
    mask = load_image(mask_path)
    # model_id = "black-forest-labs/FLUX.1-dev"
    # model_id = "/model_zoo/flux.1_dev/"
    # model_id = "/model_zoo/FLUX.1-dev"
    model_id = "/model_zoo/flux.1_fill_dev"
    
    if args.parallel_tune:
        os.environ['PARALLEL_TUNE'] = '1'
    if args.tune == True:
        os.environ["TUNE_MODE"] = "1"  # enable tune mode

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.float16,
        )
        if args.use_spas_sage_attn:
            set_spas_sage_attn_flux(transformer, verbose=args.verbose, l1=args.l1, pv_l1=args.pv_l1)

        pipe = FluxFillPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.float16,
        ).to("cuda")

        # pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        for i, prompt in enumerate(prompts[:10]):
            image = pipe(
                prompt=prompt.strip(),
                image=image,
                mask_image=mask,
                height=512,
                width=512,
                guidance_scale=30,
                num_inference_steps=args.sample_steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]

            del image
            gc.collect()
            torch.cuda.empty_cache()

        saved_state_dict = extract_sparse_attention_state_dict(transformer)
        torch.save(saved_state_dict, args.model_out_path)

    else:
        os.environ["TUNE_MODE"] = ""  # disable tune mode
        print("=====")

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            local_files_only=True,
            subfolder="transformer"
        )
        if args.use_spas_sage_attn:
            set_spas_sage_attn_flux(transformer, verbose=args.verbose, l1=args.l1, pv_l1=args.pv_l1)
            saved_state_dict = torch.load(args.model_out_path)
            load_sparse_attention_state_dict(transformer, saved_state_dict)
        if not args.use_spas_sage_attn:
            args.out_path = "evaluate/datasets/image/flux_fill_FA"
            os.makedirs(args.out_path, exist_ok=True)

        pipe = FluxFillPipeline.from_pretrained(
            model_id,
            transformer=transformer
        ).to("cuda")

        # pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()
        height =  512
        width = 512
        for i, prompt in enumerate(prompts):
            start_time = time.perf_counter() 
            image = pipe(
                prompt=prompt.strip(),
                image=image,
                mask_image=mask,
                height=height,
                width=width,
                guidance_scale=30,
                num_inference_steps=args.sample_steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            elapsed = time.perf_counter() - start_time  
            print(f"Inference [{i}] execution time: {elapsed:.2f} seconds "
                  f"({int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds)")  
            
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = prompt.replace(" ", "_").replace("/","_")[:40]
            model_name=model_id.strip().strip("/").split("/")[-1]
            if args.use_spas_sage_attn:
                save_file = (f"{model_name}_sparse_{args.l1}_{args.pv_l1}"
                             f"_steps{args.sample_steps}_{height}x{width}" 
                            f"_{formatted_prompt}_{formatted_time}"
                            f".jpg")
            else:
                save_file = (f"{model_name}_steps{args.sample_steps}_{height}x{width}" 
                            f"_{formatted_prompt}_{formatted_time}"
                            f".jpg")
            save_file = os.path.join(args.out_path, save_file)
            
            # image.save(f"{args.out_path}/{i}.jpg")
            print(f"Save file path: {save_file}")
            image.save(save_file)
            del image
            gc.collect()
            torch.cuda.empty_cache()
