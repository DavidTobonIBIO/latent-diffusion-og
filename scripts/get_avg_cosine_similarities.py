import os
import pickle as pkl
import numpy as np
from txt2img import load_model_from_config
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch

def load_prompts():
    print("Loading prompts from AG paper and DenseDiffusion datasets...")
    prompts = []
    
    this_file_dir = os.path.dirname(os.path.abspath(__file__))

    parent_dir = os.path.dirname(this_file_dir)
    data_dir = os.path.join(parent_dir, "data")

    ag_prompts_file = os.path.join(data_dir, "ag_paper_prompts", "ag_paper_prompts.txt")
    dense_diffusion_dir = os.path.join(data_dir, "dense_diffusion")
    dense_diffusion_valset_file = os.path.join(dense_diffusion_dir, "DenseDiffusion_valset.pkl")
    dense_diffusion_testset_file = os.path.join(dense_diffusion_dir, "DenseDiffusion_testset.pkl")

    # print("AG Paper Prompts File:", ag_prompts_file)
    # print("Dense Diffusion Validation Set File:", dense_diffusion_valset_file)
    # print("Dense Diffusion Test Set File:", dense_diffusion_testset_file)

    with open(ag_prompts_file, "r") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
                
    print(f"\tLoaded {len(prompts)} prompts from AG paper prompts file.")

    with open(dense_diffusion_valset_file, "rb") as f:
        dense_diffusion_valset = pkl.load(f)
        for i in dense_diffusion_valset:
            item = dense_diffusion_valset[i]
            prompt = item.get("textual_condition", "").strip()
            if prompt:
                prompts.append(prompt)
                
    print(f"\tLoaded {len(dense_diffusion_valset)} prompts from DenseDiffusion validation set.")

    with open(dense_diffusion_testset_file, "rb") as f:
        dense_diffusion_testset = pkl.load(f)
        for i in dense_diffusion_testset:
            item = dense_diffusion_testset[i]
            prompt = item.get("textual_condition", "").strip()
            if prompt:
                prompts.append(prompt)
                
    print(f"\tLoaded {len(dense_diffusion_testset)} prompts from DenseDiffusion test set.")
    
    print(f"Total prompts loaded: {len(prompts)}")
    
    return prompts

def get_output_filename():
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_file_dir)
    outputs_dir = os.path.join(parent_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_dir = os.path.join(outputs_dir, "txt2img_experiments")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "avg_cosine_similarities.png")
    
    return output_filename

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    prompts = load_prompts()

    cosine_similarities_results = {}
    timesteps = np.arange(0, 50)
    scales = [5.0, 7.5, 10.0]
    
    solvers = ["ddim", "plms"]
    
    for solver in solvers:
        config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
        model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
        model = model.to(device)
        for guidance_scale in scales:
            for prompt in prompts:
                print(f"Generating sample for prompt: {prompt}")
                with torch.no_grad():
                    uc = model.get_learned_conditioning(1 * [""])
                    c = model.get_learned_conditioning(1 * [prompt])
                cosine_similarities = 
                
                cosine_similarities_results.append(cosine_similarities)
                
                break

    cosine_similarities_means = np.mean(cosine_similarities_results, axis=0)

    output_filename = get_output_filename()

    
    