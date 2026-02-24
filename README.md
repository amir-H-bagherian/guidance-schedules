# Guidance Schedules
A small notebook / script that demonstrates sampling from Stable Diffusion with a per-step classifier-free guidance (CFG) schedule.
It loads a Stable Diffusion v1.5 model, builds different CFG schedules (constant, linear, cosine, gaussian, beta, …), and renders a grid of images so you can compare how guidance schedules affect results.

This repo is intended as a reproducible playground for experimenting with guidance schedules and deterministic sampling.

<p align="center">
  <a href="examples/results/preview.png">
    <img src="./results/output.png" alt="Guidance schedule example" width="900"/>
  </a>
</p>

<br>

# The Main Idea Of Our Project
In this project, we aimed to test different guidance schedules in order to find the best scheme to guide generation process. After this initial phase, out plan was to use the proper schedule and guide the model brutally not caring if we leave data distribution. <br>
We hypothesis this scheme ensures alignment even for poor initial noise. The "**Brutally guided**" image may seem out of distribution afterwards (e.g. it might entail saturated colors) which we try to mitigate using the second phase of project called "**Get Back**".<br>
Using some recovery schemes, we try to bring the out-of-data-distribution noisy-image back to the data manifold.<br>

Using this 2-Stage workflow, we hope to generate aesthetic images which align well with given text condition regardless of the initial noise. This might cost us a little quality reduction, but we try to keep it minimum.


# Features

- Loads runwayml/stable-diffusion-v1-5 and runs sampling with a custom, per-step CFG schedule.

- Several built-in schedules: constant, linear_decay, exponential_decay, cosine_decay, gaussian_decay, beta, sin_beta, step_early, and more.

- Optionally saves intermediate decodes for inspection.

- Deterministic-by-default settings (seed control, cuDNN flags, cuBLAS workspace config) to help reproducibility.


# Quick start

### Requirements
- Python 3.8+
- A GPU with CUDA (recommended) for reasonable runtime (CPU will work but be very slow).
- A Hugging Face token if you need access to gated models (set HF_TOKEN env var or login via huggingface-cli).

## How to run
1. Open the notebook (or paste the code into a Colab notebook).
 
2. Ensure model_id = "runwayml/stable-diffusion-v1-5" is OK for your use.

3. Run the cells in order. The final grid image is written to results/grid_results_debug.png by default.
