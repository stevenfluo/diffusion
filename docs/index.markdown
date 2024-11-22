---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Home
permalink: /
---

<style>
        .grid-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            text-align: center;
        }
        .grid-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .grid-item img {
            width: 100%;
            height: auto;
        }
        .grid-item p {
            font-style: italic;
        }
</style>

<!-- MathJax -->
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# **Part A: The Power of Diffusion Models**

# Overview

In this project, I experimented with diffusion models, implemented diffusion sampling loops, and used them for tasks including inpainting and creating optical illusions.

# Part A.1: Setup

I used Stability AI's DeepFloyd IF two-stage diffusion model. The first stage produces images of size 64x64 and the second stage takes the outputs of the first stage and generates images of size 256x256.

To test the model, I used the sample captions to generate images. I use a random seed of 180 throughout the project.

Here are the stage 1 and stage 2 outputs when `num_inference_steps = 20`.

<p align="center">
    <img src="./img/Unknown.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-1.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-2.png" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Stage 1 — 1. "an oil painting of a snowy mountain village", 2. "a man wearing a hat", 3. "a rocket ship"</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-3.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-4.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-5.png" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Stage 2 — 1. "an oil painting of a snowy mountain village", 2. "a man wearing a hat", 3. "a rocket ship"</i></p>
</p>

Here are the stage 1 and stage 2 outputs when `num_inference_steps = 50`.

<p align="center">
    <img src="./img/Unknown-6.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-7.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-8.png" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Stage 1 — 1. "an oil painting of a snowy mountain village", 2. "a man wearing a hat", 3. "a rocket ship"</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-9.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-10.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-11.png" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Stage 2 — 1. "an oil painting of a snowy mountain village", 2. "a man wearing a hat", 3. "a rocket ship"</i></p>
</p>

# Part A.2: Sampling Loops

# Part A.2.1: Implementing the Forward Process

I use sampling loops to generate images from the diffusion model through an iterative denoising process: starting form pure noise at timestep T (sample from a Gaussian distribution), we can predict and remove part of the noise, repeating this process until we arrive at a clean image. DeepFloyd models do this over 1000 timesteps.

The forward process adds noise to a clean image from a Gaussian distribution with a specific mean and variance at each timestep.

`alphas_cumprod` is the hyperparameter denotes the noise level, where smaller t values correspond to cleaner images.

The function forward(im, t) produces a noised image at step t.

<p align="center">
    <img src="./img/Unknown-12.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-13.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-14.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-15.png" alt="ad" width="22%"/>
    <p style="text-align: center;"><i>Four views of the Campanile: no noise, noisy at t=250, noisy at t=500, noisy at t=750.</i></p>
</p>

## Part A.2.2: Classical Denoising

I use Gaussian blur filtering to denoise the noised images.

<p align="center">
    <img src="./img/Unknown-13.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-14.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-15.png" alt="ad" width="30%"/>
    <p style="text-align: center;"><i> Noisy at t=250, noisy at t=500, noisy at t=750.</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-16.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-17.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-18.png" alt="ad" width="30%"/>
    <p style="text-align: center;"><i> Gaussian blur denoising at t=250, denoising at t=500, denoising at t=750.</i></p>
</p>

## Part A.2.3: One-Step Denoising

For one-step denoising, I used the UNet to denoise the image by estimating the noise. First, I estimated the noise in the new noisy image, by passing it through `stage_1.unet`, which I removed from the noisy image to estimate the original one.

<p align="center">
    <img src="./img/Unknown-13.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-14.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-15.png" alt="ad" width="30%"/>
    <p style="text-align: center;"><i> Noisy at t=250, noisy at t=500, noisy at t=750.</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-19.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-20.png" alt="ad" width="30%"/>
    <img src="./img/Unknown-21.png" alt="ad" width="30%"/>
    <p style="text-align: center;"><i> One-step denoising at t=250, denoising at t=500, denoising at t=750.</i></p>
</p>

## Part A.2.4: Iterative Denoising

Diffusion models perform better when iteratively denoising images — that's how they were designed! Even though I want to iteratively denoise my noisy images across 1000 timesteps, I skip steps to speed things up, using `strided_timesteps` to iteratively take small strided timesteps in order to produce a clean image.

<p align="center">
    <img src="./img/Unknown-22.png" alt="ad" width="15%"/>
    <img src="./img/Unknown-23.png" alt="ad" width="15%"/>
    <img src="./img/Unknown-24.png" alt="ad" width="15%"/>
    <img src="./img/Unknown-25.png" alt="ad" width="15%"/>
    <img src="./img/Unknown-26.png" alt="ad" width="15%"/>
    <img src="./img/Unknown-27.png" alt="ad" width="15%"/>
    <p style="text-align: center;"><i> Iteratively denoising: iteration 10, t=690; iteration 15, t=540; iteration 20, t=390; iteration 25, t=240; iteration 30, t=90; fully denoised.</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-12.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-27.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-28.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-29.png" alt="ad" width="22%"/>
    <p style="text-align: center;"><i> Original image, iteratively denoised, one-step denoised, and Gaussian blur denoised images. </i></p>
</p>

Iterative denoising clearly performs better than the other methods!


## Part A.2.5: Diffusion Model Sampling

Using the `iterative_denoise` function I implemented, I can also generate images from scratch! I do this by setting `i_start = 0` and passing in random noise (drawn from a Gaussian distribution) — essentially denoising pure noise. This method and the prompt "a high quality photo" yields these sampled images:

<p align="center">
    <img src="./img/Unknown-43.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-44.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-32.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-33.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-34.png" alt="ad" width="18%"/>
    <p style="text-align: center;"><i> Images generated from pure noise. </i></p>
</p>

Here's an example of the denoising process, visualized with intermediate images:
<p align="center">
    <img src="./img/Unknown-35.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-36.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-37.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-38.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-39.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-40.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-41.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-42.png" alt="ad" width="11%"/>
    <p style="text-align: center;"><i> Iteratively denoising: iteration 0, t=990; iteration 5, t=840; iteration 10, t=690; iteration 15, t=540; iteration 20, t=390; iteration 25, 5=240; iteration 30, t=90; denoised image.</i></p>
</p>

## Part A.2.6: Classifier-Free Guidance

Classifier-Free Guidance (CFG) is a technique that allows us to greatly improve image quality, at the expense of image diversity. In CFG, we compute a conditional and unconditional noise estimate, which we then use to calculate a noise estimate such that:

`noise estimate = unconditional estimate + gamma * (conditional noise estimate - unconditional noise estimate)`

To obtain the unconditional noise estimate, we use an empty prompt embedding. For the conditional noise estimate, we use the embedding for "a high quality photo". The gamma parameter controls the strength of CFG such that a gamma of 0 results in the unconditional noise estimate and a gamma of 1 we get the conditional noise estimate. By setting gamma = 7 (or any gamma such that gamma > 1) we produce much higher quality images. 

Five images produced with `iterative_denoise_cfg` following this process are shown below:

<p align="center">
    <img src="./img/Unknown-45.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-46.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-47.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-48.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-49.png" alt="ad" width="18%"/>
    <p style="text-align: center;"><i> Images generated from pure noise (with CFG). </i></p>
</p>

Here's an example of the denoising process, visualized with intermediate images:
<p align="center">
    <img src="./img/Unknown-50.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-51.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-52.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-53.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-54.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-55.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-56.png" alt="ad" width="11%"/>
    <img src="./img/Unknown-57.png" alt="ad" width="11%"/>
    <p style="text-align: center;"><i> Iteratively denoising: iteration 0, t=990; iteration 5, t=840; iteration 10, t=690; iteration 15, t=540; iteration 20, t=390; iteration 25, 5=240; iteration 30, t=90; denoised image.</i></p>
</p>

## Part A.2.7: Image to Image Translation

In image-to-image translation, we take an original image, noise it by a bit, and then force it back onto the image manifold without any conditioning. By following the SDEdit algorithm, we are able to get an image that's similar to the original one.

Procedurally, I run the forward process to get a noisy image, then I run `iterative_denoise_cfg` with different starting indices to create a series of "edits" to the image. The closer the starting index is to 0, the bigger the edit compared to the original image.

Here's the Campanile:
<p align="center">
    <img src="./img/Unknown-58.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-59.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-60.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-61.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-62.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-63.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-64.png" alt="ad" width="13%"/>
    <p style="text-align: center;"><i> SDEdit with i_start=1, i_start=3, i_start=5, i_start=7, i_start=10, i_start=20, original image.</i></p>
</p>

Here's a picture of my friend's cat, Pocky:
<p align="center">
    <img src="./img/Unknown-65.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-66.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-67.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-68.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-69.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-70.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-71.png" alt="ad" width="13%"/>
    <p style="text-align: center;"><i> SDEdit with i_start=1, i_start=3, i_start=5, i_start=7, i_start=10, i_start=20, original image.</i></p>
</p>

I guess Pocky is a dog-like cat?

Here's a picture of my other friend's two kittens:
<p align="center">
    <img src="./img/Unknown-72.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-73.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-74.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-75.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-76.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-77.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-78.png" alt="ad" width="13%"/>
    <p style="text-align: center;"><i> SDEdit with i_start=1, i_start=3, i_start=5, i_start=7, i_start=10, i_start=20, original image.</i></p>
</p>

## Part A.2.8: Editing Hand-Drawn and Web Images

Here's the sequence for an image from the web (a painting of the signing of the Constitution from Wikipedia): 
<p align="center">
    <img src="./img/Unknown-79.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-80.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-81.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-82.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-83.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-84.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-85.png" alt="ad" width="13%"/>
    <p style="text-align: center;"><i> SDEdit with i_start=1, i_start=3, i_start=5, i_start=7, i_start=10, i_start=20, original image.</i></p>
</p>

I drew a chicken:
<p align="center">
    <img src="./img/Unknown-86.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-87.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-88.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-89.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-90.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-91.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-92.png" alt="ad" width="13%"/>
    <p style="text-align: center;"><i> SDEdit with i_start=1, i_start=3, i_start=5, i_start=7, i_start=10, i_start=20, original image.</i></p>
</p>

I drew an imposter:
<p align="center">
    <img src="./img/Unknown-93.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-94.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-95.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-96.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-97.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-98.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-99.png" alt="ad" width="13%"/>
    <p style="text-align: center;"><i> SDEdit with i_start=1, i_start=3, i_start=5, i_start=7, i_start=10, i_start=20, original image.</i></p>
</p>

## Part A.2.9: Inpainting

By applying a mask such that we leave everything inside the edit mask alone but replace everything outside the edit mask with our original image (and with the correct amount of noise added for the timestep we're on), we can create a new image that has new content in the appropriately masked area. We're basically forcing the model to fill in the gaps — gaps that we create!

Campanile to lighthouse:
<p align="center">
    <img src="./img/Unknown-100.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-101.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-102.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-103.png" alt="ad" width="22%"/>
    <p style="text-align: center;"><i> Original image, mask, to replace, and inpainted image. </i></p>
</p>

Cat rhinoplasty:
<p align="center">
    <img src="./img/Unknown-104.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-105.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-106.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-107.png" alt="ad" width="22%"/>
    <p style="text-align: center;"><i> Original image, mask, to replace, and inpainted image. </i></p>
</p>

Kitten head accessories:
<p align="center">
    <img src="./img/Unknown-108.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-109.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-110.png" alt="ad" width="22%"/>
    <img src="./img/Unknown-111.png" alt="ad" width="22%"/>
    <p style="text-align: center;"><i> Original image, mask, to replace, and inpainted image. </i></p>
</p>

## Part A.2.10: Text-Conditional Image-to-Image Translation

Now, we add control using language by changing the prompt!

Campanile with the prompt "a rocket ship":
<p align="center">
    <img src="./img/Unknown-112.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-113.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-114.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-115.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-116.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-117.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-118.png" alt="ad" width="13%"/>
    <p style="text-align: center;"><i> SDEdit with i_start=1, i_start=3, i_start=5, i_start=7, i_start=10, i_start=20, original image.</i></p>
</p>

Pocky (a cat) with the prompt "a photo of a dog":
<p align="center">
    <img src="./img/Unknown-119.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-120.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-121.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-122.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-123.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-124.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-125.png" alt="ad" width="13%"/>
    <p style="text-align: center;"><i> SDEdit with i_start=1, i_start=3, i_start=5, i_start=7, i_start=10, i_start=20, original image.</i></p>
</p>

A smiling emoji with the prompt "a photo of a hipster barista":
<p align="center">
    <img src="./img/Unknown-126.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-127.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-128.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-129.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-130.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-131.png" alt="ad" width="13%"/>
    <img src="./img/Unknown-132.png" alt="ad" width="13%"/>
    <p style="text-align: center;"><i> SDEdit with i_start=1, i_start=3, i_start=5, i_start=7, i_start=10, i_start=20, original image.</i></p>
</p>

## Part A.2.11: Visual Anagrams

In this part, I create images that look different when flipped upside down! I do this by denoising an image normally with one prompt to obtain some noise estimate, and then I denoise a flipped image with a different prompt to obtain a different noise estimate. For the iterative denoising process, the final noise estimate is just the average of these two.

"an oil painting of people around a campfire" and "an oil painting of an old man"
<p align="center">
    <img src="./img/Unknown-133.png" alt="ad" width="45%"/>
    <img src="./img/Unknown-134.png" alt="ad" width="45%"/>
    <p style="text-align: center;"><i>Campfire or old man?</i></p>
</p>

"an oil painting of a snowy mountain village" and "a photo of a dog"
<p align="center">
    <img src="./img/Unknown-135.png" alt="ad" width="45%"/>
    <img src="./img/Unknown-136.png" alt="ad" width="45%"/>
    <p style="text-align: center;"><i>Snowy village or dog?</i></p>
</p>

"a photo of a hipster barista" and "a lithograph of a skull"
<p align="center">
    <img src="./img/Unknown-137.png" alt="ad" width="45%"/>
    <img src="./img/Unknown-138.png" alt="ad" width="45%"/>
    <p style="text-align: center;"><i>Barista or skull?</i></p>
</p>

## Part A.2.12: Hybrid Images
For this part, I create hybrid images. First, I apply UNet to the image with two different text prompts. Then, I take the low pass of the noise estimate for the first and the high pass of the noise estimate for the second to create the new noise estimate, resulting in an image that looks like the low-passed prompt when up close and the high-passed prompt when far away — just like in Project 2!

<p align="center">
    <img src="./img/Unknown-139.png" alt="ad" width="45%"/>
    <p style="text-align: center;"><i>A skull when close and waterfalls when far.</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-140.png" alt="ad" width="45%"/>
    <p style="text-align: center;"><i>A rocket ship when close and a pencil when far.</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-141.png" alt="ad" width="45%"/>
    <p style="text-align: center;"><i>A hipster barista when close and an old man when far.</i></p>
</p>


# Part A. 3: Bells and Whistles 
## Designing a Course Logo
I created my own prompt embeddings (including 'a brown bear typing on a computer', 'a picture of an eye', 'a laptop computer', 'a brown bear', 'a photo of sather gate', 'a photo of uc berkeley', 'a photo of the campanile') and used these to create various hybrid images that are CS180-themed. 

Try to figure out which prompts I used to create these!

<p align="center">
    <img src="./img/Unknown-154.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-155.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-156.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-157.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-158.png" alt="ad" width="18%"/>
    <p style="text-align: center;"><i> CS180 course logos? Hybrid images designed with custom prompts. </i></p>
</p>

<p align="center">
    <img src="./img/Unknown-154 copy.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-155 copy.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-156 copy.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-157 copy.png" alt="ad" width="18%"/>
    <img src="./img/Unknown-158 copy.png" alt="ad" width="18%"/>
</p>

# **Part B: Training a Diffusion Model**

# Overview

In this project, I trained my own diffusion model on MNIST!

# Part B.1: Training a Single-Step Denoising UNet

First, we build a one-step denoiser that's optimized over an L2 loss. This model maps noisy images to clean ones. Our denoiser is implemented as a [UNet](https://arxiv.org/abs/1505.04597), consisting of a few downsampling and upsampling blocks with skip connections.

## Part B.1.1: Implementing the UNet

I followed the following model architecture and operation block diagrams to implement the `UnconditionalUNet` class and the simple and composed operations:

<p align="center">
    <img src="./img/unconditional_arch.png" alt="ad" width="85%"/>
</p>
<p align="center">
    <img src="./img/atomic_ops_new.png" alt="ad" width="85%"/>
</p>

## Part B.1.2: Using the UNet to Train a Denoiser

To train the denoiser, we need pairs of (noisy, clean) MNIST images. I generate noisy images by adding noise sampled from a Gaussian distribution, scaled by some sigma. Here is a visualization of the noising process as we vary sigma, using a sample taken from the MNIST training set:

<p align="center">
    <img src="./img/Unknown-142.png" alt="ad" width="85%"/>
</p>

Clearly, the larger the sigma, the noisier the image.

I then train a model to denoise noisy images (training data consists of noised data with sigma = 0.5). I use `batch_size = 256`, `num_epochs = 5`, `learning_rate = 1e-4`, Adam optimizer, and `hidden_dims = 128`.

Our training loss curve is fairly standard: 
<p align="center">
    <img src="./img/Unknown-143.png" alt="ad" width="85%"/>
</p>

Let's take a look at our results after the first epoch of training:
<p align="center">
    <img src="./img/Unknown-144.png" alt="ad" width="85%"/>
</p>

Now for the results after the fifth epoch of training:
<p align="center">
    <img src="./img/Unknown-145.png" alt="ad" width="85%"/>
</p>

Our model seems to denoise noisy images better after being trained for 5 epochs compared to being trained for just 1 epoch. We see fewer artifacts around the number and in the background (look at the 1 and the 7) — as we'd expect after training for longer!

## Part B.1.3: Out-of-Distribution Testing

Our denoiser was trained on MNIST digits noised with `sigma = 0.5`. How does it perform on MNIST digits noised with a different value of sigma — values outside of the distribution of its training set?

<p align="center">
    <img src="./img/Unknown-146.png" alt="ad" width="85%"/>
</p>

As sigma increases past 0.5, we see the model's denoising performance is degraded, and the outputted image becomes more and more distorted. However, our model performs pretty well for a wide range of sigmas despite only being trained on images noised with sigma = 0.5, and the overall shape is clear in all outputs!

# Part B.2: Training a Diffusion Model

Now, I implement a diffusion model as a UNet to iteratively denoise images. Unlike our previous model where we predicted the clean image, this model predicts the added noise. For optimal performance, we should condition the UNet on the current timestep since the variance of the noisy image varies depends on the timestep. Part of my model implementation and many of the equations I follow come from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM).

## Part B.2.1: Adding Time Conditioning to UNet

To add the iterative element into our model, we inject a scalar timestep into our model using FCBlocks that are added after the Unflatten and the first UpBlock. The ConditionalUNet block diagram is:

<p align="center">
    <img src="./img/conditional_arch.png" alt="ad" width="85%"/>
</p>

Note that our timesteps are normalized to fall between 0 and 1 to minimize the effect of its magnitude and to allow for many timesteps.

## Part B.2.2: Training the UNet

To train the diffusion model, I pick a random image from the training set, a random timestep, and train the denoiser to predict the noise in the noisy image at the timestep. This is repeated for different images and values until the model converges.

Our training loss curve is indicative of a good training run: 
<p align="center">
    <img src="./img/Unknown-147.png" alt="ad" width="85%"/>
</p>

## Part B.2.3: Sampling from the UNet

To visualize model performance at different stages of training, I sample results for the time-conditioned UNet for 1, 5, and 20 epochs of training.

<p align="center">
    <img src="./img/Unknown-148.png" alt="ad" width="85%"/>
    <p style="text-align: center;"><i>Results after 1 epoch of training.</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-149.png" alt="ad" width="85%"/>
    <p style="text-align: center;"><i>Results after 5 epochs of training.</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-150.png" alt="ad" width="85%"/>
    <p style="text-align: center;"><i>Results after 20 epochs of training.</i></p>
</p>

Note that the results from the 20 epoch model seem more number-like than the results from earlier ones!

## Part B.2.4: Adding Class Conditioning

To make the results better and give us more control for image generation, we can condition our UNet on the class of the digit 0-9. Because we still want our UNet to work without it being conditioned on the class, we implement dropout where 10% of the time we drop the one-hot encoded class-conditioning vector by setting it to 0. The implementation is very similar to the time-conditioned UNet, with minor changes in the architecture (two more FCBlocks that take in the class-conditioning vector) and adding class-conditioning vector processing into the training loop.

Our resulting training loss curve is:
<p align="center">
    <img src="./img/Unknown-151.png" alt="ad" width="85%"/>
</p>

## Part B.2.5: Sampling from the Class-Conditioned UNet

To sample, I follow the same algorithm as sampling from the time-conditioned UNet, with changes to include class labels and also using classifier-free guidance to improve results. For my sampling implementation, I use gamma = 5.

I generate four instances of each digit while sampling results for the class-conditioned UNet after 5 and 20 epochs of training.

<p align="center">
    <img src="./img/Unknown-152.png" alt="ad" width="85%"/>
    <p style="text-align: center;"><i>Results after 5 epochs of training.</i></p>
</p>

<p align="center">
    <img src="./img/Unknown-153.png" alt="ad" width="85%"/>
    <p style="text-align: center;"><i>Results after 20 epochs of training.</i></p>
</p>

Note that the results from the 20 epoch model seem more number-like, are cleaner, and have less artifacts/distracting features than the results from the 5 epoch model!

## Reflection

This was definitely a tough project, but I enjoyed the practice of implementing models and algorithms from papers, from scratch. I was particularly surprised by how diffusion models are able to produce realistic outputs even if we sample nothing but pure noise! This project has definitely made me more curious about other generative methods, and how to implement/use those.