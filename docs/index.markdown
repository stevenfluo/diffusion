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

# **Part B: Training a Diffusion Model**

# Overview

In this project, I trained my own diffusion model on MNIST.


















# Part A.1: Taking Photos and Recovering Homographies

To capture photos that blend well, I took them in a way such that the transforms between them are projective (a.k.a. perspective) by fixing the center of projection and rotating my camera while capturing photos.

A homography is a [projective transformation](https://www.sciencedirect.com/topics/engineering/homography) between two planes (a mapping between two planar projections of an image). In other words, homographies are simple image transformations that describe the relative motion between two images, when the camera (or the observed object) moves.

We can represent the transformation with a 3x3 matrix H, such that 

$$H \cdot p_{\text{original}} = p_{\text{transformed}}$$

$$\begin{bmatrix} h_1 & h_2 & h_3 \\ h_4 & h_5 & h_6 \\ h_7 & h_8 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} wx' \\ wy' \\ w \end{bmatrix}$$

We can solve for H by solving a system of equations with eight degrees of freedom, using the keypoints we generated using [this tool](https://cal-cs180.github.io/fa23/hw/proj3/tool.html). Extending this to our full set of points gives us:

$$\begin{bmatrix}
x_1 & y_1 & 1 & 0 & 0 & 0 & -x'_1x_1 & -x'_1y_1 \\
0 & 0 & 0 & x_1 & y_1 & 1 & -y'_1x_1 & -y'_1y_1 \\
x_2 & y_2 & 1 & 0 & 0 & 0 & -x'_2x_2 & -x'_2y_2 \\
0 & 0 & 0 & x_2 & y_2 & 1 & -y'_2x_2 & -y'_2y_2 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
x_n & y_n & 1 & 0 & 0 & 0 & -x'_nx_n & -x'_ny_n \\
0 & 0 & 0 & x_n & y_n & 1 & -y'_nx_n & -y'_ny_n \\
\end{bmatrix}
\begin{bmatrix}
h_1 \\
h_2 \\
h_3 \\
h_4 \\
h_5 \\
h_6 \\
h_7 \\
h_8 \\
\end{bmatrix}
=
\begin{bmatrix}
x'_1 \\
y'_1 \\
x'_2 \\
y'_2 \\
\vdots \\
x'_n \\
y'_n \\
\end{bmatrix}$$

We solve this with a least-squares to find the minimum norm solution, since with more than four points the system becomes overdetermined. While we can solve for an exact solution with just four points, we want more in order to make our homography recovery more robust to noise. Note that after appyling the homography matrix H, in order to recover $$x'$$ and $$y'$$ we need to divide by $$w$$.

# Part A.2: Image Warping and Rectification

I implemented a warping function that takes the source image and the homography matrix $$\mathbf{H}$$ corresponding to the transformation from the source to the target image. The warp function applies $$\mathbf{H}$$ to the corners of the source image, calculates dimensions of the output image, then draws a polygon mask using the transformed corners. For each point in the polygon, I reverse interpolate from the source image by applying $$\mathbf{H}^{-1}$$ to recover the corresponding point in the source image (this is vectorized by constructing matrices).

To test the function, I rectified several images. Rectification is performed using a homography to make a known rectangular object rectangular again, even if it does not visually appear rectangular. For each image, I selected four points defining the corners of a rectangular shape, calculated the homography to a set of coordinates forming a rectangle, and then warped the image. 

For source images, I drew on photos I took during my trip to New York City this summer. The first example is a vintage subway advertisement in the New York Transit Museum. 

<p align="center">
    <img src="./img/ad.jpg" alt="ad" width="30%"/>
    <img src="./img/ad_pts.jpg" alt="ad" width="30%"/>
    <img src="./img/ad_rectified.jpg" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Original, with points labeled, and rectified images.</i></p>
</p>

This museum is one of my favorites of all time, and is defintely worth the stop in Brooklyn for.

The second image comes from the ceiling of the [Rose Main Reading Room](https://www.newyorker.com/culture/cultural-comment/nypl-rose-reading-room-and-the-real-meaning-of-luxury-in-new-york-city) in the Stephen A. Schwarzman Building —— the New York Public Library's flagship location.

<p align="center">
    <img src="./img/nypl_full.jpg" alt="ad" width="30%"/>
    <img src="./img/nypl_pts.jpg" alt="ad" width="30%"/>
    <img src="./img/nypl_rectified.jpg" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Original, with points labeled, and rectified images.</i></p>
</p>

I spent an afternoon working here before going on a tour of the United Nations!

# Part A.3: Blending and Mosaicing

I implemented a blending function to combine warped images with the base image their homography matrix was calculated relative to. First, I calculated the size of the output image. Then, I placed each image on its correct position on a blank output image using information about how much the image shifted during the warping process — if I were to stack these two images, the overlapping features should line up (I confirmed this during implementation). Finally, I blend these two images together using weights calculated from distance transforms for the images to combine the low frequencies. I used a Laplacian stack-based or a distance transform tiebreaker-based blending method to combine the high frequencies (further elaboration below).

My first mosaic is of the third floor in Berkeley Way West. For this image, I warped the image of the left side of the field of view and blended it with the center view. Until I get a research position in BAIR, I'll be underneath the 8th floor :\)
<p align="center">
    <img src="./img/bwwleft.jpg" alt="ad" width="30%"/>
    <img src="./img/bwwlm_warped_im.jpg" alt="ad" width="30%"/>
    <img src="./img/bwwmid.jpg" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Original left BWW, warped left BWW, and original center BWW.</i></p>
</p>

<p align="center">
    <img src="./img/bww_left.jpg" alt="ad" width="50%"/>
    <p style="text-align: center;"><i>Resulting mosaic.</i></p>
</p> 


I dug up an old Canon [Powershot](https://www.dpreview.com/articles/6913931643/the-vertical-elph-remembering-canons-powershot-tx1-hybrid-camera) [TX1](https://global.canon/en/c-museum/product/dcc541.html) last week, and have been putting it to use! I took photos of a particularly busy corner of my room using my digicam, and blended the warped lower perspective onto the higher perspective.

<p align="center">
    <img src="./img/room1.jpg" alt="ad" width="30%"/>
    <img src="./img/room1_warped_im_new.jpg" alt="ad" width="30%"/>
    <img src="./img/room2.jpg" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Original lower room, warped lower room, and original upper room.</i></p>
</p>

<p align="center">
    <img src="./img/room12_new_highsigma.jpg" alt="ad" width="50%"/>
    <p style="text-align: center;"><i>Resulting mosaic.</i></p>
</p>


Here's the same scene, but shot on my iPhone.
<p align="center">
    <img src="./img/r1.jpg" alt="ad" width="30%"/>
    <img src="./img/r1_warped_im_new.jpg" alt="ad" width="30%"/>
    <img src="./img/r2.jpg" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Original lower room, warped lower room, and original upper room.</i></p>
</p>

<p align="center">
    <img src="./img/r12_new_highsigma.jpg" alt="ad" width="50%"/>
    <p style="text-align: center;"><i>Resulting mosaic.</i></p>
</p>

Personally, I prefer the "vibes" of the digicam photos —— what do you think?

My last mosaics come from Busan, South Korea. Gamcheon Culture Village is a beautiful site with a great view of the ocean, though it was swelteringly hot when I was there earlier this year. 

Here's the left side of the Gamcheon lookout warped and blended with the middle view:
<p align="center">
    <img src="./img/left.jpg" alt="ad" width="30%"/>
    <img src="./img/lm_warped_im_new.jpg" alt="ad" width="30%"/>
    <img src="./img/middle.jpg" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Original left Gamcheon, warped left Gamcheon, and original middle Gamcheon.</i></p>
</p>

<p align="center">
    <img src="./img/gamcheon_left_new.jpg" alt="ad" width="50%"/>
    <p style="text-align: center;"><i>Resulting mosaic.</i></p>
</p>

Here's the right side of the Gamcheon lookout warped and blended with the middle view:
<p align="center">
    <img src="./img/right.jpg" alt="ad" width="30%"/>
    <img src="./img/rm_warped_im_new.jpg" alt="ad" width="30%"/>
    <img src="./img/middle.jpg" alt="ad" width="30%"/>
    <p style="text-align: center;"><i>Original right Gamcheon, warped right Gamcheon, and original middle Gamcheon.</i></p>
</p>

<p align="center">
    <img src="./img/gamcheon_right_new.jpg" alt="ad" width="50%"/>
    <p style="text-align: center;"><i>Resulting mosaic.</i></p>
</p>

With several warped images relative to a fixed center image, I also implemented a function to automatically put together multi-image mosaics by tracking cumulative offsets and using the shifts from each warped image. In my blending function, I use distance transforms to create weights. I use a Gaussian filter to isolate the low frequencies of the images, which I combine with a weighted average. For the higher frequencies (the methodology to isolate these is the same as creating the Laplacian stack in [project 2](https://stevenfluo.github.io/filters-frequencies)). To create the final output image, I combined the low and high frequencies — just like collapsing the Laplacian stack.

I tried two approaches to blending the high frequencies: first, a weighted average just like for the low-frequencies:

<p align="center">
    <img src="./img/gamcheon_blended_laplace.jpg" alt="ad" width="90%"/>
    <p style="text-align: center;"><i>Resulting mosaic.</i></p>
</p>

Notice that because of the fine details in the image and slight misalignment between the two, you can spot blurry lines such as on the mural in the bottom middle of the photo.

My second approach used the distance transforms as a tiebreaker to select whether to use the warped image or the base image high frequencies at a particular point: 

<p align="center">
    <img src="./img/gamcheon_total.jpg" alt="ad" width="90%"/>
    <p style="text-align: center;"><i>Resulting mosaic.</i></p>
</p>

Notice that "scaly" artifacts appear (look at the mural again!), but these are also hard to notice unless you look very closely. 

Looking at these mosaics makes me feel like I'm back on vacation!


# **Part B: Feature Matching for Autostitching**

This part of the project is essentially a simplified reimplementation of the paper ["Multi-Image Matching using Multi-Scale Oriented Patches"](https://ieeexplore.ieee.org/document/1467310), with the addition of a random sample consensus method based on the techniques described in lecture.

# Part B.1: Harris Interest Point Detection

I followed Section 2 in the paper in order to implement corner detection for my images, with the help of provided starter code in `harris.py`. First, I grayscaled my image, then used `get_harris_corners` in to calculate the Harris score and corners. Note that in order to plot the Harris corners, I had to flip the coordinates so that I could plot them in the form (x, y). 

<p align="center">
    <img src="./img/room1_harrispoints.jpg" alt="ad" width="90%"/>
    <p style="text-align: center;"><i>Harris points overlaid on the lower room image.</i></p>
</p>

# Part B.2: Adaptive Non-Maximal Suppression 

Adaptive Non-Maximal Suppression (ANMS) is used to limit the number of interest points while maintaining a good spatial distribution over the entire image. To implement ANMS, I follow Section 3 and use the recommended `c_robust = 0.9` and keep only the top 500 points of interest (though this can be easily adjusted by modifying the optional parameters).

First, I index into the Harris scores to find the score of each corner, which I pass into the provided `dist2` function to calculate a matrix of pairwise corner distances. I then create a mask that determines if $$f(x_i) < c_{robust} * f(x_j)$$ by comparing the strength of each of the corners with a preset scaling factor. I apply the mask to the distances matrix, set zero elements to infinity, then find the minimum suppression radius and sort indices in ascending order based on suppression radii. The first 500 points are kept and returned as the ANMS points. 

<p align="center">
    <img src="./img/room2_harrispoints.jpg" alt="ad" width="40%"/>
    <img src="./img/room2_anmspoints.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Harris and ANMS points overlaid on the upper room image.</i></p>
</p>

<p align="center">
    <img src="./img/room2_labeled_anms.jpg" alt="ad" width="90%"/>
    <p style="text-align: center;"><i>Labeled ANMS points overlaid on the upper room image.</i></p>
</p>

# Part B.3: Feature Descriptor Extraction

Following Section 4, I extract axis-aligned 8x8 patches, which are downsampled from a larger 40x40 window then normlized and flattened.

<p align="center">
    <img src="./img/room2_features.jpg" alt="ad" width="90%"/>
    <p style="text-align: center;"><i>Twenty features from the upper room image.</i></p>
</p>

In this case, it seems that many of the top features lie on the piano keys!

# Part B.4: Feature Matching

In order to recreate correspondence points, we need to pair up features from each image. Following Section 5, I use the features and corresponding points from each image to calculate features' pairwise differences, which is then used to create a summed squared differences matrix. Each feature's row is sorted in ascending order, and the first two nearest neighbor distances for each feature are used to calculate its Lowe score. A mask is created by compare the Lowe score with the Lowe threshold, and a filtered matrix is returned where each row has two indices corresponding to the index of the matching features from the first and second images.

<p align="center">
    <img src="./img/room_matched.jpg" alt="ad" width="90%"/>
    <p style="text-align: center;"><i>First five matching features in the room images.</i></p>
</p>

<p align="center">
    <img src="./img/room1_matched.jpg" alt="ad" width="40%"/>
    <img src="./img/room2_matched.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Correspondences between lower and upper room images calculated from feature matching.</i></p>
</p>

# Part B.5: Random Sample Consensus 

Random Sample Consensus (RANSAC) is an iterative process for estimating a model from a dataset with outliers. For my implementation, I followed the four-point RANSAC method described in lecture, using 500 iterations and a threshold of 0.8.

I use the points identified from feature matching as the input points into RANSAC. For each iteration of the algorithm, I randomly select 4 points, calculate a homography matrix, then use this `H` to transform the full set of input source points. I then compute the number of inliers on this set of points, and create and save the mask if the number of inlier points is the maximum we've seen so far. The mask is produced by comparing the Euclidian distance between the transformed source points to the target points. RANSAC returns two sets of points produced by applying the mask created from the sample producing the maximum set of inliers —— these set of points produced by masking the two sets of input points are the points that produce the best homography, and become our correspondence points for warping and blending (like the ones we manually generated in part A).

<p align="center">
    <img src="./img/room1_ransac.jpg" alt="ad" width="40%"/>
    <img src="./img/room2_ransac.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Correspondences between lower and upper room images after RANSAC.</i></p>
</p>

# Part B.6: Creating Autostitched Mosaics

My autostitching performed very well — in fact, there are virtually no noticeable differences between the autostitched mosaics and the ones I created in Part A! Slight misalignment occurred in the room mosaic, but that was likely because of a slight perspective shift as well as the RANSAC-produced points not being as spatially distributed as my manual annotations, where I was careful to create many correspondence points on the corners of the pictures on the wall. Also, all of my autostitched mosaics were blended with the distance transforms approach. Using the weighted average method on the room picture may make these artifacts less visible.

## Gamcheon Lookout - Left Side

<p align="center">
    <img src="./img/gamcheonleft_anms.jpg" alt="ad" width="40%"/>
    <img src="./img/gamcheonmiddle_anms.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Labeled ANMS points overlaid on the left and middle Gamcheon images.</i></p>
</p>

<p align="center">
    <img src="./img/gamcheon_left_features.jpg" alt="ad" width="90%"/>
    <p style="text-align: center;"><i>Five features from the each image. Top row is left Gamcheon, bottom row is middle Gamcheon.</i></p>
</p>

<p align="center">
    <img src="./img/gamcheon_matchedfeats.jpg" alt="ad" width="90%"/>
    <p style="text-align: center;"><i>First five matching features in the left and middle Gamcheon images.</i></p>
</p>

<p align="center">
    <img src="./img/gleft_matched.jpg" alt="ad" width="40%"/>
    <img src="./img/gmid_matched.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Correspondences between left and middle Gamcheon images calculated from feature matching.</i></p>
</p>

<p align="center">
    <img src="./img/gleft_ransac.jpg" alt="ad" width="40%"/>
    <img src="./img/gmid_ransac.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Correspondences between lower and upper room images after RANSAC.</i></p>
</p>

<p align="center">
    <img src="./img/gamcheon_left.jpg" alt="ad" width="40%"/>
    <img src="./img/autostiched_gamcheon_left_dist.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Manually and autostitched mosaics.</i></p>
</p>

## Gamcheon Lookout - Right Side

<p align="center">
    <img src="./img/gamcheon_right.jpg" alt="ad" width="40%"/>
    <img src="./img/autostitched_gamcheon_right_dist.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Manually and autostitched mosaics.</i></p>
</p>

## Gamcheon Lookout - Full View

<p align="center">
    <img src="./img/gamcheon_total.jpg" alt="ad" width="40%"/>
    <img src="./img/autostitched_gamcheon_total_dist.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Manually and autostitched mosaics.</i></p>
</p>

## Room (Digicam)

<p align="center">
    <img src="./img/room12_new_highsigma.jpg" alt="ad" width="40%"/>
    <img src="./img/autostitched_room_autostitched.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Manually and autostitched mosaics.</i></p>
</p>

## Berkeley Way West

<p align="center">
    <img src="./img/bww_left.jpg" alt="ad" width="40%"/>
    <img src="./img/autostitched_bww.jpg" alt="ad" width="40%"/>
    <p style="text-align: center;"><i>Manually and autostitched mosaics.</i></p>
</p>

# Reflection

I had a great time learning about how we can use homography matrices to transform images in a way that they line up with each other! I have many images of lecture slides taken at an angle —— now, I can use my rectification methods to transform them into a more rectangular shape to view :)

I think the coolest thing I learned from this project was how we can mathematically identify corners in images, and then whittle down a big list of candidate points into just a handful of highly accurate correspondences using ANMS, feature matching, and RANSAC. Reimplementing methods described in a research paper was also good practice for me as I go further down the research path outside of my classes. Honestly, the hardest part of the project was wrapping my head around the constant shifts between image coordinates, homogenous coordinates, and our standard (x,y) coordinates — many of my long debugging sessions were resolved by making sure my coordinates were what I thought they were! Keeping track of the shifts to align the images for blending was also tricky.
