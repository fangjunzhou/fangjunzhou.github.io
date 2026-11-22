+++
title = "Going Down the Rabbit Hole of Frechet Distance Derivation"
date = 2026-11-21T14:38:24-08:00
draft = false
author = "Fangjun Zhou"
summary = ""
readTime = true
math = true
+++

If you've read papers on image generation models, you've probably seen Fréchet Inception Distance (FID) used to evaluate generated images. I've been revisiting FID lately, so I decided to write up what I learned about why we use it and how it works.

## Why are generative models hard to evaluate?

Say we're training a model to generate cat photos. The model defines a distribution $p_{\theta}$ over possible outputs, and we want $p_{\theta}$ to approximate the distribution of real cat photos. To generate a photo, we draw a sample $x \sim p_{\theta}$.

Evaluating the model sounds easy enough. Take a test set of real cat photos $x_1, \ldots, x_N$ and compute their average log-likelihood under the model, $\frac{1}{N}\sum_{i=1}^{N}\log p_{\theta}(x_i)$. A higher average means the model fits the test set better by this measure.

### The curse of dimensionality

To generate an image $x$ with a **latent-variable model**, we first draw $z$ from a simpler distribution like a Gaussian. Then we sample $x$ from $p_{\theta}(x \vert z)$. To get the likelihood of a particular test image, we have to integrate over the latent variables. That can be expensive.

> Take a denoising diffusion probabilistic model (DDPM) as an example. We draw initial noise from a standard Gaussian and generate an image one step at a time using the learned reverse process.

To get $p_{\theta}(x)$, we average the conditional likelihood $p_{\theta}(x \vert z)$ over all possible values of $z$, weighted by the latent density $p(z)$.

$$
p_{\theta}(x) = \int p_{\theta}(x \vert z) p(z) \mathrm{d}z
$$

Suppose each $z$ gives us one predicted image. We could draw a lot of values of $z$ and count how often the prediction matches the test image $x_i$ exactly. If the generated images follow a continuous distribution, the chance of an exact match is zero.

We can put a narrow Gaussian around each prediction so a near match can still contribute. The Gaussian's density at $x_i$ gives us $p_{\theta}(x_i \vert z)$ for that draw.

We can now estimate the integral by drawing values of $z$ from $p(z)$ and averaging those densities. A narrow Gaussian gives much more weight to predictions close to $x_i$. If those predictions are rare, it can still take a lot of samples for the estimate to converge.

> There's a similar problem in path tracing. Say we have a diffuse surface and a single point light. If we pick a bounce direction at random, the chance of hitting the light is zero and the surface stays black. Give the light a small radius $\epsilon$ and some rays can hit it. At low sample counts, those rare hits show up as noise. With direct light sampling, we pick a point on the light and shoot a shadow ray from the surface to see if it's visible.

{{< figure-row >}}
{{< figure src="images/reading-notes/frechet-distance/point-light-random.png" alt="A completely black render" width="504" height="360" caption="Random bounce directions miss the point light, leaving the scene black." >}}

{{< figure src="images/reading-notes/frechet-distance/tiny-sphere-random.png" alt="Bright speckles loosely outline two spheres and a checkerboard floor" width="504" height="360" caption="Replacing the point light with a small sphere makes random hits possible. Those hits are rare, so the estimate takes many samples to converge." >}}

{{< figure src="images/reading-notes/frechet-distance/point-light-direct.png" alt="Two smoothly lit spheres cast sharp shadows on a checkerboard floor" width="504" height="360" caption="Aiming a shadow ray directly at the point light gives a clean render with sharp shadows." >}}
{{< /figure-row >}}

> I tend to think of direct light sampling as an extreme case of importance sampling. For a single point light, we know where to aim the ray. We can do something similar for the likelihood integral. We draw $z$ from a distribution that favors values that make the test image likely, then reweight the draws to account for how we picked them. With a good choice of sampling distribution, we can estimate $p_{\theta}(x_i)$ with fewer samples and then take the log.
>
> Heusel et al. point to **annealed importance sampling** as one way to estimate likelihoods. [*GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, Section 3](https://proceedings.neurips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf#page=5)

### Measuring the similarity between two distributions

So if estimating the likelihood of each test image takes this much work, how else can we evaluate the model? The idea behind FID is to compare the distribution of images our model generates with the distribution of the real images in the test set.

## References

- Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. NeurIPS 2017. [*GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html).
- Jonathan Ho, Ajay N. Jain, and Pieter Abbeel. NeurIPS 2020. [*Denoising Diffusion Probabilistic Models*](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html).
- Matt Pharr, Wenzel Jakob, and Greg Humphreys. MIT Press, 2023. [*Physically Based Rendering: From Theory to Implementation*, 4th ed., Section 13.1](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation).
- Yuhuai Wu, Yuri Burda, Ruslan Salakhutdinov, and Roger Grosse. ICLR 2017. [*On the Quantitative Analysis of Decoder-Based Generative Models*](https://arxiv.org/abs/1611.04273).
