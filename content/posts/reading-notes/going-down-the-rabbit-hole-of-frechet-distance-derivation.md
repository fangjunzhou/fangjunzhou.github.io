+++
title = "Going Down the Rabbit Hole of Frechet Distance Derivation"
date = 2026-11-21T14:38:24-08:00
draft = false
author = "Fangjun Zhou"
summary = ""
readTime = true
math = true
+++

If you've read papers on generative models, you've probably seen Fréchet Inception Distance (FID) in a table of results for generated images. I went back to FID recently because I wanted to understand why we'd evaluate a model this way and work through the math myself.

## Why are generative models hard to evaluate?

Say we're training a model to generate cat photos. The model defines a distribution $p_{\theta}$ that we want to be close to the distribution of real cat photos. To get a photo out of it, we draw a sample $x \sim p_{\theta}$.

How do we check if the model has learned that distribution? We could take a test set of real cat photos $x_1, \ldots, x_N$ and compute their average log-likelihood under the model, $\frac{1}{N}\sum_{i=1}^{N}\log p_{\theta}(x_i)$. By this measure, a higher average means a better fit to the test set.

{{< post-video src="media/reading-notes/frechet-distance/average-log-likelihood.mp4" poster="media/reading-notes/frechet-distance/average-log-likelihood-poster.png" label="Animation comparing the average log-likelihood of two generative models" caption="The true distribution is a mixture of two Gaussians. Both models score the same four test points in yellow, and Model 1 has the higher average log-likelihood." >}}

> A lot of blog posts and tutorials use pictures like this for the "distribution of natural images." I tend to picture natural images (and other kinds of structured data) as small islands in a very high-dimensional space. Try making an image by picking each pixel's color independently at random. You'll almost certainly get a screen full of noise. Real photos take up a tiny part of all the images you could make this way.

{{< post-video src="media/reading-notes/frechet-distance/sparse-gaussian-mixture.mp4" poster="media/reading-notes/frechet-distance/sparse-gaussian-mixture-poster.png" label="Animation transitioning from a 2D heatmap to five Gaussian blobs with samples in 3D" caption="The dots are draws from the mixture. If you picked a point uniformly from the whole plotted space, you'd usually land in the gaps between the blobs." >}}

### The curse of dimensionality

To generate an image $x$ with a **latent-variable model**, we draw $z$ from a simple distribution (say, a Gaussian), then draw $x$ from $p_{\theta}(x \vert z)$.

> A denoising diffusion probabilistic model (DDPM), for example, starts with a draw from a standard Gaussian. We use the learned reverse process to turn that noise into an image one step at a time.

What if we already have a test image and want to know its likelihood $p_{\theta}(x)$? We have to integrate over all possible values of $z$. For each value of $z$, we weight the conditional likelihood $p_{\theta}(x \vert z)$ by the latent density $p(z)$.

$$
p_{\theta}(x) = \int p_{\theta}(x \vert z) p(z) \mathrm{d}z
$$

Suppose each $z$ gives us one predicted image. We could draw a bunch of values of $z$ and count how often we get back the exact test image $x_i$. If the generated images follow a continuous distribution, the chance of an exact match is zero.

We can put a narrow Gaussian around each prediction so a near miss still contributes. For a given draw of $z$, the density of that Gaussian at $x_i$ gives us $p_{\theta}(x_i \vert z)$.

Now we can draw values of $z$ from $p(z)$ and average those densities to estimate the integral. Predictions near $x_i$ get much more weight. If we rarely draw a $z$ that puts us near $x_i$, we still need a lot of samples for the estimate to converge.

{{< post-video src="media/reading-notes/frechet-distance/latent-variable-likelihood.mp4" poster="media/reading-notes/frechet-distance/latent-variable-likelihood-poster.png" label="Animation showing how a latent-variable model estimates log-likelihood by sampling the prior" caption="The generator maps draws from a standard Gaussian into five sparse clusters. For each of four test points, we average the conditional densities from 48 latent draws and then take the log." >}}

> This reminds me of a problem in path tracing. Say we're shading a diffuse surface lit by a single point light. If we bounce a ray in a random direction, it has zero chance of hitting the light, so the surface comes out black. Give the light a small radius $\epsilon$, and a ray can hit it. Those hits are rare, though, so the render is noisy until we take a lot of samples. With direct light sampling, we pick a point on the light and shoot a shadow ray from the surface to see if anything blocks it.

{{< figure-row >}}
{{< figure src="media/reading-notes/frechet-distance/point-light-random.png" alt="A completely black render" width="504" height="360" caption="Random bounce directions miss the point light, leaving the scene black." >}}

{{< figure src="media/reading-notes/frechet-distance/tiny-sphere-random.png" alt="Bright speckles loosely outline two spheres and a checkerboard floor" width="504" height="360" caption="Replacing the point light with a small sphere makes random hits possible. Those hits are rare, so the estimate takes many samples to converge." >}}

{{< figure src="media/reading-notes/frechet-distance/point-light-direct.png" alt="Two smoothly lit spheres cast sharp shadows on a checkerboard floor" width="504" height="360" caption="Aiming a shadow ray directly at the point light gives a clean render with sharp shadows." >}}
{{< /figure-row >}}

> I tend to think of direct light sampling as an extreme case of importance sampling. With a point light, we already know where to aim the ray. For the likelihood integral, we can draw $z$ from a distribution that puts more weight on values that make the test image likely. Then we reweight the draws to account for how we picked them. If we pick a good sampling distribution, we can estimate $p_{\theta}(x_i)$ with fewer samples before taking the log.
>
> The FID paper points to **annealed importance sampling** as one way to estimate likelihoods. [*GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, Section 3](https://proceedings.neurips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf#page=5)

### Measuring the similarity between two distributions

If getting the likelihood of each test image takes this many samples, what else could we measure? With FID, we compare the distribution of generated images with the distribution of real images in the test set.

1. Start with a test set of real images.
2. Generate another set of images with the model.
3. Run both sets through the same image encoder. Heusel et al. used a pretrained Inception-v3 and took 2,048 features per image from its final pooling layer (`pool_3`). [Original FID paper](https://proceedings.neurips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf#page=6)
4. Compute the mean vector and covariance matrix of the features in each set. That gives us two multivariate Gaussians, one for real images and one for generated images.
5. Use the **Fréchet distance** to compare the two Gaussians. FID is the squared distance, so the closer the fits are, the lower the score.

> Fitting a Gaussian to each set means we're treating the encoded features as **Gaussian distributed**. Say the features from the test set and the generated set have the same mean and covariance. The two distributions can look quite different and still give us the same Gaussian fit.

{{< figure src="media/reading-notes/frechet-distance/matching-gaussian-moments.png" alt="A single tilted Gaussian on the left and four separate peaks on the right. Both panels have the same mean and covariance ellipse." width="1920" height="1080" caption="The yellow dots mark the shared mean, and the dashed ellipses show the shared covariance. The Gaussian fits are identical, so FID is zero in this example." >}}

## Frechet distance

In step 3 of the previous section, the encoder gives us a feature vector in $\mathbb{R}^d$ for each image. Heusel et al. get 2,048 features per image from the final pooling layer of Inception-v3, so $d = 2048$ in their setup.

I'll use $\mu_r$ and $\Sigma_r$ for the mean and covariance of the features from the **real test set**, and $\mu_g$ and $\Sigma_g$ for the **generated features**. The means are $d$-dimensional vectors and the covariance matrices are $d \times d$.

A Gaussian is completely determined by where it's centered and how it's spread out. We describe that with the mean and covariance (the mean and variance in 1D).

So how close can two Gaussians get? If their means and covariances match, they're the same distribution and the Fréchet distance is zero. Change the mean or covariance of one of them, and the distance becomes positive.

The squared Fréchet distance between the two fitted Gaussians is

$$
d_{\mathrm{F}}^2 =
\underbrace{\|\mu_r-\mu_g\|_2^2}_{\text{difference in average features}} +
\underbrace{\operatorname{tr}\!\left(
\Sigma_r+\Sigma_g
-2\left(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}\right)^{1/2}
\right)}_{\text{difference in feature spread and correlations}}.
$$

We'll work through where those two terms come from, step by step.

### Comparing two 1D Gaussian distributions

Let's start with $d = 1$. Each image gives us a single feature value, so our Gaussian fits are $\mathcal{N}(\mu_r, \sigma_r^2)$ and $\mathcal{N}(\mu_g, \sigma_g^2)$. Here $\sigma_r$ and $\sigma_g$ are the standard deviations.

Say the two curves have the same width. To line them up, we just slide one over by the difference in their means. The squared shift is $(\mu_r - \mu_g)^2$.

To see the difference in spread, line up the centers. For each curve, make a mark on the horizontal axis one standard deviation to the right of the center. The squared gap between those marks is $(\sigma_r - \sigma_g)^2$.

Put the two terms together and we get the squared Fréchet distance in 1D.

$$
d_{\mathrm{F}}^2 = (\mu_r - \mu_g)^2 + (\sigma_r - \sigma_g)^2
$$

Things get trickier in higher dimensions. The means are vectors now, so we can measure the squared shift with $\lVert \mu_r - \mu_g \rVert_2^2$. How do we compare the two covariance matrices? Let's stay in 1D a little longer and see where the spread term $(\sigma_r - \sigma_g)^2$ comes from.

The term $(\mu_r - \mu_g)^2$ already accounts for the shift between their centers, so we can subtract each Gaussian's mean and compare their spread with both centered at zero. Let $X \sim \mathcal{N}(0, \sigma_r^2)$ be a draw from the centered real distribution and $Y \sim \mathcal{N}(0, \sigma_g^2)$ a draw from the centered generated distribution. What's the average squared gap between the two draws?

$$
\begin{aligned}
\mathbb{E}[(X-Y)^2]
&= \mathbb{E}[X^2 - 2XY + Y^2] \\
&= \mathbb{E}[X^2] + \mathbb{E}[Y^2] - 2\mathbb{E}[XY] \\
&= \sigma_r^2 + \sigma_g^2 - 2\mathbb{E}[XY].
\end{aligned}
$$

Since both means are zero, $\mathbb{E}[X^2]$ and $\mathbb{E}[Y^2]$ are just the variances. The only term that depends on how we pair the draws is $\mathbb{E}[XY]$. Say the standard deviations are $2$ and $1$, so $X \sim \mathcal{N}(0, 4)$ and $Y \sim \mathcal{N}(0, 1)$. We can draw the pair a few different ways.

1. Draw $X$ and $Y$ independently. Then $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y] = 0$, so the average squared gap is $4 + 1 = 5$.
2. Draw one $Z \sim \mathcal{N}(0, 1)$ and set $X = 2Z$ and $Y = Z$. The two draws land at the same percentile in their respective distributions. Now $\mathbb{E}[(X-Y)^2] = \mathbb{E}[(2Z-Z)^2] = \mathbb{E}[Z^2] = 1$.
3. Draw one $Z \sim \mathcal{N}(0, 1)$ and set $X = 2Z$ and $Y = -Z$. Now a point to the right of one mean pairs with a point to the left of the other. We get $\mathbb{E}[(X-Y)^2] = \mathbb{E}[(2Z+Z)^2] = \mathbb{E}[(3Z)^2] = 9$.

{{< post-video src="media/reading-notes/frechet-distance/gaussian-pairings.mp4" poster="media/reading-notes/frechet-distance/gaussian-pairings-poster.png" label="Monte Carlo comparison of independent, shared, and sign-flipped pairings of two Gaussians" caption="Each run averages the squared gaps from 16,384 pairs. The red dashed line shows the exact expectation for that pairing." >}}

Which pairing should we use? Give both Gaussians a standard deviation of $2$, so they both follow $\mathcal{N}(0, 4)$. Independent draws give an average squared gap of $8$, and pairing $X=2Z$ with $Y=-2Z$ gives $16$. With $X=Y=2Z$, every gap is zero. A squared gap is always nonnegative, so its average can be zero only if $X=Y$ with probability one.

{{< post-video src="media/reading-notes/frechet-distance/identical-gaussian-pairings.mp4" poster="media/reading-notes/frechet-distance/identical-gaussian-pairings-poster.png" label="Monte Carlo comparison of three pairings for two identical Gaussians, with zero squared gap for shared draws" caption="The dashed green curve lies on top of the blue one. In the shared-draw run, both values land on the same point in every pair." >}}

So we look for the pairing that gives the smallest $\mathbb{E}[(X-Y)^2]$ while keeping the two distributions fixed. The variances are already set, so minimizing the gap comes down to maximizing $\mathbb{E}[XY]$. By Cauchy-Schwarz, that term can be at most $\sqrt{\mathbb{E}[X^2]\mathbb{E}[Y^2]} = \sigma_r\sigma_g$.

We can reach that bound by using the same standard-normal draw for both points. Draw $Z \sim \mathcal{N}(0, 1)$ and set $X = \sigma_r Z$ and $Y = \sigma_g Z$. Since $\mathbb{E}[Z^2] = 1$, this gives us $\mathbb{E}[XY] = \sigma_r\sigma_g$. Plug that in and we get the smallest possible average squared gap.

$$
\sigma_r^2 + \sigma_g^2 - 2\sigma_r\sigma_g = (\sigma_r - \sigma_g)^2.
$$

Adding the mean term back in gives the squared Fréchet distance in 1D.

$$
d_{\mathrm{F}}^2 = (\mu_r - \mu_g)^2 + (\sigma_r - \sigma_g)^2
$$

{{< post-video src="media/reading-notes/frechet-distance/frechet-distance-breakdown.mp4" poster="media/reading-notes/frechet-distance/frechet-distance-breakdown-poster.png" label="The squared Fréchet distance and its mean and spread terms as two 1D Gaussians change" caption="The yellow segment spans the two means. The red segment spans the one-standard-deviation marks on the centered plot." >}}

### Moving to higher dimension

WIP

## References

- Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. NeurIPS 2017. [*GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html).
- Jonathan Ho, Ajay N. Jain, and Pieter Abbeel. NeurIPS 2020. [*Denoising Diffusion Probabilistic Models*](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html).
- Matt Pharr, Wenzel Jakob, and Greg Humphreys. MIT Press, 2023. [*Physically Based Rendering: From Theory to Implementation*, 4th ed., Section 13.1](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation).
- Yuhuai Wu, Yuri Burda, Ruslan Salakhutdinov, and Roger Grosse. ICLR 2017. [*On the Quantitative Analysis of Decoder-Based Generative Models*](https://arxiv.org/abs/1611.04273).
