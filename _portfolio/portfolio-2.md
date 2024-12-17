---
title: "Are generative classifier more robust to Adversarial attacks ?"
excerpt: "This work presents an overview on the article Are Generative Classifiers More Robust to Adversarial Attacks ? (the link in the portfolio page). We implemented the authors experiment on MNIST, and applied the methods on the German Traffic Sign Recognition Benchmark dataset, under black-box adversarial attacks, and were unable to conclude on whether generative classifiers were more robust to adversarial attacks than discriminative classifiers. <br/><img src='/images/adversarial_image.jpg'>"
collection: portfolio-2
---

## Introduction

Generative classifiers have been proposed as a potentially more robust alternative to discriminative classifiers. Deep Bayes classifiers, an improvement on classical Naive Bayes models, use deep latent variable models (LVMs) trained via variational inference. We explore the robustness of generative classifiers under hand-crafted black-box adversarial attacks, on what would be a real world setting, training and testing on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, a domain where robustness is extremely important. Traffic sign recognition systems are vital for autonomous driving, but are particularly susceptible to real-world adversarial attacks, such as strategically placed stickers. The different factorizations of \\(p(x, z, y)\\) enable diverse robustness and detection capabilities, that we explore in our experiments.

## Deep Bayes

LVMs introduce unobserved latent variables $z$ to model the joint distribution $p(x, y)$ of inputs $x$ and labels $y$. The joint distribution is expressed as:
$$
\displaylines{
p(x, z, y) = p(z)p(y|z)p(x|z, y),
}
$$

Variational Autoencoders approximate $p(x|z, y)$ using neural networks. The training involves optimizing the variational lower bound:
$$
\displaylines{
\mathbb{E}_\mathcal{D}[\mathcal{L}_{VI}(x, y)] = \frac{1}{N} \sum_{n=1}^N \mathbb{E}_{q} \left[ \log \frac{p(x_n, z_n, y_n)}{q(z_n|x_n, y_n)} \right],
}
$$
After training, the generative classifiers predict the label $y^*$ for a given input $x^*$ by approximating Bayes' rule using importance sampling. The predicted class probability is computed as:
$$
\displaylines{
p(y^*|x^*) \approx \text{softmax}_{c=1}^C \left[ \log \frac{1}{K} \sum_{k=1}^K \frac{p(x^*, z_c^k, y_c)}{q(z_c^k | x^*, y_c)} \right],
}
$$

## The models

We evaluated seven models, four generative and three discriminative classifiers. Each model follows a distinct factorization of $p(x, z, y)$ (see Figure \ref{fig:models_schema}).
$$
\begin{aligned}
p(x, z, y) &= p(z)p(y|z)p(x|z, y) & \text{(GFZ)} \\
p(x, z, y) &= p_\mathcal{D}(y)p(z|y)p(x|z, y) & \text{(GFY)} \\
p(x, z, y) &= p(z)p(y|z)p(x|z) & \text{(GBZ)} \\
p(x, z, y) &= p(y)p(z|y)p(x|z) & \text{(GBY)} \\
p(x, z, y) &= p(x)p(z|x)p(y|z, x) & \text{(DFX)} \\
p(x, z, y) &= p(z)p(x|z)p(y|z, x) & \text{(DFZ)} \\
p(x, z, y) &= p(x)p(z|x)p(y|z) & \text{(DBX)}
\end{aligned}
$$

<a name="Figure1"></a>

![fig 1](https://francklaborde.github.io/portfolio/portfolio-2/fig/graphical_model_color.png)

*Figure 1: A visualisation of the graphical models.*

As shown in [Figure 1](#Figure1), the graphical models illustrate the **G**enerative or **D**iscriminative **F**ully connected and **B**ottleneck architectures, with the last character indicating the first node in the topological order of the graph.

## Detecting adversarial attacks with classifiers

Three detection methods are proposed.

-  **Marginal detection**: reject the input data that are far from the data manifold
$$
\displaylines{
    -\log p(x) > \delta
}$$
with $\delta = \bar{\mu}_{\mathcal{D}}+\alpha \bar{\sigma}_{\mathcal{D}}$ and $\bar{\mu}_{\mathcal{D}}= \mathbb{E}_{x\sim\mathcal{D}}[-\log p(x)]$, $\bar{\sigma}_{\mathcal{D}}=\sqrt{\mathbb{V}_{x\sim\mathcal{D}}[\log p(x)]}$.

-  **Logit detection**:reject the data that are far from the joint density $$\displaylines{-\log p(x, F(x))>\delta_{y}}$$
with $\delta_{y_c}=\bar{\mu}_c+\alpha\bar{\sigma}_c$, for each class $c=\{1,\dots,C\}$.

-  **Divergence detection**: reject inputs with over-confident and/or under-confident predictions. $$\displaylines{D[p_{c^*} \| p(\bm{x}^*)] > \bar{\mu}_{c^*} + \alpha \bar{\sigma}_{c^*}}$$.
with $c^*=arg \max p(x^*)$ and $p_{c^*}= \mathbb{E}_{(x, y_{c^*}) \in \mathcal{D}}[p(x)]$

## Adversarial attacks

An adversarial attack is called a white-box attack when the method has access to the model's weights to generate adversarial examples. It is called a black-box attack when the method has only access to the inputs and outputs of the model, but not its weights. We defined the following black-box method, that are illustrated in Figure \ref{fig:adv-attacks}.
- **Gaussian**: It modifies an image by adding Gaussian noise to each pixel.
$$
\displaylines{
    x_{\text{adv}} = x + \eta 
}
$$
where $\eta \sim \mathcal{N}(0, \varepsilon^2)$.
- **Sticker**: It overlays a patch (a sticker) over an image, near its center. The sticker's size $\varepsilon$ represents a fraction of the image area. The sticker's color is randomly picked for each test image between different flashy colors: bright yellow, neon green, neon pink, bright cyan, bright orange.
Each method is used to attack each of the 7 models, with varying $\varepsilon$.

![Black box attack on GTSRB](https://francklaborde.github.io/portfolio/portfolio-2/fig/attacks_bbox_gtsrb_E.png)
*Figure 2.Example of the black-box adversarial attacks with different $\varepsilon$ values.*

## Results: Accuracy vs. Attack Strength with Detection

Our results are summarized below :


![Gaussian attack on GTSRB results](https://francklaborde.github.io/portfolio/portfolio-2/fig/Gaussian_gtsrb_combined.png)

![Sticker attack on GTSRB results](https://francklaborde.github.io/portfolio/portfolio-2/fig/Sticker_gtsrb_combined.png)

*Figure 3 Victim accuracy on left against **black-box attacks** and detection with the three methods on right on GTSRB. The higher the better.*

## Discussion

- 4 models robust to black-box adversarial attacks: GFZ, GFY, DBX, and DFZ, though not the most accurate ones.
- 3 models have a high accuracy, but show little robustness to black-box attacks
- For the detection, the KL-detection rate is surprisingly high, may be induced by a divergence that is not representative of the distribution as there are 43 classes. 
- Overall, generative classifiers do not seem to show more robustness to black-box adversarial attacks on harder problems with real-life data.

All the code for the experiments can be be found [here](https://github.com/francklaborde/DeepBayesTorch).

## Reference

[Yingzhen Li, John Bradshaw, and Yash Sharma.
Are generative classifiers more robust to adversarial attacks ? 2019](https://arxiv.org/pdf/1802.06552)