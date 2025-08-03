According to proximal policy optimization, the RL process is updated by the surrogate optimization. It means it does not optimize on the true thing what we want, it optimize on something else but can link to the true target in some sense.

So, I am asking a question about is there any other optimization task in machine learning is also surrogate optimization, and why we want to do this surrogate optimization rather than the true one.

Of course. Here is a revised breakdown that includes Proximal Policy Optimization (PPO), detailing the mathematical insight and derivation for each surrogate function.

The central theme is replacing a mathematically problematic objective with a well-behaved, computable proxy that guides the optimization.

***

### Classification: Creating a Convex Upper Bound

* **The Problem**: The true 0-1 loss, $L_{0-1} = \mathbf{1}_{y \cdot f(x) \leq 0}$, has a zero gradient almost everywhere, making it impossible to optimize with gradient descent.

* **Derivation Insight**: If we can't optimize the function, we can find a simpler function that's always "above" it. By pushing down on this **convex upper bound**, we also push down the true loss. This new function must be convex and differentiable to provide a useful gradient.

* **The Mathematical Derivation (Hinge Loss)**:
    1.  We want a surrogate loss $L_{surr}$ such that $L_{surr}(y, f(x)) \geq L_{0-1}(y, f(x))$ for all inputs.
    2.  The **Hinge Loss** is constructed to create a "margin" of safety, penalizing predictions that are too close to the boundary:
        $$L_{Hinge} = \max(0, 1 - y \cdot f(x))$$
    3.  This is a valid upper bound because if a classification is wrong ($y \cdot f(x) \leq 0$), then $L_{Hinge} \geq 1$, which is $\geq$ the 0-1 loss of 1. If it's correct, both losses are $\geq 0$.
    4.  The result, $L_{Hinge}$, is a continuous, convex function with a well-defined gradient, making it an excellent surrogate for optimization.

***

### Variational Autoencoders (VAEs): Finding a Tractable Lower Bound

* **The Problem**: The true objective, the log-likelihood $\log p(x)$, is **intractable** because it contains an unsolvable integral over the high-dimensional latent space: $\log \int p(x,z) dz$.

* **Derivation Insight**: We find a **lower bound** on this intractable value that *is* computable. By maximizing this lower bound, we guarantee that the true objective is also pushed upward. The key mathematical tool to create this bound is **Jensen's Inequality**.

* **The Mathematical Derivation (ELBO)**:
    1.  Start with $\log p(x)$ and introduce an arbitrary distribution $q(z|x)$ (this is like multiplying by 1):
        $$\log p(x) = \log \int q(z|x) \frac{p(x,z)}{q(z|x)} dz = \log \mathbb{E}_{q(z|x)} \left[ \frac{p(x,z)}{q(z|x)} \right]$$
    2.  Apply **Jensen's Inequality**: Since logarithm is a concave function, $\log(\mathbb{E}[X]) \geq \mathbb{E}[\log(X)]$. This gives us:
        $$\log p(x) \geq \mathbb{E}_{q(z|x)} \left[ \log \frac{p(x,z)}{q(z|x)} \right]$$
    3.  The right-hand side is our tractable surrogate: the **Evidence Lower Bound (ELBO)**. We can rewrite it in its more common, computable form:
        $$\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))$$
    4.  This expression can be estimated with samples and optimized via gradient ascent.

***

### Proximal Policy Optimization (PPO): Creating a Clipped, Pessimistic Bound

* **The Problem**: The true objective in Reinforcement Learning is to maximize expected reward. Vanilla Policy Gradient methods do this but are unstable. A single large policy update, based on data from the old policy, can catastrophically collapse performance.

* **Derivation Insight**: To prevent destructive updates, we must constrain how much the policy can change at once. The insight of PPO is to achieve this constraint not with a complex calculation, but with a simple **clipping** mechanism on the objective function itself. This creates a pessimistic but safe surrogate objective.

* **The Mathematical Derivation (PPO-Clip)**:
    1.  The standard policy gradient objective is to increase the probability of actions that have a high **Advantage** ($A_t$), which measures how much better an action was than the average. Let $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ be the probability ratio between the new and old policies. The objective is to maximize:
        $$L^{PG}(\theta) = \mathbb{E}_t [r_t(\theta) A_t]$$
    2.  To prevent $r_t(\theta)$ from becoming too large or small, PPO clips it. We define a clipped version of the objective using a hyperparameter $\epsilon$ (e.g., 0.2):
        $$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta)A_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t) \right]$$
    3.  This `min` function creates a pessimistic bound.
        * If $A_t > 0$ (a good action), the update is capped once the ratio $r_t$ tries to exceed $1+\epsilon$. This prevents over-optimism.
        * If $A_t < 0$ (a bad action), the update is capped once $r_t$ tries to go below $1-\epsilon$. This prevents over-correction.
    4.  This clipped objective is a simple, first-order surrogate that effectively keeps the policy update within a safe "trust region," leading to much more stable and reliable training.

***

### Generative Adversarial Networks (GANs): Reframing Divergence as a Game

* **The Problem**: The true objective is to minimize the statistical distance (e.g., JSD) between the real data distribution $P_{data}$ and the generator's $P_g$. This is **intractable** because the formula for $P_{data}$ is unknown.

* **Derivation Insight**: Instead of computing the distance, we re-frame the problem as an **adversarial game**. A theorem shows that the JSD is maximized by the optimal classifier between two distributions. We can therefore create a game where a "discriminator" acts as this classifier, and its performance provides the learning signal.

* **The Mathematical Derivation (Minimax Objective)**:
    1.  The GAN value function is:
        $$V(G, D) = \mathbb{E}_{x \sim P_{data}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$
    2.  For a fixed generator $G$, the theoretically optimal discriminator is $D^*(x) = \frac{P_{data}(x)}{P_{data}(x) + P_g(x)}$.
    3.  If we plug this optimal $D^*(x)$ back into the value function, it can be shown that the result is directly proportional to the JSD:
        $$V(G, D^*) = 2 \cdot JSD(P_{data} || P_g) - 2 \log 2$$
    4.  This proves that the `min_G max_D V(G,D)` game is a **surrogate for minimizing the JSD**. The value function $V(G,D)$ is a tractable objective we can estimate with samples, turning an impossible calculation into a practical competition.
