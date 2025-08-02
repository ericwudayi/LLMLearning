您的要求很棒，這將讓整個學習過程的數學邏輯變得更加清晰。我們將一步步地展示 DQN 和策略梯度是如何從它們的目標（最小化 Loss 或最大化 Reward）推導出最終的梯度更新步驟。

---

### DQN 的數學流程：從 Loss 到 Gradient

DQN 的核心是**最小化**一個損失函數，來讓它的 Q-network 能夠準確預測 Q 值。

#### Step 1: 定義損失函數 (Loss Function)

我們的目標是讓預測值 $Q(s, a; \theta)$ 逼近目標值 $y_t$。我們使用均方誤差 (MSE) 作為損失函數：

$$L(\theta) = \frac{1}{N} \sum_{i=1}^{N} (y_i - Q(s_i, a_i; \theta))^2$$

其中，**目標值** $y_i$ 來自於 Bellman 方程，並使用**固定權重**的目標網路來計算：

$$y_i = r_i + \gamma \max_{a'} Q(s'_i, a'; \theta^-)$$

#### Step 2: 求損失函數對權重的梯度

為了最小化這個損失函數，我們需要對它求導，得到它相對於網路權重 $\theta$ 的梯度 $\nabla_{\theta} L(\theta)$。

$$\nabla_{\theta} L(\theta) = \nabla_{\theta} \left[ \frac{1}{N} \sum_{i=1}^{N} (y_i - Q(s_i, a_i; \theta))^2 \right]$$

根據鏈式法則 (Chain Rule)，我們可以將梯度帶入求和符號和平方項：

$$\nabla_{\theta} L(\theta) = \frac{1}{N} \sum_{i=1}^{N} 2(y_i - Q(s_i, a_i; \theta)) \cdot \nabla_{\theta}(y_i - Q(s_i, a_i; \theta))$$

由於 $y_i$ 是由目標網路 $\theta^-$ 計算的，與當前優化的權重 $\theta$ 無關，因此 $\nabla_{\theta} y_i = 0$。所以，公式可以簡化為：

$$\nabla_{\theta} L(\theta) = - \frac{2}{N} \sum_{i=1}^{N} (y_i - Q(s_i, a_i; \theta)) \cdot \nabla_{\theta} Q(s_i, a_i; \theta)$$

#### Step 3: 梯度下降更新權重

最後，我們使用**梯度下降法**來更新權重。新的權重 $\theta_{new}$ 將沿著梯度的反方向移動，以減小損失。

$$\theta_{new} \leftarrow \theta_{old} - \alpha \nabla_{\theta} L(\theta)$$

其中 $\alpha$ 是學習率。這個步驟會讓 $Q(s, a; \theta)$ 的預測值更接近目標值 $y$。

---

### 策略梯度 (Policy Gradient) 的數學流程：從 Reward 到 Gradient

策略梯度的核心是**最大化**一個預期總獎勵函數 $J(\theta)$。

#### Step 1: 定義優化目標 (Objective)

我們的目標是最大化**預期總獎勵** $J(\theta)$。這個預期值是所有可能的軌跡 $\tau$ 的總獎勵 $R(\tau)$ 的期望。

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)]$$

#### Step 2: 求優化目標對權重的梯度

為了最大化 $J(\theta)$，我們需要求其相對於策略網路權重 $\theta$ 的梯度 $\nabla_{\theta} J(\theta)$。根據**策略梯度定理**，這個梯度可以表示為：

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \left( \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right) \cdot R(\tau) \right]$$

在實際應用中，我們不能計算期望值，而是通過**採樣 (sampling)** 來估計它。我們讓智能體執行一個回合，得到一個軌跡 $\tau$。這個軌跡的梯度估計值為：

$$\hat{g} = \left( \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right) \cdot R(\tau)$$

這裡， $\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$ 就是**對數策略的梯度**，它指明了如何調整權重來增加動作 $a_t$ 的機率。$R(\tau)$ 則是這個動作所帶來的**總獎勵**。

#### Step 3: 梯度上升更新權重

最後，我們使用**梯度上升法**來更新權重。新的權重 $\theta_{new}$ 將沿著梯度的正方向移動，以增加總獎勵。

$$\theta_{new} \leftarrow \theta_{old} + \alpha \hat{g}$$

其中 $\alpha$ 是學習率。這個步驟會讓那些帶來高獎勵的動作，其在策略網路中的機率變得更高。
