So Lets start to study Qlearning.

I already know Q-learning tried so solve the sequential decision making prblom with model-free setting. For the reason of dimension cursing, we cannot build up a pretty large table to record all the Q-value of (s,a) pair.

Given above constraint, we tried to use deep nerual network to modeling the Q(s,a). But this should have a lot of problem, can you point out the first paper of this idea? What problem they were facing when adapt dnn, and how they solve this.

# My summarization

The target of Q-learning is: $y_{t} = r_{t} + max_{a} (\gamma * Q(a, s_{t+1} | \theta))$ , where $\theta$ is parameter of DQN.Following the Bellman function.

Loss is: $L = |y_{t} - Q(s_t, a)|$

## Problem:

1.If we train the model using each play, this sampling is not i.i.d -> would lead bad optimization. They use experience replay that do the samping

2. Model is learn by-itself, it should be very unstable, so the weights are only updated every 1000 steps.

3. Only few data point have r_{t}

## Details of Learning
好的，這兩個問題非常核心，我們來拆解一下。

1. 經驗回放緩衝區的容量 (Experience Replay Buffer Size)
緩衝區的大小是一個需要根據任務和計算資源調整的超參數 (hyperparameter)。在最初的 DQN 論文中，作者設定了一個非常大的緩衝區，容量為 100 萬 (1,000,000) 個經驗。

這個巨大的容量是為了：

長時間的記憶：讓模型能記住很早之前的經驗，特別是那些稀有的、高價值的經驗。

確保訓練數據的去相關性：緩衝區越大，隨機取樣時選到相鄰經驗的機率就越低，從而更好地打破數據的時間相關性。

2. 每一次 Batch Update 的過程
你所描述的直覺是正確的，DQN 的學習過程是一個循環。我們來詳細說明一個完整的互動-學習週期是如何運作的：

假設經驗回放緩衝區已經存有足夠的經驗 (例如，已經預先填滿到一定比例)。

Step 1: 智能體與環境互動

觀察狀態 (s 
t
​
 )：智能體從環境中獲得當前的狀態。

選擇動作 (a 
t
​
 )：根據當前的 Q-network，智能體使用 ϵ-greedy 策略來選擇一個動作。

以 ϵ 的機率，隨機選擇一個動作來進行探索 (exploration)。

以 1−ϵ 的機率，選擇 Q-network 預測 Q 值最高的動作來進行利用 (exploitation)。

執行動作：智能體執行動作 a 
t
​
 ，環境進入下一個狀態 s 
t+1
​
 ，並返回一個即時獎勵 r 
t
​
 。

Step 2: 儲存經驗

將這一步完整的經驗元組 (s 
t
​
 ,a 
t
​
 ,r 
t
​
 ,s 
t+1
​
 ) 儲存到經驗回放緩衝區中。

如果緩衝區滿了，最舊的經驗會被替換掉，以維持緩衝區的大小。

Step 3: 隨機抽取 mini-batch

從經驗回放緩衝區中，隨機抽取一個包含 N 個經驗的 mini-batch。這個 N 就是你的 Batch Size，通常為 32 或 64。

Step 4: 更新線上網路 (Online Network)

計算目標 Q 值 (y 
t
​
 )：對於 mini-batch 中的每一個經驗 (s 
t
​
 ,a 
t
​
 ,r 
t
​
 ,s 
t+1
​
 )，我們利用目標網路 (Target Network) 來計算其目標值：
y 
t
​
 =r 
t
​
 +γmax 
a 
′
 
​
 Q 
target
​
 (s 
t+1
​
 ,a 
′
 )

計算線上 Q 值：同時，我們利用線上網路來計算當前狀態 s 
t
​
  和動作 a 
t
​
  的預測 Q 值：
Q 
online
​
 (s 
t
​
 ,a 
t
​
 ;θ 
online
​
 )

計算損失並反向傳播：計算目標值 y 
t
​
  和預測值 Q 
online
​
  之間的均方誤差 (MSE) 作為損失函數。然後，利用梯度下降法，根據這個 mini-batch 的所有經驗的平均損失，來更新線上網路的權重 θ 
online
​
 。

Step 5: 定期更新目標網路

每隔一個固定的步數 (例如每 10,000 步)，我們會將線上網路的權重完整地複製到目標網路。

這個過程會不斷循環。智能體不斷地與環境互動、累積經驗，並定期從這些經驗中學習和更新網路。這回答了你關於「幾個動作產生幾個 y(t) 然後把幾個 roll out 出去」的問題：模型在每次 mini-batch 更新時，會根據 mini-batch 中的 N 個經驗，計算 N 個不同的目標值，並同時更新線上網路。 而互動產生的經驗則是一個一個被「roll out」並儲存到緩衝區中。


# Gemini Report


# **Deep Q-Networks: Pioneering Deep Reinforcement Learning and Overcoming Early Challenges**

## **I. Introduction to Deep Q-Networks (DQN)**

Reinforcement learning (RL) fundamentally addresses the challenge of sequential decision-making, where an autonomous agent learns to perform actions within an environment to maximize a cumulative reward over time. Q-learning, a prominent model-free RL algorithm, aims to learn an optimal action-value function, denoted as Q(s,a). This function quantifies the expected future reward for taking a specific action a in a given state s and subsequently following an optimal policy. Traditionally, Q-learning relies on tabular representations to store these Q(s,a) values. However, this approach faces a severe limitation: the "curse of dimensionality." In environments characterized by large or continuous state and action spaces, such as complex video games where states are represented by raw pixel inputs, building and populating a comprehensive Q-table becomes computationally intractable and practically impossible due to memory constraints and the sheer number of possible state-action pairs. This inherent limitation posed a significant barrier to applying Q-learning to real-world, high-dimensional problems.

The core conceptual shift introduced by Deep Q-Networks (DQN) was to overcome this curse of dimensionality by leveraging a powerful function approximator: a deep neural network (DNN). Instead of an explicit table, the DNN learns to approximate the Q(s,a) function, taking the state as input and outputting the Q-values for all possible actions.1 This approach strategically capitalized on the DNN's, specifically convolutional neural networks (CNNs), remarkable ability to learn complex, high-level features directly from raw input data. This capability had recently propelled deep learning to achieve state-of-the-art results in diverse domains like computer vision and speech recognition.1 By enabling the model to directly process raw pixels, DQN eliminated the need for laborious manual feature engineering, which had been a significant bottleneck in previous RL applications.2 This end-to-end learning paradigm represented a revolutionary step for reinforcement learning.6

The introduction of DQN marked a pivotal moment in artificial intelligence, arguably launching the entire field of deep reinforcement learning.1 It demonstrated for the first time that a deep learning model could successfully learn complex control policies directly from high-dimensional sensory input using reinforcement learning principles. This was concretely evidenced by its achievement of human-level performance in challenging domains like Atari games.2 This accomplishment validated the immense potential of combining deep learning's robust representation power with reinforcement learning's sophisticated decision-making framework, thereby opening vast new avenues for AI research and development.

## **II. The Seminal Work: Identifying the First Deep Q-Network Paper**

The pioneering work that first proposed and successfully demonstrated the use of deep neural networks for Q-value approximation within a reinforcement learning context, specifically for deriving control policies directly from high-dimensional sensory input, was "Playing Atari with Deep Reinforcement Learning." This paper was authored by Volodymyr Mnih, Koray Kavukcuoglu, David Silver, and their colleagues, and was initially published on arXiv in **December 2013**.1 This initial publication presented a convolutional neural network model trained with a variant of Q-learning. Its design allowed it to take raw pixel inputs from Atari games and output a value function that estimated future rewards for various actions.2

While the arXiv paper in 2013 served as the initial public release and conceptual breakthrough, a more comprehensive and refined version of this seminal work, titled "Human-level control through deep reinforcement learning," was subsequently published in the prestigious journal *Nature* in **February 2015**.3 This Nature publication significantly solidified the work's impact and brought it to a much wider scientific audience. It expanded the evaluation to a remarkable 49 Atari games, consistently demonstrating human-level or even superhuman performance across this diverse set, all while utilizing the same underlying algorithm and network architecture.1 This progression from an initial arXiv pre-print to a high-impact journal publication highlights a common trajectory in rapidly advancing scientific fields, where initial discoveries are quickly shared for community feedback, followed by more extensive validation and broader dissemination to establish foundational contributions.

The Mnih et al. paper showcased a truly groundbreaking achievement in artificial intelligence through its application to Atari game play. The Deep Q-Network (DQN) agent learned to play a variety of Atari 2600 games—initially seven games in the 2013 arXiv paper, expanding to 49 in the 2015 Nature paper—directly from raw pixel inputs.2 Crucially, this was accomplished without any game-specific information, hand-crafted features, or prior domain knowledge.4 The agent's ability to outperform all previous approaches on a majority of the games and even surpass human experts on several 1 was a monumental achievement. It powerfully demonstrated the potential for general-purpose learning agents in complex, high-dimensional environments.4 The network's capacity for complex strategic reasoning was evident in its ability to learn optimal strategies, such as getting the ball stuck in the top of the screen in Breakout.1 This indicated a significant step towards the broader ambition of Artificial General Intelligence (AGI), as the ability to apply a single, fixed architecture to diverse tasks, learning from scratch, represented a notable departure from previous game-playing AI, which often relied on game-specific algorithms.7

## **III. Challenges in Integrating Deep Neural Networks with Q-Learning**

While the conceptual appeal of using deep neural networks (DNNs) to approximate Q-values was immense for overcoming the curse of dimensionality, its direct application to reinforcement learning presented significant inherent instabilities. The combination of model-free reinforcement learning algorithms like Q-learning with non-linear function approximators (such as deep neural networks) or off-policy learning had been previously shown to cause Q-networks to diverge.2 This fundamental instability posed a major hurdle that severely limited the practical application of deep learning in reinforcement learning.

### **Detailed Discussion of the Inherent Difficulties:**

* Data Requirements: Learning from Sparse, Noisy, and Delayed Scalar Reward Signals.  
  Most successful deep learning applications, particularly in supervised learning, rely on vast amounts of meticulously hand-labeled training data, where each input has a clear, immediate target output. In stark contrast, reinforcement learning algorithms must learn from a single, scalar reward signal that is often sparse (rewards are infrequent, occurring only after specific achievements), noisy (subject to unpredictable variations), and significantly delayed (the consequence of an action might only be observed thousands of timesteps later).2 This substantial delay between actions and their eventual positive or negative consequences makes the "temporal credit assignment problem"—determining which specific actions contributed to a distant reward—exceptionally challenging for deep learning models.  
* Correlated Data Samples: Sequential Observations Leading to Highly Dependent Training Data.  
  Deep learning algorithms typically operate under the assumption that training data samples are independent and identically distributed (i.i.d.) for effective and stable gradient-based optimization.2 However, in reinforcement learning, an agent interacts with its environment sequentially. This means that consecutive observations (states) and actions are highly correlated (e.g., successive frames in a video game are very similar). Learning directly from these highly correlated sequences can lead to several issues: inefficient updates, high variance in gradients, and a tendency for the network to "forget" older experiences, potentially causing oscillations or convergence to suboptimal local minima.  
* Non-stationary Data Distribution: The Constantly Changing Policy and Q-values Creating a Moving Target.  
  A fundamental and particularly problematic characteristic of reinforcement learning is that the data distribution from which the agent learns is not fixed. Instead, it dynamically shifts as the algorithm learns new behaviors and its policy improves.2 As the Q-network learns and its predicted Q-values are updated, the agent's optimal policy (how it chooses actions) also changes. This, in turn, alters the sequence of states and actions it experiences, creating a constantly shifting, non-stationary target distribution for the deep learning model. For deep learning methods, which generally assume a fixed underlying data distribution for stable convergence, this non-stationarity can be highly problematic. The target Q-values, which are used to update the network's weights, are themselves derived from the same network (or a slightly older version), creating a moving and potentially unstable target.  
* Divergence Issues: Instability with Off-Policy Learning and Non-linear Function Approximators.  
  A critical theoretical and practical hurdle was the known tendency for Q-learning to diverge when combined with non-linear function approximators, especially in an off-policy setting.2 Off-policy learning implies that the agent is learning about an optimal policy while following a different, often more exploratory, behavior policy (e.g., an  
  ϵ-greedy strategy). When the target values for the Q-network updates are derived from the same network that is being updated, a small change in the network's weights can lead to large changes in the target values. This creates a positive feedback loop that can cause the network to become unstable and diverge.8 This issue historically led researchers to focus on linear function approximators for Q-learning, despite their limited representational power.

The challenges detailed above—sparse rewards, correlated data, non-stationary distributions, and divergence issues—are not merely practical hurdles; they represent a fundamental incompatibility between the assumptions underpinning most successful deep learning (supervised learning) and the inherent dynamics of reinforcement learning. Supervised learning thrives on large, independent, and identically distributed (i.i.d.) datasets with clear, fixed labels. Reinforcement learning, conversely, operates in a continuous feedback loop where data generation, policy improvement, and target calculation are intertwined. This inherent difference explains why a direct application of deep learning to Q-learning without modifications was prone to failure. The non-stationary data distribution and divergence issues are particularly interconnected and can be summarized as the "moving target" problem. When the network being trained is also generating the targets for its own updates (via the Bellman equation), any change in the network's weights immediately affects the target. This creates a highly unstable optimization landscape where the "ground truth" is constantly shifting. This specific problem is particularly insidious because it can lead to positive feedback loops and catastrophic divergence, making stable convergence extremely difficult. Understanding this "moving target" as a root cause clarifies the necessity of the innovative solutions implemented in DQN.

## **IV. Innovative Solutions Implemented in Deep Q-Networks**

To systematically address the profound challenges of instability, correlated data, and non-stationarity, Mnih et al. introduced several innovative techniques in their Deep Q-Network architecture. These solutions were crucial for achieving stable and effective learning in high-dimensional, complex environments.

**Table 1: Key Challenges and DQN Solutions**

This table provides a concise overview of the primary challenges faced when integrating deep neural networks with Q-learning and the specific solutions implemented in the original Deep Q-Network (DQN) paper to mitigate these issues.

| Challenge Faced by Deep Q-Learning with DNNs | DQN Solution Implemented | How the Solution Mitigates the Challenge | Relevant Snippet IDs |
| :---- | :---- | :---- | :---- |
| Sparse, Noisy, and Delayed Rewards | Reward Clipping | Normalizes reward scale (to \-1, 0, 1), making learning rates consistent across games and stabilizing error derivatives, simplifying the reward landscape. | 2 |
| Correlated Data Samples | Experience Replay | Stores agent experiences in a buffer and randomly samples minibatches, breaking temporal correlations and reducing variance of updates. | 2 |
| Non-stationary Data Distribution | Experience Replay | Averages the behavior distribution over many previous states, smoothing learning and preventing oscillations or divergence due to a constantly changing policy. | 2 |
| Divergence Issues with Non-linear Function Approximators & Bootstrapping ("Moving Target") | Target Network | Decouples the estimation of current Q-values from target Q-values by using a separate, periodically updated network, providing stable and consistent targets. | 3 |
| High Computational Demands of Raw Pixel Inputs | Preprocessing & Frame Skipping | Raw frames are converted to grayscale, down-sampled, and cropped. Frame skipping allows the agent to select actions less frequently, repeating actions over multiple frames, which reduces computational load per effective timestep. | 2 |
| Lack of Generality Across Diverse Tasks | Fixed Architecture & Hyperparameters | Demonstrates that the same network architecture and learning parameters can achieve high performance across diverse games without game-specific tuning, showcasing robustness and broad applicability. | 2 |

### **Experience Replay Mechanism:**

To effectively address the critical problems of correlated data samples and non-stationary data distributions, DQN introduced a novel and highly effective technique: the **experience replay mechanism**. This mechanism involves storing the agent's experiences—defined as tuples of (current state, action taken, reward received, next state) or (s\_t, a\_t, r\_t, s\_{t+1})—in a large, circular data structure referred to as a "replay memory" or "experience buffer".2 During the training phase, instead of learning directly from the agent's current interaction with the environment (online learning), Q-learning updates are applied to mini-batches of experiences that are randomly sampled from this stored pool of past transitions. This off-line sampling process is central to its effectiveness.2

This approach yields several significant advantages. Firstly, **reduced correlation** is achieved because randomly sampling experiences from the replay memory effectively breaks the strong temporal correlations that naturally exist between consecutive observations in an agent's interaction history. This decorrelation is crucial for gradient-based deep learning methods, as it significantly reduces the variance of the gradient updates, leading to a more stable and efficient learning process.2 Secondly, it addresses

**non-stationarity by smoothing the training distribution**. By drawing samples from a diverse pool of experiences collected over many different behaviors and policy iterations (as the policy evolves), experience replay averages the behavior distribution. This smoothing effect stabilizes the learning process and helps prevent the network parameters from oscillating or diverging due to a constantly shifting, non-stationary data distribution.2 Thirdly, a significant advantage of experience replay is its

**data efficiency**. Each stored experience tuple can be reused multiple times for training updates, maximizing the utility of every interaction the agent has with its environment.2 Finally, the use of experience replay inherently necessitates an

**off-policy learning** algorithm like Q-learning, as the data being learned from (past experiences) was generated by a potentially different policy than the one currently being optimized.2 This makes Q-learning a natural fit for this mechanism.

### **Target Network:**

To directly combat the inherent instability caused by the "moving target" problem—where the target Q-values used for updates are generated by the same network that is being updated—DQN introduced the concept of a **target network**.3 The target network is a separate, identical copy of the main Q-network (often referred to as the "online network"). However, its weights (

θt​) are updated much less frequently or more slowly than the weights (θ) of the online network.8

The primary function of the target network is to provide a stable, fixed reference point for calculating the target Q-values in the Bellman equation (i.e., r+γmaxa′​Q(s′,a′;θt​)). By decoupling the target calculation from the rapid, step-by-step updates of the online network, the target network effectively breaks the problematic feedback loop that can lead to oscillations and divergence.8 By holding the target network's weights constant for a predetermined number of steps (e.g., every 1,000 training steps, as in the original paper, or through softer, gradual updates in later variants), the optimization process becomes significantly more stable. This is analogous to having a fixed reference point in an optimization landscape, making convergence more reliable.10 This heuristic was a critical component in enabling the stable training of large, non-linear neural networks within a reinforcement learning context, despite the historical lack of strong theoretical convergence guarantees for such combinations.5

The combined effectiveness of experience replay and the target network highlights a crucial aspect of DQN's success. Experience replay primarily addresses the challenges related to the *data distribution* (correlated samples, non-stationarity) by creating a more independent and identically distributed (i.i.d.) dataset, which is a fundamental requirement for stable gradient-based deep learning. However, even with i.i.d. data, the bootstrapping nature of Q-learning (where the target value for an update is derived from the very network being updated) still poses a significant challenge, leading to the "moving target" problem. The target network then steps in to stabilize this target calculation. Thus, experience replay makes the *data* suitable for deep learning, while the target network makes the *learning update rule* stable for deep learning. They are not redundant but complementary, each tackling a different layer of the instability problem. This synergistic design demonstrates a sophisticated understanding of the complex interplay between data characteristics, learning algorithms, and neural network dynamics, and it became a foundational blueprint for many subsequent deep reinforcement learning algorithms.

### **Practical Enhancements:**

* Reward Clipping: Normalizing Reward Scales for Consistent Learning.  
  To manage the wide variance in score scales across different Atari games and to simplify the process of using a single, consistent learning rate across all environments, Mnih et al. implemented a simple yet effective technique: reward clipping. All positive rewards were fixed to \+1, all negative rewards were clipped to \-1, and zero rewards remained unchanged.2 This normalization prevents large reward magnitudes from dominating the error derivatives during training, ensuring a more stable and consistent learning signal regardless of the specific game being played. While this pragmatic choice might reduce the agent's ability to differentiate between rewards of different magnitudes (e.g., a score of \+10 vs. \+100), its practical benefit for stability and generalization across games was significant.5  
* Frame Skipping: Improving Computational Efficiency.  
  To enhance computational efficiency and allow the agent to experience more game time without significantly increasing runtime, a frame-skipping technique was employed. The agent was configured to select an action only on every k-th frame (typically k=4 for most games, k=3 for Space Invaders to ensure laser visibility) and then repeat that chosen action for all the skipped frames in between.2 This approach effectively reduces the frequency of decision-making and expensive neural network evaluations, making the overall training process much more computationally efficient while still allowing the agent to learn effectively from the environment's dynamics.  
* Fixed Architecture and Hyperparameters: Demonstrating Generality and Robustness.  
  A powerful testament to DQN's robustness and generality was the fact that the authors successfully used the same convolutional neural network architecture, the same learning algorithm, and the same set of hyperparameters across all the diverse Atari games they tested.2 This approach moved beyond traditional methods that often required game-specific feature engineering or extensive hyperparameter tuning. The architecture itself consisted of convolutional layers for automatic feature extraction from raw pixel inputs, followed by fully-connected layers, culminating in an output layer that provided the predicted Q-values for each possible action.2 This consistency underscored the method's potential for developing truly general-purpose learning agents.

The initial breakthrough achieved by DQN relied on a pragmatic blend of theoretically sound principles (Q-learning, CNNs) and clever engineering heuristics. Techniques like reward clipping, frame skipping, and the target network itself, while highly effective in practice for achieving stable learning and impressive performance, introduce approximations or delays that deviate from theoretically "pure" or real-time online learning. For instance, reward clipping simplifies the reward landscape, making it easier for the network to learn consistent gradients, but at the cost of losing fine-grained information about reward magnitudes. The target network, while stabilizing, introduces a temporal lag between the online policy and the target policy, which is not strictly "online." This demonstrates that in nascent fields like deep reinforcement learning, practical success often precedes complete theoretical understanding or perfectly optimal solutions. The ability to make effective engineering compromises to overcome immediate empirical hurdles is crucial for rapid progress. These pragmatic choices then serve as a foundation upon which subsequent research can build, leading to further theoretical refinements and more sophisticated, often less heuristic-dependent, solutions (e.g., Double DQN, Prioritized Experience Replay, and later attempts to remove the target network as mentioned in 8). This iterative process of empirical discovery and theoretical refinement is characteristic of scientific progress in complex domains.

## **V. Conclusion**

The seminal work by Mnih et al. on Deep Q-Networks (DQN) marked a watershed moment in artificial intelligence. It demonstrated, for the first time, the robust and scalable integration of deep neural networks with reinforcement learning principles. This allowed autonomous agents to learn complex control policies directly from high-dimensional, raw sensory inputs, a feat previously considered highly challenging due to the "curse of dimensionality" and the inherent instability of combining non-linear function approximators with model-free reinforcement learning. DQN's remarkable achievement of human-level performance across a diverse and challenging suite of Atari games, crucially using a single, fixed architecture and set of hyperparameters, underscored its profound generality and robustness. This capability fundamentally shifted the paradigm in AI research, paving the way for the broader field of deep reinforcement learning and inspiring widespread adoption and further innovation.

The core algorithmic innovations introduced in DQN—namely, the **experience replay mechanism** and the **target network**—were instrumental in overcoming the fundamental challenges that had previously plagued the combination of off-policy Q-learning with non-linear function approximators. Experience replay effectively addressed the issues of correlated data samples and non-stationary data distributions by decorrelating and smoothing the training signal. Concurrently, the target network provided stable and consistent targets for the Q-value updates, thereby preventing the oscillations and divergence inherent in the "moving target" problem. These foundational solutions, complemented by practical enhancements such as reward clipping for consistent learning signals and frame skipping for computational efficiency, transformed deep reinforcement learning from a theoretical concept with limited applicability into a practical and powerful framework. This enabled AI agents to learn complex, human-like behaviors in previously intractable, high-dimensional domains. DQN's success not only provided effective solutions to immediate technical problems but also established a robust methodological framework and a set of proven techniques. This framework became foundational for a wave of subsequent advancements in deep reinforcement learning, inspiring extensive research and diverse applications across various fields, from robotics to game AI and beyond.

#### **引用的著作**

1. Deep Q Networks (DQN) · Deep Reinforcement Learning \- Steven Schmatz, 檢索日期：8月 1, 2025， [https://stevenschmatz.gitbooks.io/deep-reinforcement-learning/content/deep-q-networks.html](https://stevenschmatz.gitbooks.io/deep-reinforcement-learning/content/deep-q-networks.html)  
2. Playing Atari with Deep Reinforcement Learning \- University of ..., 檢索日期：8月 1, 2025， [https://www.cs.toronto.edu/\~vmnih/docs/dqn.pdf](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
3. Constrained Deep Q-Learning Gradually Approaching Ordinary Q-Learning \- PMC, 檢索日期：8月 1, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC6914867/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6914867/)  
4. Playing Atari Games with Deep Reinforcement Learning and Human Checkpoint Replay arXiv:1607.05077v1 \[cs.AI\] 18 Jul 2016, 檢索日期：8月 1, 2025， [https://arxiv.org/pdf/1607.05077](https://arxiv.org/pdf/1607.05077)  
5. Playing Atari with Deep Reinforcement Learning, 檢索日期：8月 1, 2025， [https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)  
6. Human-level control through deep reinforcement learning \- PubMed, 檢索日期：8月 1, 2025， [https://pubmed.ncbi.nlm.nih.gov/25719670/](https://pubmed.ncbi.nlm.nih.gov/25719670/)  
7. Playing Atari Games with Deep Reinforcement Learning \- CSE \- IIT Kanpur, 檢索日期：8月 1, 2025， [https://cse.iitk.ac.in/users/cs365/2015/\_submissions/varshajn/report.pdf](https://cse.iitk.ac.in/users/cs365/2015/_submissions/varshajn/report.pdf)  
8. DeepMellow: Removing the Need for a Target Network in Deep Q-Learning \- IJCAI, 檢索日期：8月 1, 2025， [https://www.ijcai.org/proceedings/2019/0379.pdf](https://www.ijcai.org/proceedings/2019/0379.pdf)  
9. Human-level control through deep reinforcement learning | Request PDF \- ResearchGate, 檢索日期：8月 1, 2025， [https://www.researchgate.net/publication/272837232\_Human-level\_control\_through\_deep\_reinforcement\_learning](https://www.researchgate.net/publication/272837232_Human-level_control_through_deep_reinforcement_learning)  
10. What are target networks in DQN? \- Milvus, 檢索日期：8月 1, 2025， [https://milvus.io/ai-quick-reference/what-are-target-networks-in-dqn](https://milvus.io/ai-quick-reference/what-are-target-networks-in-dqn)  
11. Deep Q-Network (DQN) Agent \- MATLAB & Simulink \- MathWorks, 檢索日期：8月 1, 2025， [https://www.mathworks.com/help/reinforcement-learning/ug/dqn-agents.html](https://www.mathworks.com/help/reinforcement-learning/ug/dqn-agents.html)
