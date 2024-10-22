# [Diffusion Models in Vision: A survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10081412)

- Diffusion 的特点：forward adding Gaussian noise + reverse recover process

- Diffusion 的分类：

1. DDPM. 利用了 non-equilibrium thermodynamics theory, 是一种 latent variable model. 可以认为像 VAE

2. NCSN. 





# [DDPM](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)


# [NCSN](https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf)

- challenge of score matching: score __undefined__ in ambient space & bad langevin in low density region (slow mixing)

- 核心思想：__perturb__ the data with random Gaussian noise of various magnitudes; annealed version of Langevin dynamics

模型具体细节: 使用了 dilated UNet; modified conditioned instance normalization (加在每个 conv 和 pool 之前)

# [improved skills for NCSN](https://arxiv.org/abs/2006.09011)

1. 在 langevin 最后额外添加一步梯度但不加噪声, 不影响观感但是可以提升 FID

2. 研究 max noise: 理论分析得出, 应该选择样本之间最大的欧几里得距离 (比如 50 in CIFAR-10, 28 in MNIST); 几何衰减的噪声是合理的, 按照下面的规律选择:

$D$ 是数据维度; $\Phi$ 是高斯分布的 CDF; 应该选择比例 $\gamma$ 使得

$$\Phi(\sqrt{2D}(\gamma - 1)+3\gamma)-\Phi(\sqrt{2D}(\gamma - 1)-3\gamma) = 0.5$$

经过计算, MNIST 上选择 $\gamma = 10/9$ 比较合适. 也就是说 $L=75$.

原论文中的 $\gamma = 1.58$. 差距很大！！

3. 重点简化: 在神经网络中, 不再传入 noise scale, 而是直接除以 noise scale. 数学上近似是对的, 更加容易学?

4. 关于 langevin 中 $\epsilon$ 的选择 (原论文中说这个选取可以跨域1-2个数量级): 最小化

$$(1-\frac{\epsilon}{\sigma_L^2})^{2T}\left(\gamma^2 - \frac{2\epsilon}{\sigma_L^2\left(1-\left(1-\dfrac{\epsilon}{\sigma_L^2}\right)^2\right)}\right)+\frac{2\epsilon}{\sigma_L^2\left(1-\left(1-\dfrac{\epsilon}{\sigma_L^2}\right)^2\right)}$$

而 $T$ 可以比较大.

在 MNIST 上, $\epsilon = 1e-5$ 是合适的.

5. 在 eval 的时候使用 exponential moving average, 来减缓模型参数的过大变化.