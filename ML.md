# 损失函数

损失函数用于在训练神经网络时，提供一个量化指标，用以评估模型的当前表现。通过最小化这个损失值，可以优化模型参数，使模型的预测结果逐步接近真实值。

# 梯度

梯度是损失函数关于模型参数的偏导数，它提供了损失函数增加或减少的最快方向。

在数学和物理中，梯度向量指向一个函数在给定点上增长最快的方向。

在机器学习中，我们通常寻找梯度的反方向进行参数更新，因为我们的目标是最小化损失函数，而不是最大化它。



# eg
### 线性回归模型

假设我们有数据点集，我们想通过一个线性关系来模拟这些点。线性模型可以表示为：

$`y = wx + b`$

其中：
- $`y`$ 是预测值。
- $`x`$ 是输入特征。
- $`w`$ 是权重（斜率）。
- $`b`$ 是偏置（截距）。

### 损失函数

我们使用均方误差（MSE）作为损失函数，来衡量预测值 $`y`$ 与真实值 $`t`$ 之间的差距：

$`L = \frac{1}{N} \sum_{i=1}^N (y_i - t_i)^2`$

这里，$`N`$ 是样本数量，$`y_i`$ 是第 $`i`$ 个样本的预测值，$`t_i`$ 是第 $`i`$ 个样本的真实值。

### 计算梯度

为了最小化损失函数，我们需要找到损失函数相对于模型参数 $`w`$ 和 $`b`$ 的梯度：

- **关于 $`w`$ 的梯度**：

  $`\frac{\partial L}{\partial w} = \frac{2}{N} \sum_{i=1}^N (y_i - t_i) \cdot x_i`$

- **关于 $`b`$ 的梯度**：

  $`\frac{\partial L}{\partial b} = \frac{2}{N} \sum_{i=1}^N (y_i - t_i)`$

### 梯度下降更新规则

一旦我们计算了这些梯度，我们可以使用梯度下降算法来更新参数：

- **更新权重 $`w`$**：

  $`w := w - \eta \frac{\partial L}{\partial w}`$

- **更新偏置 $`b`$**：

  $`b := b - \eta \frac{\partial L}{\partial b}`$

其中 $`\eta`$ 是学习率，一个小的正数，决定了在梯度方向上前进的步长大小。

**使用梯度下降算法寻找最小损失函数值的过程想象为在 w 和 b 组成的参数空间中寻找一个点，使得在这个点上，损失函数 L 的值最小。**

```
import torch

# 随机生成一些数据
# 输入x和输出y的关系为 y = 3x + 2 + noise
x = torch.randn(100, 1)  # 100个样本，每个样本1个特征
y = 3 * x + 2 + torch.randn(100, 1) * 0.1  # 真实模型加上一些噪声

# 初始化权重和偏置
w = torch.randn(1, requires_grad=True)  # 需要计算梯度
b = torch.randn(1, requires_grad=True)  # 需要计算梯度

# 设置学习率
learning_rate = 0.01

# 训练模型
for i in range(1000):  # 进行1000次迭代
    # 前向传播
    y_pred = x * w + b  # 模型预测
    loss = ((y_pred - y) ** 2).mean()  # 计算均方误差损失

    # 显示训练过程中的损失
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss.item()}")

    # 反向传播
    loss.backward()  # 计算损失关于所有参数的梯度

    # 参数更新
    with torch.no_grad():  # 更新参数时不需要计算梯度
        w -= learning_rate * w.grad  # 更新权重
        b -= learning_rate * b.grad  # 更新偏置

        # 清零梯度，准备下一轮迭代
        w.grad.zero_()
        b.grad.zero_()

print(f"Trained model: y = {w.item()}x + {b.item()}")
```
