[ref](https://www.bilibili.com/video/BV1Km421K7X8/?spm_id_from=333.337.search-card.all.click&vd_source=e0dc0ee350340932342c58cb327ec5a2)

![Screenshot from 2024-08-21 13-46-10](https://github.com/user-attachments/assets/7247ad57-1c99-4f03-bfb8-eb249a6a0c07)

# Embedding

![Screenshot from 2024-08-21 14-23-03](https://github.com/user-attachments/assets/cdfce41f-abd8-4993-8b0e-ba5bb72f1f31)

**词embedding：**

将句子编码,升维（512）。将位置信息编码，升维，相加。维度需要合适，太大了过拟合。过少，模型可能无法充分捕捉数据中的复杂模式或语义关系。

例如，输入句子 "I like apples"，它可以表示为三个单词的索引序列：

**输入：**

[103,459,892]

**输出：**

[0.23,−0.67,0.45,...,0.12,0.98] : I

[0.13,−0.12,0.35,...,0.42,0.08] : like

[0.32,−0.42,0.45,...,0.88,0.12] : apples

**位置embedding**：

一般使用sincos或者RoPE

**x = 词embedding + 位置embedding**


# Self-Attention

权重矩阵 WQ​、WK​、WV​ 是在模型训练过程中通过反向传播算法学习得到的。

对于输入序列中的每个词，我们首先生成它的 Query 向量、Key 向量和 Value 向量。

![Screenshot from 2024-08-21 14-29-48](https://github.com/user-attachments/assets/2b41ea71-3a89-4ce3-b57d-9510a2085fea)

$`Q*K^T`$ ：计算 Query 和 Key 向量的点积，表示当前词与其他词的相关性。点积越大，表示相关性越强。

$`1/\sqrt{dk}`$ 这是一个缩放因子，用来防止点积的数值过大，避免梯度消失或爆炸。dk​ 是 Key 向量的维度。

$`Softmax`$ :对每个词的相关性分数进行归一化，将其转化为概率分布，表示每个词对当前词的贡献大小。

$`V`$ ：将注意力权重应用于 Value 向量，加权求和，得到当前词的输出表示。



![Screenshot from 2024-08-21 14-41-34](https://github.com/user-attachments/assets/911c7af7-c2da-4ed6-a1e8-f2759ddd2ee4)





**多头self-attention：** 
训练时生成多组Wq,Wk,Wv，生成不同的QKV

![Screenshot from 2024-08-21 14-53-14](https://github.com/user-attachments/assets/8af9d4c0-5e39-45ec-8e10-c508980e417a)

Concat: n* (h*dk)

Linear变换： (h*dk) * 512

Z : n*512 与 输入x维度相同

# Encoder(包括多头)

![Screenshot from 2024-08-21 15-01-51](https://github.com/user-attachments/assets/e869dcc7-336a-4478-b17f-1fc43331e070)

LayerNorm( X + MultiHeadNorm(X) ) 由两部分组成，add和norm

Add为残差连接

![Screenshot from 2024-08-21 15-33-33](https://github.com/user-attachments/assets/14c541a6-1c60-452d-bef1-0c7b879b1d28)

Norm能加速收敛

![Screenshot from 2024-08-21 15-32-30](https://github.com/user-attachments/assets/c091e9f2-6edc-4585-bc5f-c8e15bbc8b78)

**FFN（FeedForward）**

每个编码器或解码器层中的Feedforward部分是一个两层的全连接神经网络，具有如下形式：

**第一次变换：**

线性变换 + 非线性激活函数（通常使用ReLU激活函数）升维
    
```math
   FFN(x) = \text{ReLU}(xW_1 + b_1)
```


**x** 是输入向量（如从多头注意力得到的输出）。

**W1** 是第一层的权重矩阵。

**b1** 是第一层的偏置项。

第一层的权重矩阵 **W1** 和偏置项 **b1**​ 都是通过模型训练过程学到的。

Feedforward网络中间层的维度，通常比 dmodeldmodel​ 更大，增加了模型的容量。

**第二次线性变换：**

```math
FFN(x)=\text{ReLU}(xW1 ​+ b1​) W2​ + b2​
```

**W2** 是第二层的权重矩阵。

**b2​** 是第二层的偏置项。

**这两步线性变换将输入从原始维度 dmodel​ 转换为一个更高维度的表示 dff​，然后再将其还原到原始维度 dmodel。**


# Decoder

![Screenshot from 2024-08-21 16-48-41](https://github.com/user-attachments/assets/671b4a49-b9ac-45d1-9dcb-1b99f3f96436)


## Q&A

**与 Seq2Seq 模型的不同之处**

在每个时间步，我们输入直到当前时间步所产生的整个输出序列，而不是只输入上一个时间步产生的词（类似输入序列长度可变的自回归模型）。


这非常重要，把原文粘过来：The difference from the Seq2Seq model is that, at each timestep， we re-feed the entire output sequence generated thus far, rather than just the last word.
