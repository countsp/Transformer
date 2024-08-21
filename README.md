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

分别与x相乘生成QKV
![Screenshot from 2024-08-21 14-29-48](https://github.com/user-attachments/assets/2b41ea71-3a89-4ce3-b57d-9510a2085fea)

![Screenshot from 2024-08-21 14-41-34](https://github.com/user-attachments/assets/911c7af7-c2da-4ed6-a1e8-f2759ddd2ee4)
 
**多头self-attention：** 
训练时生成多组QKV，使用不同的QKV

![Screenshot from 2024-08-21 14-53-14](https://github.com/user-attachments/assets/8af9d4c0-5e39-45ec-8e10-c508980e417a)

Concat: n* (h*dk)

Linear变换： (h*dk) * 512

Z : n*512 与 输入x维度相同
