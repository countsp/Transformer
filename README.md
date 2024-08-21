![Screenshot from 2024-08-21 13-46-10](https://github.com/user-attachments/assets/7247ad57-1c99-4f03-bfb8-eb249a6a0c07)

# Embedding

![Screenshot from 2024-08-21 14-23-03](https://github.com/user-attachments/assets/cdfce41f-abd8-4993-8b0e-ba5bb72f1f31)

将句子编码,升维。将位置信息编码，升维，相加。维度需要合适，太大了过拟合。过少，模型可能无法充分捕捉数据中的复杂模式或语义关系。

例如，输入句子 "I like apples"，它可以表示为三个单词的索引序列：

**输入：**

[103,459,892]

**输出：**

[0.23,−0.67,0.45,...,0.12,0.98] : I

[0.13,−0.12,0.35,...,0.42,0.08] : like

[0.32,−0.42,0.45,...,0.88,0.12] : apples
