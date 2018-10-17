条件随机场 (conditional random fields, CRFs)
=======================================

### 简介
条件随机场由J.Latterty等人（2001）年提出，近几年来在自然语言处理和图像处理等领域得到了广泛的应用。

CRF是用来标注和划分结构数据的概率化模型。言下之意，就是对于给定的输出标识序列Y和观察序列X，条件随机场通过定义概率P(Y|X)，而不是联合概率分布P(X,Y)来描述模型。CRF也可以看作是一个无向图模型或者马尔可夫随机场

### 定义
设G=（V,E）为一个无向图，V为结点集合，E为无向边的集合。
<img src="https://latex.codecogs.com/gif.latex?Y=\{{Y_\upsilon&space;|\upsilon&space;\in&space;V}\}" title="Y=\{{Y_\upsilon |\upsilon \in V}\}" />
，即V中的每个结点对应于一个随机变量
<img src="https://latex.codecogs.com/gif.latex?Y_\upsilon" title="Y_\upsilon" />
，其取值范围为可能的标记集合{y}。如果以观察序列X为条件，每一个随机变量
<img src="https://latex.codecogs.com/gif.latex?Y_\upsilon" title="Y_\upsilon" />
都满足以下马尔可夫特性：

<img src="https://latex.codecogs.com/gif.latex?p(Y_\upsilon&space;|X,Y_\omega,\omega&space;\neq&space;\upsilon)=p(Y_\upsilon&space;|X,Y_\omega,\omega&space;\sim&space;\upsilon)" title="p(Y_\upsilon |X,Y_\omega,\omega \neq \upsilon)=p(Y_\upsilon |X,Y_\omega,\omega \sim \upsilon)" />

其中，
<img src="https://latex.codecogs.com/gif.latex?\omega&space;\sim&space;\upsilon" title="\omega \sim \upsilon" />
表示两个结点在图G中是邻近结点。那么，（X,Y)为一个条件随机场。

### CRF和逐帧Softmax的不同

原文 ： [简明条件随机场CRF介绍 | 附带纯Keras实现](https://mp.weixin.qq.com/s/BEjj5zJG3QmxvQiqs8P4-w)

逐帧 softmax 和 CRF 的根本不同了：前者**将序列标注看成是 n 个 k 分类问题**，后者**将序列标注看成是 1 个 k^n 分类问题**。

具体来讲，在CRF的序列标注问题中，我们要计算的是条件概率：

<img src="https://latex.codecogs.com/gif.latex?P(y_1,&space;...,&space;y_n|x_1,...,x_n)&space;=&space;P(y_1,&space;...,&space;y_n|x),&space;x&space;=&space;(x_1,...,x_n)" title="P(y_1, ..., y_n|x_1,...,x_n) = P(y_1, ..., y_n|x), x = (x_1,...,x_n)" />

为了得到这个概率的估计，CRF做了两个假设：

*假设一：该分布是指数分布。*

这个假设意味着存在函数
<img src="https://latex.codecogs.com/gif.latex?f(y_1,&space;...,&space;y_n;x)" title="f(y_1, ..., y_n;x)" />
，使得：

<img src="https://latex.codecogs.com/gif.latex?P(y_1,&space;...,&space;y_n|x)&space;=&space;\frac{1}{Z(x)}exp\{f(y_1,&space;...,&space;y_n;x)\}" title="P(y_1, ..., y_n|x) = \frac{1}{Z(x)}exp\{f(y_1, ..., y_n;x)\}" />

其中Z(x)是归一化因子，因为这个是条件分布，所以归一化因子跟x有关。这个f函数可以视为一个打分函数，打分函数取指数并归一化后就得到概率分布。

*假设二：输出之间的关联仅发生在相邻位置，并且关联是指数相加性的。*

这个假设意味着函数
<img src="https://latex.codecogs.com/gif.latex?f(y_1,&space;...,&space;y_n;x)" title="f(y_1, ..., y_n;x)" />
可以进一步简化为：

<img src="https://latex.codecogs.com/gif.latex?f(y_1,&space;...,&space;y_n;x)&space;=&space;h(y_1;x)&plus;g(y_1,y_2;x)&plus;h(y_2;x)&plus;...&plus;g(y_{n-1},y_n;x)&plus;h(y_n;x)" title="f(y_1, ..., y_n;x) = h(y_1;x)+g(y_1,y_2;x)+h(y_2;x)+...+g(y_{n-1},y_n;x)+h(y_n;x)" />

这也就是说，现在我们只需要对每一个标签和每一个相邻标签对分别进行打分，然后将所有打分结果求和得到总分。

**线性链CRF**

尽管已经做了大量简化，但一般来说，上式所表示的概率模型还是过于复杂，难以求解。于是考虑到当前深度学习模型中，RNN或层叠CNN等模型已经能够比较充分捕捉各个y与输出x的联系，因此不妨考虑函数g跟x无关，那么：

<img src="https://latex.codecogs.com/gif.latex?f(y_1,&space;...,&space;y_n;x)&space;=&space;h(y_1;x)&plus;g(y_1,y_2)&plus;h(y_2;x)&plus;...&plus;g(y_{n-1},y_n)&plus;h(y_n;x)" title="f(y_1, ..., y_n;x) = h(y_1;x)+g(y_1,y_2)+h(y_2;x)+...+g(y_{n-1},y_n)+h(y_n;x)" />

这时候g实际上就是一个有限的、待训练的参数矩阵而已，而单标签的打分函数
<img src="https://latex.codecogs.com/gif.latex?h(y_i;x)" title="h(y_i;x)" />
我们可以通过RNN或者CNN来建模。因此，该模型可以简历的，其中概率分布变为：

<img src="https://latex.codecogs.com/gif.latex?P(y_1,...,y_n|x)=\frac{1}{Z(x)}exp(h(y_1;x)&space;&plus;&space;\sum^{n-1}_{k=1}g(y_k,y_{k&plus;1})&plus;h(y_{k&plus;1};x))" title="P(y_1,...,y_n|x)=\frac{1}{Z(x)}exp(h(y_1;x) + \sum^{n-1}_{k=1}g(y_k,y_{k+1})+h(y_{k+1};x))" />

这就是线性链CRF的概念。

**归一化因子**

为了训练CRF模型，我们用最大似然方法，也就是用：

<img src="https://latex.codecogs.com/gif.latex?-logP(y_1,...,y_n|x)" title="-logP(y_1,...,y_n|x)" />

作为损失函数，可以算出它等于：

<img src="https://latex.codecogs.com/gif.latex?-(h(y_1;x)&space;&plus;&space;\sum^{n-1}_{k=1}g(y_k,y_{k&plus;1})&plus;h(y_{k&plus;1};x))&space;&plus;&space;log(Z(x))" title="-(h(y_1;x) + \sum^{n-1}_{k=1}g(y_k,y_{k+1})+h(y_{k+1};x)) + log(Z(x))" />

其中第一项是原来概率式的分子的对数，它是目标序列的打分，并不难计算。*真正的难度在于分母的对数
<img src="https://latex.codecogs.com/gif.latex?logZ(x)" title="logZ(x)" />
这一项。

归一化因子，在物理上也叫配分函数，在这里它需要我们对所有可能的路径的打分进行指数求和，而我们前面已经说到，这样的路径数是指数量级的（k^n），因此直接来算几乎是不可能的。
事实上，*归一化因子难算，几乎是所有概率图模型的公共难题*。幸运的是，在 CRF 模型中，由于我们只考虑了临近标签的联系（马尔可夫假设），因此我们可以递归地算出归一化因子，这使得原来是指数级的计算量降低为线性级别。

具体来说，我们将计算到时刻 t 的归一化因子记为 Zt，并将它分为 k 个部分：

<img src="https://latex.codecogs.com/gif.latex?Z_t&space;=&space;Z^{(1)}_t&space;&plus;&space;Z^{(2)}_t&space;&plus;&space;...&plus;Z^{(k)}_t" title="Z_t = Z^{(1)}_t + Z^{(2)}_t + ...+Z^{(k)}_t" />

其中
<img src="https://latex.codecogs.com/gif.latex?Z^{(1)}_t&space;,...,Z^{(k)}_t" title="Z^{(1)}_t ,...,Z^{(k)}_t" />
分别是截止到当前时刻t中、以标签1,...,k为终点的所有路径的得分指数和。那么，我们可以递归地计算：

<img src="https://latex.codecogs.com/gif.latex?Z^{(1)}_{t&plus;1}=(Z^{(1)}_tG_{11}&plus;Z^{(2)}_tG_{21}&plus;...&plus;Z^{(k)}_tG_{k1})h_{t&plus;1}(1|x)" title="Z^{(1)}_{t+1}=(Z^{(1)}_tG_{11}+Z^{(2)}_tG_{21}+...+Z^{(k)}_tG_{k1})h_{t+1}(1|x)" />

<img src="https://latex.codecogs.com/gif.latex?Z^{(2)}_{t&plus;1}=(Z^{(1)}_tG_{12}&plus;Z^{(2)}_tG_{22}&plus;...&plus;Z^{(k)}_tG_{k2})h_{t&plus;1}(2|x)" title="Z^{(2)}_{t+1}=(Z^{(1)}_tG_{12}+Z^{(2)}_tG_{22}+...+Z^{(k)}_tG_{k2})h_{t+1}(2|x)" />

...

<img src="https://latex.codecogs.com/gif.latex?Z^{(k)}_{t&plus;1}=(Z^{(1)}_tG_{1k}&plus;Z^{(2)}_tG_{2k}&plus;...&plus;Z^{(k)}_tG_{kk})h_{t&plus;1}(k|x)" title="Z^{(k)}_{t+1}=(Z^{(1)}_tG_{1k}+Z^{(2)}_tG_{2k}+...+Z^{(k)}_tG_{kk})h_{t+1}(k|x)" />

可以简写为矩阵形式：

<img src="https://latex.codecogs.com/gif.latex?Z_{t&plus;1}&space;=&space;Z_tG\bigotimes&space;H(y_{t&plus;1}|x)&space;,&space;Z_t&space;=&space;\{Z^{(1)}_t,...,Z^{(k)}_t\}" title="Z_{t+1} = Z_tG\bigotimes H(y_{t+1}|x) , Z_t = \{Z^{(1)}_t,...,Z^{(k)}_t\}" />

G是对
<img src="https://latex.codecogs.com/gif.latex?g(y_i,y_j)" title="g(y_i,y_j)" />
各个元素取指数后的矩阵，即
<img src="https://latex.codecogs.com/gif.latex?G&space;=&space;e^{g(y_i,y_j)}" title="G = e^{g(y_i,y_j)}" />
；而
<img src="https://latex.codecogs.com/gif.latex?H(y_{t&plus;1}|x)" title="H(y_{t+1}|x)" />
是编码模型
<img src="https://latex.codecogs.com/gif.latex?h(y_{t&plus;1}|x)" title="h(y_{t+1}|x)" />
（RNN,CNN等）对位置t+1的各个标签的打分的指数，即
<img src="https://latex.codecogs.com/gif.latex?H(y_{t&plus;1}|x)&space;=&space;e^{h(y_{t&plus;1}|x)}" title="H(y_{t+1}|x) = e^{h(y_{t+1}|x)}" />
，也是一个向量。

**动态规划**

写出损失函数 −logP(y1,…,yn|x) 后，就可以完成模型的训练了，因为目前的深度学习框架都已经带有自动求导的功能，只要我们能写出可导的 loss，就可以帮我们完成优化过程了。

那么剩下的最后一步，就是模型训练完成后，如何根据输入找出最优路径来。跟前面一样，这也是一个从 k^n 条路径中选最优的问题，而同样地，因为马尔可夫假设的存在，它可以转化为一个动态规划问题，用 viterbi 算法解决，计算量正比于 n。

动态规划在本博客已经出现了多次了，它的递归思想就是：一条最优路径切成两段，那么每一段都是一条（局部）最优路径。在本博客右端的搜索框键入“动态规划”，就可以得到很多相关介绍了，所以不再重复了。

一个例子

[crf](https://github.com/bojone/crf)
