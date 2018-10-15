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

