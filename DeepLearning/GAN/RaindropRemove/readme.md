# 雨滴去除相关文献论文总结

#### pix2pix

##### 相关代码

1. [pix2pix官方github(pytorch)](https://github.com/phillipi/pix2pix)
2. [pix2pix tensorflow实现](https://github.com/affinelayer/pix2pix-tensorflow)

##### 目的
传统的图片到图片的“转化”通常需要人为构造复杂且合理的损失函数，针对不同的问题都必须采用特定的机制，虽然他们的背景都是从像素到像素的映射（pix2pix）。但是，GAN是一个不需要构建复杂损失函数的结构，它会自动学习从输入到输出图片的映射。因此，应用这个到图片“翻译”问题中，就可以实现一个泛化的模型。

##### 结果贡献

##### 目标函数
Pix2Pix的算是函数为：
<img src="https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/pix2pix/4.png" style="border:none;">
为了做对比，同时再去训练一个普通的GAN， 即只让D判断时候为真实图像
<img src="https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/pix2pix/5.png" style="border:none;">
对于图像翻译任务而言，G的输入和输出之间其实共享了很多信息，比如图像上色任务，输入和输出之间就共享了边信息。因而为了保证输入图像和输出图像之间的相似度。还加入了L1 Loss
<img src="https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/pix2pix/6.png" style="border:none;">

那么，汇总的损失函数为
<img src="https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/pix2pix/7.png" style="border:none;">


##### 网络结构

> 对抗框架：
> Pix2Pix是基于GAN框架的，那么首先定义输入输出，Pix2Pix在Generator部分的输入不同于一般的GAN，而是跟Conditional GAN相同，输入为一张
> 模型结构:

> Generator: U-Net
>输入和输出之间会共享很多的信息。如果使用普通的卷积神经网络，那么会导致每一层都承载保存着所有的信息，这样神经网络很容易出错，因而，使用U-Net来进行减负。首先U-Net也是Encoder-Decoder模型，其次，Encoder和Decoder是对称的。
所谓的U-Net是将第i层拼接到第n-i层，这样做是因为第i层和第n-i层的图像大小是一致的，可以认为他们承载着类似的信息。
<img src="https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blog2017/pix2pix/3.png" style="border:none;">

> Discriminator: PatchGAN
>在损失函数中，L1被添加进来来保证输入和输出的共性。这就启发出了一个观点，那就是图像的变形分为两种，局部的和全局的。既然L1可以防止全局的变形。那么只要让D去保证局部能够精准即可。

>于是，Pix2Pix中的D被实现为Patch-D，所谓Patch，是指无论生成的图像有多大，将其切分为多个固定大小的Patch输入进D去判断。

>这样有很多好处：

>D的输入变小，计算量小，训练速度快。
>因为G本身是全卷积的，对图像尺度没有限制。而D如果是按照Patch去处理图像，也对图像大小没有限制。就会让整个Pix2Pix框架对图像大小没有限制。增大了框架的扩展性。
##### 训练过程
###### 训练细节
- 梯度下降，G、D交替训练
- 使用Adam算法训练
- 在inference的时候，与train的时候一样，这和传统CNN不一样，因为传统上inference时dropout的实现与train时不同。
- 在inference的时候，使用test_batch的数据。这也和传统CNN不一样，因为传统做法是使用train set的数据。
- batch_size = 1 or 4，为1时batch normalization 变为instance normalization
###### 评测
- AMT，一种人工评测平台，在amazon上。
- FCN-8，使用预训练好的语义分类器来判断图片的可区分度，这是一种不直接的衡量方式
##### 试验结果
###### Loss function试验

###### 色彩试验

###### Patch对比实验

