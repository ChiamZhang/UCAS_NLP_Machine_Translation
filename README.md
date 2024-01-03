**IWSLT14 En-Zh机器翻译作业**

1. 实验目的
   1. 进一步加深对机器翻译任务基本目标和流程的理解。
   2. 掌握卷积神经网络、循环神经网络、Transformer等处理文本的各项技术。
   3. 加强对pytorch、tensorflow等深度学习框架的使用能力。
2. 实验要求
   1. 任选一个深度学习框架建立一个机器翻译模型，实现在IWSLT14 En-Zh数据集上进行的机器翻译。
   2. 按规定时间在课程网站提交实验报告、代码。
3. 任务介绍

标准的机器翻译流程包括：

   1. 数据预处理及分词（BPE，JIEBA等）
   2. 生成模型的构建及训练（RNN, CNN, Transformer等）
   3. 生成模型的解码（Greedy, Beam Search等）

本作业主要针对生成模型的构建及训练部分进行考核，对分词方式和解码方式不做过多的要求。**序列生成任务需要大量的计算资源，公平起见，本作业会按照提交的代码判定成绩而不是模型的训练效果。**

1. 数据集介绍

本次实验旨在实现一个简单的中英翻译系统。所用的数据集是从小规模的机器翻译数据集IWSLT14 En-Zh中抽取的数据，其全部数据均在附件的en-zh.rar压缩包中。这个数据集包含了143920个训练样本，19989个验证样例和15992个测试样例。其中，训练集数据在train.zh/train.en中，验证数据在valid.zh/valid.en中，测试集数据在test.zh/test.en中。X.zh与X.en中的数据每一行是对齐的。
**如果计算资源有限，可以减小使用的数据比例。**

1. 参考模型

本次实验要求实现一种机器翻译模型即可，可采用**下面三篇参考论文中的任意一种或自己设计的结构**均可。

   1. 使用CNN进行机器翻译，可参考2017年ICML的ConvS2S进行机器翻译的论文[[1]](#endnote-2)。
   2. 使用RNN/LSTM进行机器翻译，可参考2015年ICLR的基于LSTM+Attention的机器翻译论文[[2]](#endnote-3)。
   3. 计算资源比较充足的同学，也可采用Transformer进行机器翻译，可参考2017年NIPS的基于Transformer的机器翻译论文[[3]](#endnote-4)。

1. 作业提交说明
   1. 要求使用上述参考论文中的结构或自己实现的结构来实现机器翻译模型。
   2. 提交的作业除代码外，需包含一份简要报告来说明自己使用的分词策略、模型结构以及解码策略。如果有自己的一些独特的思考和实现可以详细写出。
2. 参考文献
3. [] Gehring J, Auli M, Grangier D, et al. Convolutional sequence to sequence learning[C]//International Conference on Machine Learning. PMLR, 2017: 1243-1252. [↑](#endnote-ref-2)

4. [] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.. [↑](#endnote-ref-3)

5. [] Vaswani et al. Attention is All You Need. NIPS 2017. [↑](#endnote-ref-4)

