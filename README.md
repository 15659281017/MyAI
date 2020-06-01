# MyAI
# AI团队任务

·小组名：林同学请打卡队。
·小组成员：穆浩然（组长）、罗家宏、林聪杰、朱心溁、郭晓文。

·项目名称：智能垃圾分类系统。
·主要功能：实现对垃圾图片的识别与分类。

·操作流程：
	1、运行train.py文件，生成许多训练模型，然后会调用plot.py绘制每一次epoch的模型的训练准确率accuracy、训练损失loss、验证准确率val_acc、验证损失val_loss，并将图片保存到本地。
	2、选择准确率最高的一个模型，使用predict.py载入模型。
	3、运行main.py，会调用prepare_image.py来进行最后的展示。
	4、prepare_image.py会从测试集里随机抽取任意种类的垃圾图片共15张，按每行5张，一共3行排列，并在每张图片的上方显示预测种类“pred”和实际种类“truth”来对比模型的准确性，并将图片保存到本地。
	
·模型：
    我们在机器学习的时候采用CNN卷积神经网为原理，使用Sequential线性模型，在分类方法上使用了Softmax多分类方法。
    模型结构我们设置了：
        ·卷积层：5层，其中32x3x3的2个，64x3x3的2个，128x3x3的1个；
        ·池化层：5层，池大小均为2x2；
        ·全连接层：2层，一个位Dense(128,relu)，一个为Dense(6,softmax)。
    而在损失函数和优化器的选择上，我们选择了：
        ·损失函数：loss='categorical_crossentropy'；
        ·优化器：optimizer='rmsprop'。

·项目的优点：
    1、我们的项目能够在每一次epoch之后保存模型的训练准确率accuracy、训练损失loss、验证准确率val_acc、验证损失val_loss，并保存到数组中，用来绘制模型的accuracy和loss折线图，并用来分析模型的属性变化。
    2、在项目的结果展示时，系统会从数据集中随机抽取15张图片进行排列、分析、预测和对比，有较强的随机性，展示也很直观。
    3、模型使用的是CNN卷积神经网络模型。最终的模型识别准确率有76%，可以识别较多垃圾的种类并给出正确的种类预测。
    4、我们会将折线图和结果展示都保存到本地，以便于以后的复盘分析和处理。

·项目的不足：
    1、在测试过程中，我们发现了一问题，就是识别系统的准确性十分不稳定，这是由于机器学习模型的准确率不够高导致的。虽然模型的准确率达到了76%，但实际测试时还是会出现部分预测错误的情况。
    在epochs的设置方面，最开始是30，后来发现提高epochs可以增加准确率后，我们一味地堆加epochs到50、100.后来在accuracy和loss的折线图中我们发现，当epochs为100时，训练集的loss在50左右下降到最低点，然后开始缓慢升高。这就是出现了过拟合现象！最终我们采取epochs为50进行训练。
    在模型方面，我们适当增加了卷积层、池化层、全连接层的数量，事实证明准确率的确有一定提升。
    我们尝试了几种不同的方法来提高准确性，如增加迭代次数，增加卷积层、池化层、全连接层数量，使用图片增强等方式。但由于时间和能力有限，这个应该修复的问题，没有时间在这个版本修复，只能延迟到下一个版本再来完成了。
    2、我们的垃圾分类一共有六大类：cardboard、metal、glass、paper、plastic、trash。但是由于trash类中的图片比较杂乱，是其他垃圾类，导致学习模型不能分类出trash。也就是说，分析结果为trash类的情况从未在测试中出现过。
    3、有时测试时会出现预测种类为None的情况，pred:None。这种情况发生时，该图片的相似度数组的值都非常奇怪，而非正常的0或1。这一情况还不清楚原因，推测可能是与六大种类的相似度都不高。

# AI团队任务
# MyAI
