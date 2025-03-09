## 说明

	MNIST手写数字识别程序
	
	其中：
	(1) main_CNN.py：训练和测试简单的CNN模型并保存
            main_CNN_2.0.py:卷积层添加至三层，增加正则化防止过拟合
	  
	(2) main_mobileNetV2.py是使用mobileNetV2预训练模型进行训练和测试的代码（需要用到main_CNN.py中的部分模块）。
	
	(3) mnist_mobilenetV2_alone.py也是使用mobileNetV2预训练模型进行训练和测试的代码，是独立可以运行的。

        (4)predict_CNN_2.0.py：增加RGB图灰度化，增加gradio图形化交互界面，可以上传图片以识别输出。




### 以上代码通常均输出：

[8]

表示识别结果为8，即识别正确。
