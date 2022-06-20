# 本项目yolov4代码参考:

https://www.bilibili.com/video/BV1Q54y1D7vj?spm_id_from=333.999.0.0

# 剪枝paper参考:

Pruning Filters for Efficient ConvNets

------

CSDN：https://blog.csdn.net/z240626191s/article/details/124326595

环境：

windows 10

pytorch 1.7.0(低版本应该也是可以的)
torchvision 0.8.0
torch_pruning

**2022-06-20更新说明：**

上个版本中训练自己的数据集默认采用torch.save(model,'xxx.pth')保存，在剪枝的时候发生了报错(这是由于**分布式训练**导致的)，因此在utils/utils_fit.py模型保存方式为model.state_dict()

训练完在自己的数据集后在利用save_whole_model函数将模型保存。

修改了剪枝微调训练的一些bug问题

------------------------------------------------------

# 安装包
```python
pip install torch_pruning
```



------------------------------------------------------

# 导入包
```python
import torch_pruning as tp
```



# 模型的实例化(针对已经训练好的模型)
```python
model = torch.load('权重路径')
model.eval()
```


# 对于非单层卷积的通道剪枝(不用看3.1)
剪枝之前统计一下模型的参数量

```python
num_params_before_pruning = tp.utils.count_params(model)
```



# 1. setup strategy (L1 Norm) 计算每个通道的权重
```python
strategy = tp.strategy.L1Strategy()
```



# 2.建立依赖图(与torch.jit很像)
```python
DG = tp.DependencyGraph()
DG = DG.build_dependency(model, example_inputs=torch.randn(1, 3, input_size[0], input_size[1])) # input_size是网络的输入大小
```



# 3.分情况(1.单个卷积进行剪枝 2.层进行剪枝)

## 3.1 单个卷积

**(会返回要剪枝的通道索引，这个通道是索引是根据L1正则得到的)**

```python
pruning_idxs = strategy(model.conv1.weight, amount=0.4)  # model.conv1.weigth是对特定的卷积进行剪枝,amount是剪枝率
```

将根据依赖图收集所有受影响的层，将它们传播到整个图上，然后提供一个PruningPlan正确修剪模型的方法。

```python
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv, idxs=pruning_idxs )
pruning_plan.exec()
torch.save(model, 'pru_model.pth')
```



## 3.2 层剪枝

**(需要筛选出不需要剪枝的层，比如yolo需要把头部的预测部分取出来，这个是不需要剪枝的)**

```python
excluded_layers = list(model.model[-1].modules())
for m in model.modules():
    if isinstance(m, nn.Conv2d) and m not in excluded_layers:
        pruning_plan = DG.get_pruning_plan(m,tp.prune_conv, idxs=strategy(m.weight, amount=0.4))
        print(pruning_plan)
        # 执行剪枝
        pruning_plan.exec()
```



**如果想看一下剪枝以后的参数，可以运行：**

```
num_params_after_pruning = tp.utils.count_params(model)
print( "  Params: %s => %s"%( num_params_before_pruning, num_params_after_pruning))
```


# 剪枝完以后模型的保存

**(不要用torch.save(model.state_dict(),...))**

```python
torch.save(model, 'pruning_model.pth')
```



------------------------------------------------------------------------------------
pyunmodel.py是对yolov4剪枝代码

如果你的权重是用torch.save(model.state_dict())保存的，请重新加载模型后用torch.save(model)保存[或调用save_whole_model函数]

如果你需要对单独的一个卷积进行剪枝，可以调用Conv_pruning(模型权重路径[包含网络结构图])，然后在k处修改你要剪枝的某一个卷积

如果需要对某部分进行剪枝，可以调用layer_pruning()函数，included_layers是你想要剪枝的部分[如果需要剪枝别的地方，需要修改list里面的参数，注意尽量不要对head部分剪枝]

--------------------------------------以上是剪枝部分-----------------------------------

# 代码使用：

2022.06.09更新说明：对原来项目进行了更新，将以前的诸多功能进行了合并，可以直接在终端进行传参来调用功能

所有的功能和参数都在本项目中Yolov4.py中

**图像预测：**

```python
python Yolov4.py --predict --image --weigths [your weights path]
```

**视频预测：**

```python
python Yolov4.py --predict --video --video_path [your video path, default is 0] --weigths [your weights path]
```

**FPS测试**

```python
python Yolov4.py --predict --fps
```

Note:上面的预测中权重都是包含了结构图和权值的，如果你只保存了权值，在预测的时候需要去代码中修改，将网络实例化一下。

**mAP测试**

```python
python Yolov4.py --mAP --classes_path [your classes txt file] --input_shape [416/608] --conf_thres 
```

涉及到于测试预测等有关的，**根据自己的数据集添加相应的参数**，注意注意这几个参数：weights[权重路径]，input_shape[网络输入大小]，conf_thres[置信度阈值]，classes_path【类名的txt文件，一般存放在model_data下】num_classes【类的数量】

**训练(未剪枝)：**

```python
python Yolov4.py --train --weights [your weights path] --classes_path [your classes path] --input_shape [416/608]
```

这个训练就是普通的训练，权重是**只包含权值的，不含结构图**

**微调训练(剪枝后训练)：**

```python
python Yolov4.py --train_fine --weights [your pruned weights path] --classes_path [your classes path] --input_shape [416/608]
```



**保存完整的模型：**

如果你的得到的pth文件仅含有权值不含结构图，可以调用以下函数：

```python
python Yolov4.py --save_whole_model --num_classes[your num classes] --weights [only weights path]
```

```python
eg:python Yolov4.py --save_whole_model --num_classes 80 --weights model_data/yolo4_weights.pth
```

num_classes参数是你类的数量，coco是80个类，voc20个类，根据自己类可以修改。

默认保存在model_data文件下



**对某部分进行剪枝**

```python
python Yolov4.py --pruning_model --weights model_data/whole_model.pth
```

```
针对自己要剪的目标层，需要进入prunmodel.py中自己修改，默认是model.backbone.modules()
```



**特定卷积的剪枝**

```python
python Yolov4.py --pruning_single_conv --weights model_data/whole_model.pth
```

```
具体修剪某个剪枝，需要进入该函数中进行修改，默认为backbone.conv1.conv.weight
```



该训练时剪枝后的微调训练，需要注意的是：剪枝前后的input_shape要一致；微调训练的epoch需要自己根据需要修改。

------

我尝试了一下对主干剪枝，发现精度损失严重(在剪枝40%情况下，不经过重新的训练，置信度阈值要0.1才能出结果，可能微调训练会好一些)，大家想剪哪部分可以自己去尝试，我只是把框架给搭建起来方便大家的使用，对最终的效果不保证，需要自己去炼丹。

可以训练自己的模型，剪枝后应该对模型进行一个重训练的微调提升准确率。

