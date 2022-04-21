# 本项目yolov4代码参考:https://www.bilibili.com/video/BV1Q54y1D7vj?spm_id_from=333.999.0.0
# 剪枝paper参考:Pruning Filters for Efficient ConvNets

**本项目只是负责把框架搭建起来，没有进行重训练的微调或者去研究应该剪哪里比较好，需要自己去研究**

pytorch 1.7.0(低版本应该也是可以的)

torchvision 0.8.0

torch_pruning


# 安装包
pip install torch_pruning

------------------------------------------------------

# 导入包
import torch_pruning as tp

# 模型的实例化(针对已经训练好的模型)
model = torch.load('权重路径')
model.eval()
# ---------------对于非单层卷积的通道剪枝(不用看3.1)----
#                剪枝之前统计一下模型的参数量
# --------------------------------------------------------
num_params_before_pruning = tp.utils.count_params(model)

**1. setup strategy (L1 Norm) 计算每个通道的权重**
strategy = tp.strategy.L1Strategy()

**2.建立依赖图(与torch.jit很像)**
DG = tp.DependencyGraph()
DG = DG.build_dependency(model, example_inputs=torch.randn(1, 3, input_size[0], input_size[1])) # input_size是网络的输入大小

**3.分情况(1.单个卷积进行剪枝 2.层进行剪枝)**

**3.1 单个卷积(会返回要剪枝的通道索引，这个通道是索引是根据L1正则得到的)**

pruning_idxs = strategy(model.conv1.weight, amount=0.4)  # model.conv1.weigth是对特定的卷积进行剪枝,amount是剪枝率

**将根据依赖图收集所有受影响的层，将它们传播到整个图上，然后提供一个PruningPlan正确修剪模型的方法。**
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv, idxs=pruning_idxs )
pruning_plan.exec()
torch.save(model, 'pru_model.pth')

**3.2 层剪枝(需要筛选出不需要剪枝的层，比如yolo需要把头部的预测部分取出来，这个是不需要剪枝的)**
excluded_layers = list(model.model[-1].modules())
for m in model.modules():
    if isinstance(m, nn.Conv2d) and m not in excluded_layers:
        pruning_plan = DG.get_pruning_plan(m,tp.prune_conv, idxs=strategy(m.weight, amount=0.4))
        print(pruning_plan)
        # 执行剪枝
        pruning_plan.exec()
        
**如果想看一下剪枝以后的参数，可以运行：**
num_params_after_pruning = tp.utils.count_params(model)
print( "  Params: %s => %s"%( num_params_before_pruning, num_params_after_pruning))

**剪枝完以后模型的保存(不要用torch.save(model.state_dict(),...))**
torch.save(model, 'pruning_model.pth')

------------------------------------------------------------------------------------
pyunmodel.py是对yolov4剪枝代码

如果你的权重是用torch.save(model.state_dict())保存的，请重新加载模型后用torch.save(model)保存[或调用save_whole_model函数]

如果你需要对单独的一个卷积进行剪枝，可以调用Conv_pruning(模型权重路径[包含网络结构图])，然后在k处修改你要剪枝的某一个卷积

如果需要对某部分进行剪枝，可以调用layer_pruning()函数，included_layers是你想要剪枝的部分[我这里是对SPP后面的三个卷积剪枝，如果需要剪枝别的地方，需要修改list里面的参数，注意尽量不要对head部分剪枝]

--------------------------------------以上是剪枝部分-----------------------------------

预测部分，将剪枝后的权重路径填写在pruning_yolo.py中的"model_path"处，默认是coco的类[因为剪枝后的模型中已经保存了图结构，所以预测的时候不需要在实例化模型]。
然后运行predict_pruning.py 可以修改mode，'predict'是预测图像，'video'是视频[默认打开自己摄像头]
FPS测试，将mode改为fps[我的硬件是英伟达1650,cuda10.2]
对SPP网络后三个卷积层剪枝以后，FPS为18 剪枝之前是FPS15

我尝试了一下对主干剪枝，发现精度损失严重，大家想剪哪部分可以自己去尝试，我只是把框架给搭建起来方便大家的使用，对最终的效果不保证，需要自己去炼丹。
可以训练自己的模型，剪枝后应该对模型进行一个重训练的微调提升准确率，这部分代码我还没有加入进去，可以自己把剪枝后的权重放训练代码中微调一下就行。后期有时间会加入微调训练部分。

CSDN：https://blog.csdn.net/z240626191s/article/details/124326595?spm=1001.2014.3001.5502

权重链接：
链接：https://pan.baidu.com/s/1neHeyWsm_SQ-ZrFwDZUrUA 
提取码：yypn
