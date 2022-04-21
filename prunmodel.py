import torch
import torch.nn as nn
import torch_pruning as tp
from nets.yolo import YoloBody
# 剪枝的时候不能只减通道，也要有图结构，不然无法加载
# 如果权重没有保存图结构，即训练完是用torch.save(model.state_dict()，'.pth')保存的,可以将下面的注释掉


def save_whole_model(weights_path):

    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model = YoloBody(anchors_mask, 80)
    model_dict = model.state_dict()
    predtrained_dict = torch.load(weights_path)
    predtrained_dict = {k: v for k, v in predtrained_dict.items() if predtrained_dict.keys()==model_dict.keys()}
    model_dict.update(predtrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    torch.save(model, './model_data/whole_model.pth')

def Conv_pruning(whole_model_weights):
    model = torch.load(whole_model_weights)  # 模型的加载
    model_dict = model.state_dict()  # 获取模型的字典

    # -------------------特定卷积的剪枝--------------------
    #                  比如要剪枝以下卷积
    #              'backbone.conv1.conv.weight'
    # --------------------------------------------------
    for k, v in model_dict.items():
        if k == 'backbone.conv1.conv.weight':  # 对主干网络中的backone.conv1.conv权重进行剪枝
            # 1. setup strategy (L1 Norm)
            strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()

            # 2. build layer dependency
            DG = tp.DependencyGraph()
            DG.build_dependency(model, example_inputs=torch.randn(1, 3, 416, 416))  # 这里可以改成608大小

            # 3. get a pruning plan from the dependency graph.
            pruning_idxs = strategy(v, amount=0.4)  # or manually selected pruning_idxs=[2, 6, 9, ...]
            pruning_plan = DG.get_pruning_plan(model.backbone.conv1.conv, tp.prune_conv, idxs=pruning_idxs)
            print(pruning_plan)

            # 4. execute this plan (prune the model)
            pruning_plan.exec()
    torch.save(model, './model_data/Conv_pruning.pth')

def layer_pruning(whole_model_weights):
    model = torch.load(whole_model_weights)  # 模型的加载
    # -----------------对整个模型的剪枝--------------------
    strategy = tp.strategy.L1Strategy()
    DG = tp.DependencyGraph()
    DG = DG.build_dependency(model, example_inputs=torch.randn(1, 3, 608, 608))

    num_params_before_pruning = tp.utils.count_params(model)

    # 可以对照yolov4结构进行剪枝
    included_layers = list(model.conv2.modules())  # 对SPP网络后出来的三个卷积剪枝

    # 尽量不要对头部剪枝
    #excluded_layers = list((model.yolo_head1[-1].modules(),model.yolo_head2[-1].modules(),model.yolo_head3[-1].modules()))

    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m in included_layers:
            pruning_plan = DG.get_pruning_plan(m,tp.prune_conv, idxs=strategy(m.weight, amount=0.4))
            print(pruning_plan)
            # 执行剪枝
            pruning_plan.exec()
    # 获得剪枝以后的参数量
    num_params_after_pruning = tp.utils.count_params(model)
    # 输出一下剪枝前后的参数量
    print( "  Params: %s => %s"%( num_params_before_pruning, num_params_after_pruning))
    # 剪枝完以后模型的保存(不要用torch.save(model.state_dict(),...))
    torch.save(model, 'model_data/layer_pruning.pth')

if __name__ == '__main__':
    weights_path = r'./model_data/yolo4_weights.pth'  # 只有权重
    whole_model_weights = r'./model_data/whole_model.pth'  # 完整的模型

    # --------------保存完整的模型---------------------
    save_whole_model(weights_path)
    # ----------------------------------------------

    # ----------------特定卷积的剪枝-------------------
    Conv_pruning(whole_model_weights)

    # ---------------对某部分进行剪枝------------------
    layer_pruning(whole_model_weights)
