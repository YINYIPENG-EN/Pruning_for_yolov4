# 原yolov4代码参考b站up主：https://www.bilibili.com/video/BV1Q54y1D7vj?spm_id_from=333.999.0.0
# 通道剪枝参考论文：Pruning Filters for Efficient ConvNets
# 说明：本项目是参考以上资料展开的，效果方面不予保证，针对自己的数据集和应用场景需要自己调整
# 项目作者：Yin Yipeng
# 联系方式：邮箱：15930920977@163.com 微信：y24065939s
import argparse
from train import Train
from train_fine import Train_fine
from predict_pruning import Predict
from get_map import Get_mAP
from prunmodel import layer_pruning, Conv_pruning, save_whole_model
if __name__ == "__main__":
    pargse = argparse.ArgumentParser()
    pargse.add_argument('--model', type=str, default='yolov4', help='choose model')
    pargse.add_argument('--cuda', action='store_true', default=True, help='Use Cuda')
    pargse.add_argument('--output', type=str, default='', help='output file path')
    pargse.add_argument('--conf_thres', type=float, default=0.6, help='detection confidence threshold')
    pargse.add_argument('--nms_thres', type=float, default=0.5, help='detection iou threshold')
    pargse.add_argument('--weights', type=str, default='', help='weights path')
    pargse.add_argument('--predict', action='store_true', default=False, help='predict')
    pargse.add_argument('--image', action='store_true', default=False, help='predict image')
    pargse.add_argument('--video', action='store_true', default=False, help='predict video')
    pargse.add_argument('--video_path', type=str, default='0', help='video path, default is 0')
    pargse.add_argument('--fps', action='store_true', default=False, help='FPS test')
    pargse.add_argument('--dir_predict', action='store_true', default=False, help='dir predict')
    pargse.add_argument('--classes_path', type=str, default='model_data/coco_classes.txt', help='classes txt file path')
    pargse.add_argument('--input_shape', type=int, default='608', help='input shape')
    pargse.add_argument('--mAP', action='store_true', default=False, help='get mAP')

    pargse.add_argument('--train', action='store_true', default=False, help='model train')
    pargse.add_argument('--train_fine', action='store_true', default=False, help='pruned model train')
    pargse.add_argument('--batch_size', type=int, default=4, help='batch size')
    pargse.add_argument('--Init_lr', type=float, default=1e-2, help='init learning rate')
    pargse.add_argument('--focal_loss', action='store_true', default=False,help='focal loss')
    pargse.add_argument('--save_period', type=int, default=1, help='save model period')

    pargse.add_argument('--save_whole_model', action='store_true', default=False, help='save whole model')
    pargse.add_argument('--num_classes', type=int, default=80, help='num classes')
    pargse.add_argument('--pruning_model', action='store_true', default=False, help='model layer pruning')
    pargse.add_argument('--pruning_single_conv', action='store_true', default=False, help='pruning single conv')



    opt = pargse.parse_args()

    if opt.predict:
        # if you predict image,you should input : python Yolov4.py --predict --image --weigths [your weights path]
        # also if you want to predict video, input:python Yolov4.py --predict --video --video_path
        # [your video path, default is 0] --weigths [your weights path]
        # Note:预测的时候权值pth文件都是包含权值和结构图的，如果不需要，自己可以在代码里修改
        Predict(opt)

    if opt.train:
        # python Yolov4.py --train --weights [your weights path] --classes_path [your classes path] --input_shape [
        # 416/608]
        Train(opt)  # 该训练是未剪枝之前训练  加载的权重是只有权值不含网络结构!!

    if opt.train_fine:
        # python Yolov4.py --train_fine --weights [your pruned weights path] --classes_path [your classes path]
        # --input_shape [416/608]
        # Note：剪枝前后模型的input_shape要一致
        Train_fine(opt)  # 该训练是剪枝后的微调训练，需要自己修改训练多少epoch，默认和未剪枝epoch以及训练参数一样

    if opt.save_whole_model:
        # 保存完整的权重
        # python Yolov4.py --save_whole_model --num_classes[your num classes] --weights [only weights path]
        # eg:python Yolov4.py --save_whole_model --num_classes 80 --weights model_data/yolo4_weights.pth
        save_whole_model(opt.weights, opt.num_classes)

    if opt.get_mAP:
        # 详细的使用说明可进行函数查看
        Get_mAP(opt)


    # ---------------对某部分进行剪枝------------------
    if opt.pruning_model:
        # 说明该功能可以对网络层进行剪枝
        # weights是包含结构图和权值的pth文件，即采用torch.save(model, )形式，而不是torch.save(model.state_dict(), )
        # 针对自己要剪的目标层，需要进入prunmodel.py中自己修改，默认是model.backbone.modules()
        # you can input in terminal:python Yolov4.py --pruning_model --weights model_data/whole_model.pth
        layer_pruning(opt.weights)

    # ----------------特定卷积的剪枝-------------------
    if opt.pruning_single_conv:
        # 说明：该功能可以对某一个卷积进行剪枝
        # 具体修剪某个剪枝，需要进入该函数中进行修改，默认为backbone.conv1.conv.weight
        # weights是包含结构图和权值的pth文件，即采用torch.save(model, )形式，而不是torch.save(model.state_dict(), )
        # 网络结构中所有卷积命名可以看yolov4结构.txt
        # In terminal you can input :python Yolov4.py --pruning_single_conv --weights model_data/whole_model.pth
        Conv_pruning(opt.weights)




