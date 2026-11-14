# 文件：trainer/train.py
# 核心作用：训练入口：读 configs/config.py -> build_loader -> RetinaNet -> compute_loss -> 反向 -> 保存权重 -> TensorBoard
# 调用关系：python -m trainer.train（在 w4_retinanet_voc 目录下执行）
#           -> data/voc.py（数据）、models/retinanet.py（模型）、losses/loss.py（损失）
#           -> 产物 checkpoints/last.pth + runs/ 事件文件
# 对应参考：mmdet tools/train.py + mmengine Runner；detectron2 tools/train_net.py + DefaultTrainer
# 计划包含：build_loader()、train_one_epoch()、save_checkpoint()、main()
# 状态：占位文件，待实现
