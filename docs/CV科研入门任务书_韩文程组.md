# 计算机视觉科研入门任务书

**课题组：** 韩文程课题组 · 山东大学  
**适用对象：** 新进研究生（硕士/博士）  
**总周期：** 约 8 周（2 个月）  
**核心理念：** 论文阅读 × 代码复现 × 写作训练 × AI 辅助 四位一体

---

## 一、任务总览

| 阶段  | 时间      | 核心目标                            | 交付物                |
| --- | ------- | ------------------------------- | ------------------ |
| 突击期 | 第 1 周   | 环境搭建 + PyTorch 上手 + Vibe Coding | 环境就绪 + CIFAR-10 脚本 |
| 阅读期 | 第 2-3 周 | 论文系统阅读（AI 辅助 + 自精读结合）           | 论文笔记 × 22 + 流派对比笔记 |
| 实战期 | 第 4-5 周 | 集中复现一个 mini 项目                  | 可跑通训练的完整代码仓库       |
| 产出期 | 第 6-8 周 | Mini Survey + 方向确定 + 开题         | 综述 + 开题报告 + 答辩 PPT |

---

## 二、具体任务

### 第 1 周：突击筑基

#### Day 1-2：开发环境 & 工具链

- [ ] Linux 基础：命令行、SSH、tmux
- [ ] conda 虚拟环境 + PyTorch 安装（CUDA）
- [ ] 算力可用（校内 / AutoDL / Colab）
- [ ] **Git/GitHub**：学会 clone / commit / push / pull，从第一天起每天至少 commit 一次
- [ ] **Markdown + LaTeX (Overleaf)**：笔记用 Markdown，论文用 LaTeX
- [ ] **Zotero / Mendeley**：建立个人文献库
- [ ] **VSCode / Cursor** 配置顺手

#### Day 3-5：PyTorch 速通（CIFAR-10 分类）

自己写完整流程：

- [ ] Dataset / DataLoader
- [ ] Model 定义（简单 CNN → ResNet-18）
- [ ] Training loop（checkpoint + early stopping + LR 衰减）
- [ ] 评估（准确率 + 混淆矩阵）
- [ ] TensorBoard 可视化

> 参考：pytorch.org/tutorials、d2l.ai

#### Day 6-7：Vibe Coding & 实验管理入门

- [ ] **Cursor / Windsurf / Trae** 选一个，学会用 AI 写代码 + 调试 bug
  - 原则：AI 帮你写，但你能读懂每一行
- [ ] **WandB / TensorBoard** 跑第一个实验，记录超参数
- [ ] **实验日志**：以后每次实验记录 config + 曲线 + 结果截图 + 结论
- [ ] **随机种子固定**：``torch.manual_seed(42)`` 写入模板
- [ ] **数学短板自检**：线性代数（SVD/PCA）、概率论（贝叶斯/高斯）、优化（SGD/Adam）——哪里不会补哪里，不要求一步到位

**W1 交付物：** GitHub 仓库 + CIFAR-10 训练脚本 + 环境配置文档

---

### 第 2-3 周：论文系统阅读

**阅读方法说明：**

| 阅读方式          | 说明                                   | 时长         |
| ------------- | ------------------------------------ | ---------- |
| 🤖 **AI 辅助读** | 用 LLM 总结核心 idea → 精读关键图表/公式/实验 → 写笔记 | 约 0.5-1h/篇 |
| 📖 **自己精读**   | 从头到尾完整读，理解推导细节、设计动机、消融实验逻辑           | 约 2-3h/篇   |

**标记说明：**

- 🔴 **必须自精读** — 理解不透后面全崩，必须自己读通
- 🟢 **AI 辅助可读** — LLM 帮你搭骨架，你再精读关键血肉

**笔记格式（每篇必写）：**

```
1. 一句话说清：解决什么问题？怎么解决的？
2. 核心图/公式（手画或截图）
3. 关键实验数据
4. 【精读篇专属】我的思考：局限性？边界条件？消融实验为什么选这些？
```

---

#### 第 2 周：骨干网络 & 网络增强模块

**一条主线 + 四个分支**：

```text
主线：怎么让网络"更深、更强"
  ① AlexNet(2012)  证明"深 + GPU + ReLU + Dropout"能赢 → 但没说清"深到底靠什么"
        ↓ 提出问题：结构怎么搭最划算？
  ② VGG(2014)      答案：全部用 3×3 小核堆叠，深度优先 → 但堆到 20 层以上就训不动
        ↓ 提出问题：深了为什么训不动？
  ③ BN(2015)       答案：每层归一化，训练稳、可用大学习率 → 但堆到 56 层，训练误差反而变大
        ↓ 提出问题：这不是过拟合，是"退化"
  ④ ResNet(2015)🔴  答案：残差连接 F(x)+x，让"更深至少不更差" → 成为后续一切网络的底座
        ↓ 有了稳定底座，开始"在 ResNet 上做加法"，四个分支：
  ⑤ ResNeXt        加"宽度/分组"维度（cardinality）
  ⑥ Dilated Conv   加"感受野"维度（不降分辨率）
  ⑦ DeformConv🔴   加"采样几何"维度（卷积核会变形）
  ⑧ FPN            加"多尺度"维度（融合深浅特征）→ 直接服务于检测
  ⑨ SENet          加"通道注意力"维度（自动给通道加权）
```

| #   | 论文                                                           | 阅读方式       | 阅读要点                                                          |
| --- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------- |
| 1   | **AlexNet** — ImageNet Classification with Deep CNNs         | 🟢 AI 辅助   | ReLU、Dropout、数据增强，历史意义                                        |
| 2   | **VGG** — Very Deep Convolutional Networks                   | 🟢 AI 辅助   | 小卷积核堆叠哲学，深度 vs 性能                                             |
| 3   | **Batch Normalization** — Accelerating Deep Network Training | 🟢 AI 辅助   | 归一化原理、训练稳定性 **---一种卷积模块**                                     |
| 4   | 🔴 **ResNet** — Deep Residual Learning for Image Recognition | 🔴 **自精读** | **CV 里程碑**——残差连接解决退化问题，后续一切网络的基石 **---一种结构+基础+瓶颈块--残差块（能深度）** |
| 5   | **ResNeXt / Wide ResNet**                                    | 🟢 AI 辅助   | Group conv、宽度 vs 深度 trade-off **---一种卷积模块（分支-分组卷积）**          |
| 6   | **Dilated Convolution** — Multi-Scale Context Aggregation    | 🟢 AI 辅助   | 膨胀卷积扩大感受野 **---一种卷积模块（视野）（主要解决语义分割问题（大感受野 + 不丢分辨率”））**        |
| 7   | 🔴 **Deformable Conv v1/v2** — Deformable ConvNets           | 🔴 **自精读** | 空间自适应卷积，理解"可变形"的数学和实现 **---一种卷积模块                             |
| 8   | **FPN** — Feature Pyramid Networks                           | 🟢 AI 辅助   | 多尺度特征融合，后续检测分割标配 **---一种结构（拓扑）--用于目标检测骨干网络**                  |
| 9   | **SENet** — Squeeze-and-Excitation Networks                  | 🟢 AI 辅助   | Channel attention 经典实现 **---一种卷积模块（通道-特征权重）**                 |

| 网络                         | 核心痛点                            | 核心创新                                         | 核心算子 / 模块特性                                                                           |
| -------------------------- | ------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------- |
| ResNet                     | 深度增加网络退化                        | 残差 shortcut 跳跃连接                             | BasicBlock/Bottleneck；让极深网络可以训练                                                       |
| Dilated Convolution 空洞卷积   | 分割任务想要大感受野，就要牺牲分辨率              | 空洞卷积，不降低分辨率扩大感受野                             | 空洞卷积 + 上下文模块；采样网格固定规整；扩大感受野同时保留特征分辨率                                                  |
| Deformable ConvNets DCN v1 | 卷积 / RoI 池化采样网格固定，不能适配物体形变姿态    | 学习稠密动态 2D 偏移量                                | 可变形卷积、可变形 RoI 池化；采样点位置动态偏移贴合物体轮廓；少量额外计算开销                                             |
| FPN 特征金字塔网络                | 图像金字塔速度慢；CNN 底层高层语义鸿沟大，小物体检测效果差 | 单图输入，自下而上 + 自上而下 + 侧向连接，构建每层都具备强语义的多尺度金字塔    | 侧向 1×1 卷积、上采样逐元素融合、跨层共享检测头；多用于检测分割，解决多尺度问题                                            |
| SENet SE 模块                | 卷积通道依赖隐式、局限局部感受野，缺少显式通道筛选机制     | Squeeze‑Excitation 通道注意力；利用全局信息动态重标定各个通道特征响应 | 全局平均池化 (Squeeze)、瓶颈两层全连接 + Sigmoid 门控 (Excitation)、通道逐通道缩放；轻量即插即用，动态给通道打分增强有用特征抑制无用特征 |

> ✅ **第 2 周末小任务：** 用 PyTorch 实现 ResNet 的 BasicBlock / Bottleneck 和 DeformConv 模块，写独立可跑 demo。

---

#### 第 3 周：目标检测 & Transformer 新范式---带正负样本（物体和背景）

| #   | 论文                                                             | 阅读方式       | 阅读要点                                                                 |
| --- | -------------------------------------------------------------- | ---------- | -------------------------------------------------------------------- |
| 10  | 🔴 **Faster R-CNN** — Towards Real-Time Object Detection       | 🔴 **自精读** | **两阶段检测开山**——RPN 候选框+ ROI Pooling + 端到端训练ResNet backbone（似乎是四步交替训练吧） |
| 11  | **Cascade R-CNN** — High Quality Object Detection              | 🟢 AI 辅助   | 级联 IoU 阈值提升质量**（与传统两阶段FPN（只是在Faster基础上改一点结构）/Faster R-CNN）**         |
| 12  | 🔴 **YOLOv1** — You Only Look Once                             | 🔴 **自精读** | **单阶段回归思想的开端**，理解"把检测当成回归"**（模型读取整张图片全局上下文、公开 Fast‑YOLO 超轻量版本）**     |
| 13  | **YOLOv3** — An Incremental Improvement                        | 🟢 AI 辅助   | 多尺度预测（FPN） + Darknet-53 + 工程化范式（可多标签分类）                              |
| 14  | **YOLOv8**                                                     | 🟢 AI 辅助   | ultralytics 框架的现代化设计                                                 |
| 15  | 🔴 **Focal Loss / RetinaNet** — Focal Loss for Dense Detection | 🔴 **自精读** | **单阶段不平衡核心方案**，理解 loss 设计如何影响检测（**单阶段精度差问题，loss对样本的权重）**             |
| 16  | **CornerNet / CenterNet**                                      | 🟢 AI 辅助   | Anchor-free 检测的早期探索                                                  |
| 17  | 🔴 **Attention Is All You Need (Transformer)**                 | 🔴 **自精读** | **NLP→CV 的桥梁**——self-attention、多头注意力、位置编码                            |
| 18  | 🔴 **ViT** — An Image is Worth 16×16 Words                     | 🔴 **自精读** | **CV 范式转换**——纯 Transformer 做分类                                       |
| 19  | 🔴 **DETR** — Detection with Transformers                      | 🔴 **自精读** | **端到端检测**——set prediction + bipartite matching                       |
| 20  | **Deformable DETR**                                            | 🟢 AI 辅助   | 用可变形注意力解决 DETR 收敛慢问题                                                 |
| 21  | **MAE** — Masked Autoencoders Are Scalable Vision Learners     | 🟢 AI 辅助   | 自监督学习在 ViT 上的突破                                                      |
| 22  | **Mamba / Vision Mamba (VMamba)**                              | 🟢 AI 辅助   | 状态空间模型替代注意力，线性复杂度新方向                                                 |

> ✅ **第 3 周末小任务：** 用 HuggingFace ``transformers`` 跑通 ViT + DETR demo，写一篇对比笔记：**Conv vs Transformer vs Mamba**（核心思想、优劣势、适合场景）

> **W2+W3 共交付：** 论文笔记 × 22 篇 + 1 篇流派对比笔记 + ResNet/DeformConv 核心模块复现 demo

---

### 第 4-5 周：集中复现项目

**二选一，** 明确目标，控制在 2 周内跑通：

| 选项                | 内容                                               | 预期成果              |
| ----------------- | ------------------------------------------------ | ----------------- |
| **简化版 RetinaNet** | ResNet-18 + FPN + Focal Loss + 在 VOC subset 上训练  | mAP > 40% @VOC    |
| **简化版 DETR**      | ResNet-18 + 2-layer Transformer + Hungarian loss | 在 COCO subset 上收敛 |

**规范要求：**

- 先写 README 说清设计思路
- 目录结构：``config/``, ``models/``, ``trainer/``, ``utils/``
- 全程用 WandB / TensorBoard 记录
- **实验管理规范落实：** 每次实验 seed 固定、提交 commit + tag、commit message 写清楚改了啥结果如何
- **最早第 1 周末即可启动 Code Review：** 同学间 PR 互审，至少一人 approve 才能合并

**参考框架：** mmdetection / detectron2（先读明白设计思路，再自己写）

---

### 第 6 周：探索 + 写作

#### 6.1 半开放探索（5 天）

基于复现项目做一个小改进：

- [ ] 换 backbone（ResNet-18 → ResNet-50 / ConvNeXt-Tiny）

- [ ] 或改一个模块（FPN → BiFPN；Focal Loss → Varifocal Loss）

- [ ] 或切一个不同数据集看看泛化

- 对比实验 + 写分析

- 如果改进有效，这就是你的第一个科研亮点

#### 6.2 Mini Survey 写作（周末动笔）

选择一个细分方向，写 2000-3000 字综述：

- "基于 FPN 的多尺度特征融合进展"
- "目标检测中标签分配策略的演进"
- "从卷积到状态空间模型：视觉骨干网络变迁"
- 按学术论文格式（Introduction → 方法分类 → 对比分析 → 未来方向）

---

### 第 7 周：方向确定 + 开题报告

- [ ] 基于复现心得 + Mini Survey + 和韩老师讨论，确定研究方向（**至少讨论 2 轮**）
- [ ] 3 句话说清：研究问题是什么？为什么重要？初步思路是什么？
- [ ] 写出开题报告：问题定义 + 相关工作 + 初步方法 + 实验计划 + 时间表
- [ ] 学会画科研流程图（draw.io / Excalidraw）和结果图（Matplotlib / Seaborn）

**方法论提示（贯穿全程）：**

- 读论文时问：**"它在什么场景下会崩？"**
- 看实验结果问：**"Table 为什么选这几个 ablation？"**
- 复现时问：**"原论文有没有隐瞒细节？"**
- 每周五问自己：**"这周最重要的一个 insight 是什么？"** — 3 句话发群里

---

### 第 8 周：开题答辩

- [ ] 开题报告终稿（导师反馈后修改）
- [ ] 开题答辩 PPT（15 分钟讲清问题 + 背景 + 方法 + 计划）
- [ ] 组内预讲一遍，收集反馈

---

## 三、研究方向参考

不要着急定方向，第 1-3 周广泛读，第 4-5 周动手做，自然会有感觉：

| 方向           | 说明                       | 建议从哪切入         |
| ------------ | ------------------------ | -------------- |
| **目标检测**     | 经典赛道，工业需求大               | 从 YOLO 复现切入    |
| **视觉基础模型**   | ViT/Foundation Model/自监督 | 从 ViT + MAE 切入 |
| **轻量化/边缘部署** | 模型压缩、量化、蒸馏               | 从 MobileNet 切入 |
| **多模态/VLM**  | LLaVA/Qwen-VL            | 需要先打好单模态基础     |
| **时序/视频理解**  | 视频检测、行为识别                | 先吃透单帧检测        |

---

## 四、里程碑激励

| Milestone             | 条件                     | 意义     |
| --------------------- | ---------------------- | ------ |
| 🏅 **第一个 GitHub 仓库**  | CIFAR-10 训练脚本上传        | 代码能力认证 |
| 🏅 **第一篇论文笔记**        | ResNet 笔记给导师 review 通过 | 阅读能力认证 |
| 🏅 **第一次 PR approve** | 同学 Code Review 通过      | 协作能力认证 |
| 🏅 **第一次完整复现**        | 项目跑通训练并获得 baseline     | 工程能力认证 |
| 🏅 **第一次组会汇报**        | 完整讲 15 分钟              | 表达能力认证 |
| 🏅 **Mini Survey 完稿** | 获得导师反馈                 | 写作能力认证 |

---

## 五、资源清单

### 课程

- CS231n — [cs231n.stanford.edu](http://cs231n.stanford.edu/)
- d2l.ai (动手学深度学习) — [d2l.ai](https://d2l.ai/)

### 代码库

- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- [HuggingFace transformers](https://github.com/huggingface/transformers)
- [timm](https://github.com/huggingface/pytorch-image-models)

### 论文检索

- Google Scholar + Semantic Scholar + PapersWithCode + arXiv

---

## 六、考核标准

| 维度          | 占比  | 考核方式                  |
| ----------- | --- | --------------------- |
| 论文阅读深度      | 25% | 笔记质量 + 组会提问           |
| 代码实现能力      | 25% | 复现项目完成度 + 代码规范性       |
| AI 辅助能力     | 15% | 是否用 AI 加速了学习，而非替代思考   |
| 写作与表达       | 15% | Mini Survey + 组会/开题表达 |
| 主动性 & 批判性思考 | 20% | 提问质量、主动探索、质疑精神        |

---

> **给学生的寄语：**
> 科研不是背论文，不是调参数。科研是——
> **看懂别人的路，走自己的路。**
> 这份任务书不是枷锁，而是地图。标记好的路你自己走，没标记的路你去发现。

---

*任务书版本：v2.0（8 周精简版）· 2026 年 6 月 · 韩文程*
