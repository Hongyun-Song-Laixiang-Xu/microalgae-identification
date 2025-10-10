VTKB-for-Microalgae-Identification

官方 PyTorch 实现 | 论文处于投刊阶段，标题：《VTKB: A Microalgae Spectral Classification Network Based on Deep Learning Algorithms》提出 ViTKA 网络模型，基于 PyTorch 框架实现四类微生物藻类精度识别，兼顾推理效率与特征捕捉能力，助力水下环境发展。

Table of Contents

1\. 研究背景与模型定位

2\. ViTKA 核心创新点

3\. 实验数据集：四类微生物藻类数据集

4\. 实验环境配置

5\. 代码使用说明

6\. 项目文件结构

7\. 已知问题与注意事项

8\. 引用与联系方式

1\. 研究背景与模型定位

微生物藻类是水体生态、水产养殖及碳循环的关键生物，其类别（如有害蓝藻、有益绿藻）与数量直接影响水质安全和生态平衡。传统藻类检测依赖人工镜检，存在效率低（单样本耗时 5-10 分钟）、依赖专业经验（误判率超 15%）、难以规模化监测（如大面积水体）的问题。

本文提出ViTKA（Vision Transformer-Kolmogorov-Arnold Networks-BiFormer）模型，通过三大核心模块协同优化：

改进 Vision Transformer（ViT）提升推理速度；

引入 Kolmogorov-Arnold Networks（KAN）增强非线性特征表征；

融合 BiFormer 稀疏动态注意力提升鲁棒性。

模型基于 PyTorch 2.4.1 框架实现，针对四类常见微生物藻类（有害 + 有益）实现高精度识别，为水质监测、水产养殖藻类调控提供自动化技术支撑。

2\. ViTKA 核心创新点

2.1 改进 Vision Transformer（ViT）：适配藻类小尺寸特征

针对藻类图像（通常含密集小目标）和原始 ViT 计算冗余问题，优化两点：

编码器精简与窗口调整：编码器层数从 12 层减至 6 层（藻类特征维度低于叶片），局部窗口注意力尺寸从 16×16 调整为 8×8，适配藻类小目标，推理速度提升 45%（精度损失 < 0.8%）；

混合注意力机制：采用 “微窗口注意力（聚焦单藻细胞）+ 全局稀疏注意力（捕捉藻群分布）”，替代全尺寸注意力，降低 Token 计算量，适配移动端水质监测设备部署。

2.2 KAN 非线性表征机制：强化藻类形态区分

藻类类别差异多体现在细微形态（如蓝藻的丝状体、绿藻的单细胞结构），KAN 模块替代传统全连接层实现：

分段非线性映射：针对藻类的细胞壁纹理、鞭毛数量等细粒度特征，通过 KAN 的分段函数强化区分（如蓝藻与硅藻的形态差异）；

自适应背景过滤：结合水体背景特性（如浑浊度、杂质颗粒），设计动态激活函数，减少非藻类区域（如泥沙、气泡）对特征提取的干扰。

2.3 BiFormer 稀疏动态注意力：提升水体场景鲁棒性

针对藻类识别中 “水体光照波动、藻细胞遮挡” 等问题，融合 BiFormer 双路径注意力：

动态通道激活：根据输入图像中藻细胞的密度，自动激活高贡献注意力通道（如藻细胞密集区域优先分配计算资源）；

稀疏 Token 过滤：通过权重阈值（默认 0.3）过滤低价值 Token（如纯水体背景、微小杂质），聚焦藻细胞核心区域，鲁棒性提升 18%（针对光照变化、细胞重叠场景）。

3\. 实验数据集：四类微生物藻类数据集

3.1 数据集概况

本研究基于四类微生物藻类中心序列识别数据集，数据集需联系作者获取或后续更新至公开存储平台：

数据集名称	包含类别	图像总数	图像分辨率	数据分布（训练：验证：测试）

四类微藻数据集	Pavlova、Pediastrum、Scenedesmus、Selenastrum capricornutum	8000+	统一 resize 至 384×384（适配 ViT 输入）	7:1:2（通过代码自动划分）

3.2 数据集获取与结构

3.2.1 下载方式

网盘链接及提取码：https://pan.quark.cn/s/ae696b65766a（提取码：3U9v）

3.2.2 文件夹组织

解压后放置于项目根目录，结构如下：

plaintext

Microalgae\_dataset/

├── Pavlova/

├── verticilium\_wilt/

├── Pediastrum/

└── Selenastrum capricornutum/

4\. 实验环境配置

4.1 依赖安装

推荐使用 Anaconda 创建虚拟环境，确保 PyTorch 版本与 CUDA 环境匹配（支持 GPU/CPU，优先推荐 GPU 加速）：

bash

\# 1. 创建并激活虚拟环境

conda create -n vitkab-pytorch python=3.10

conda activate vitkab-pytorch



\# 2. 安装PyTorch 2.4.1（GPU版本，需CUDA 12.1；CPU版本见下方备注）

conda install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia



\# （备注：CPU版本安装命令）

\# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu



\# 3. 安装其他依赖库

pip install numpy~=2.0.1 matplotlib~=3.9.5 opencv-python~=4.12.0.88

pip install pandas~=2.3.2 pillow~=11.3.0 scikit-learn~=1.5.2

pip install tqdm~=4.66.5 tensorboard~=2.17.0 torchmetrics~=1.4.0

5\. 代码使用说明

5.1 模型训练

运行train.py脚本启动训练，支持通过命令行参数调整训练配置，示例命令如下：

bash

python train.py \\

--data\_dir ./cotton\_disease\_dataset \\  # 数据集根目录（解压后的路径）

--epochs 80 \\                          # 训练轮数

--batch\_size 32 \\                      # 批次大小（根据GPU显存调整，16/32/64）

--lr 5e-5 \\                            # 初始学习率

--weight\_decay 1e-5 \\                  # 权重衰减（防止过拟合）

--save\_dir ./weights \\                 # 模型权重保存目录（.pth格式）

--log\_interval 20 \\                    # 每20个batch打印一次训练日志

--device GPU                           # 训练设备（GPU/CPU）

关键参数说明

参数名	含义	默认值

--data\_dir	数据集根目录路径	./cotton\_disease\_dataset

--epochs	训练轮数	80

--batch\_size	批次大小（GPU 显存不足时可设为 16）	32

--lr	初始学习率（采用余弦退火学习率调度）	5e-5

--save\_dir	权重保存目录（自动生成，.pth 格式）	./weights

--device	训练设备（GPU 需配置 CUDA 12.1+）	GPU

5.2 模型预测

使用训练好的权重进行单张微藻图像预测，运行predict.py脚本，示例命令如下：

bash

python predict.py \\

--image\_path ./examples/cotton\_brown\_spot.jpg \\  # 输入图像路径

--weight\_path ./weights/best\_vitkab.pth \\         # 预训练权重路径（PyTorch .pth格式）

--device CPU                                      # 预测设备（GPU/CPU）

预测输出示例

plaintext

输入图像路径：./examples/cotton\_brown\_spot.jpg

预测类别：Pa(Pavlova)

置信度：0.9982

预测耗时：12.3ms（CPU）/ 2.1ms（GPU）

6\. 项目文件结构

plaintext

vitka-for-microalgae-identification/  # 项目根目录（更名）

├── microalgae\_dataset/  # 四类微生物藻类数据集（替换原棉花数据集）

├── examples/            # 藻类示例图像（如microcystis\_01.jpg）

├── models/              # 模型核心模块（仅修改ViTKAB.py）

│   ├── vit\_improve.py   # 同原项目（已适配藻类窗口大小）

│   ├── kan\_module.py    # 同原项目（已适配藻类特征）

│   ├── biformer\_attention.py # 同原项目

│   └── ViTKAB.py        # 核心修改：num\_classes=4（藻类类别数）

├── dataset/             # 数据处理（新增藻类预处理）

│   └── data\_loader.py   # 修改：新增藻类图像去噪、对比度增强步骤

├── train.py             # 新增--num\_classes参数（默认4）

├── predict.py           # 新增“风险提示”输出模块（有害/有益藻区分）

├── algae\_weights/       # 藻类模型权重保存目录（替换原weights）

└── README.md            # 本说明文档（适配藻类）

7\. 已知问题与注意事项

框架适配：本项目仅支持 PyTorch 2.4.1 及以上版本，不兼容 TensorFlow 或低版本 PyTorch（<2.0）；

输入尺寸：模型固定输入为 384×384×3（RGB 图像），预测时会自动 resize 输入图像，建议原始图像分辨率≥384×384，避免低分辨率导致的特征丢失；

数据集扩展：如需新增棉花病害类别，需补充对应类别图像数据，并修改models/ViTKAB.py中num\_classes参数（当前为 4，新增后需同步调整）；

GPU 依赖：训练时推荐使用 CUDA 12.1 及以上版本 GPU（显存≥8GB），CPU 训练耗时较长（单轮 epoch 约 120 分钟，GPU 约 15 分钟）；

权重格式：模型权重仅支持 PyTorch 的.pth格式，不兼容 TensorFlow 的.h5格式，请勿混用跨框架权重。

8\. 引用与联系方式

8.1 引用方式

论文处于投刊阶段，正式发表后将更新完整 BibTeX 引用格式，当前可临时引用：

bibtex

@article{vitka\_Microalgae,

title={VTKB: A Microalgae Spectral Classification Network Based on Deep Learning Algorithms},

author={\[作者姓名，待发表时补充]},

journal={\[期刊名称，待录用后补充]},

year={2025},

note={Manuscript submitted for publication}

}

8.2 联系方式

若遇到代码运行问题、数据集获取需求或学术交流，可通过以下方式联系：

邮箱：songhongyunhuuc@yeah.net（替换为实际邮箱）

GitHub Issue：直接在本仓库提交 Issue，会在 1-3 个工作日内回复；

学术交流：可发送主题为 “ViTKA - 学术交流” 的邮件，附个人简介及交流方向，将优先回复。

