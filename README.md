# Yuri AI Core

基于BERT微调与LLM的百合作品轻重分类系统

## 功能介绍

本项目用于计算百合文学作品的"重百合隶属度"，通过多维度分析给出0-1之间的评分：
- **接近1** → 重百合倾向
- **接近0** → 轻百合倾向

### 核心功能

| 功能 | 说明 |
|------|------|
| 文本分类 | BERT模型对文本进行重/轻百合分类 |
| 对话分析 | LLM分析对话内容中的百合元素 |
| 动作分析 | LLM分析动作/心理描写中的百合元素 |
| 综合计算 | 加权融合三项指标得出最终隶属度 |
| 可视化界面 | Streamlit Web界面展示分析结果 |

## 方法概述

### 计算流程

```
输入文本 → 三路并行分析 → 加权求和 → 输出隶属度
           │
           ├── BERT推理 (权重0.6)
           ├── LLM对话分析 (权重0.3)  
           └── LLM动作分析 (权重0.1)
```

### 技术方案

1. **BERT分类器**: 基于 `chinese-roberta-wwm-ext` 微调，对清洗后文本进行二分类
2. **LLM对话分析**: 使用Kimi提取对话内容，分析百合相关对话占比
3. **LLM动作分析**: 使用Kimi提取动作/心理描写，分析百合相关描写占比
4. **加权融合**: `最终分数 = 0.6×BERT + 0.3×对话 + 0.1×动作`

## 快速使用

### 环境配置

```bash
# 1. 创建并激活conda环境
conda create -n yuri-backend python=3.10
conda activate yuri-backend

# 2. 安装依赖
pip install -r requirements.txt
```

**核心依赖** (基于yuri-backend环境)：
```
torch==2.1.1+cu121          # PyTorch (CUDA 12.1)
transformers==4.35.2        # HuggingFace模型库
streamlit==1.51.0           # Web界面
pandas==2.1.4               # 数据处理
scikit-learn==1.3.2         # 机器学习
jieba==0.42.1               # 中文分词
openai==2.7.1               # LLM API
plotly==6.5.0               # 可视化
networkx==3.3               # 图形处理
beautifulsoup4==4.14.2      # HTML解析
datasets==4.4.1             # 数据集
```

**GPU要求**: CUDA 12.1 (如需其他CUDA版本，访问 [PyTorch官网](https://pytorch.org/) 安装对应版本)

### 模型准备

1. 下载BERT模型：[HuggingFace](https://huggingface.co/yeyeye0118/BERT-Yuri-CLS-Large)
2. 放置到 `models/` 目录
3. 修改 `config.json` 中的模型路径：
   ```json
   "bert_checkpoint": "./models/checkpoint-47200"
   ```

### API配置

在 `config.json` 中填入Moonshot API密钥：
```json
"api_key": "your-api-key",
"moonshot_api_key": "your-api-key"
```

获取地址：https://platform.moonshot.cn/

### 启动应用

```bash
conda activate yuri-backend
streamlit run gui_app.py
```

浏览器访问 `http://localhost:8501`

## 使用方式

### 方式一：Web界面（推荐）

启动后在浏览器中操作：
1. 上传或选择待分析文本
2. 点击运行分析
3. 查看各维度分数和最终隶属度

### 方式二：脚本调用

```bash
# 将文本放入 txt_test/ 目录，命名格式：书号.txt
# 运行集成推理
python script/integrated_inference.py
```

结果输出到 `csv/prediction/` 和 `csv/weighted/`

### 方式三：单独模块

```bash
# 仅BERT推理
python script/model_BERT_infer.py

# 仅LLM对话分析
python script/LLM_dialogue.py

# 仅LLM动作分析
python script/LLM_verb.py
```

## 项目结构

```
├── gui_app.py              # Web界面入口
├── config.json             # 配置文件
├── requirements.txt        # 依赖列表
├── script/
│   ├── integrated_inference.py  # 集成推理（推荐）
│   ├── model_BERT_infer.py      # BERT推理
│   ├── LLM_dialogue.py          # 对话分析
│   ├── LLM_verb.py              # 动作分析
│   ├── clean_txt.py             # 文本清洗
│   └── dialogue_cut.py          # 对话提取
├── models/                 # 存放BERT模型
├── txt_test/               # 输入文本目录
└── csv/                    # 输出结果目录
```

## 配置说明

`config.json` 主要配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `bert_checkpoint` | BERT模型路径 | `./models/checkpoint-47200` |
| `bert_max_len` | 最大文本长度 | 512 |
| `bert_batch_size` | 批处理大小 | 8 |
| `api_key` | Moonshot API密钥 | - |
| `LLM_threads_dialogue` | 对话分析线程数 | 7 |
| `LLM_threads_verb` | 动作分析线程数 | 20 |

## 许可证

MIT License
