# GEFM Ubuntu 20.04 环境配置指南

## 📋 系统要求

- ✅ Ubuntu 20.04
- ✅ CUDA 11.6 / 11.8 / 12.1（推荐 **11.8**）
- ✅ Python 3.9+
- ✅ NVIDIA GPU（推荐 8GB+ 显存）

---

## 🚀 方法一：自动安装（推荐）

### 一键安装脚本

```bash
# 1. 复制 GEFM 到 Ubuntu（如果还没有）
# 从 WSL 复制：
cp -r /home/user/GEFM ~/GEFM

# 2. 复制安装脚本
cp /home/user/setup_gefm_ubuntu.sh ~/setup_gefm_ubuntu.sh

# 3. 赋予执行权限
chmod +x ~/setup_gefm_ubuntu.sh

# 4. 运行安装脚本
bash ~/setup_gefm_ubuntu.sh
```

安装过程大约需要 **20-30 分钟**，取决于网速。

脚本会自动：
- ✅ 检查 CUDA 版本
- ✅ 安装系统依赖
- ✅ 创建 Python 环境（conda 或 venv）
- ✅ 安装 PyTorch + CUDA
- ✅ 安装 VLFM 依赖
- ✅ 安装 GEFM 依赖
- ✅ （可选）安装 Habitat-Sim
- ✅ （可选）安装 Ollama 本地 LLM
- ✅ 运行测试

---

## 🔧 方法二：手动安装（逐步）

### 步骤 1：检查 CUDA 版本

```bash
# 查看已安装的 CUDA 版本
nvcc --version

# 查看可用的 CUDA 版本
ls /usr/local/ | grep cuda
```

**推荐使用 CUDA 11.8**（PyTorch 1.12.1 最佳兼容）

设置 CUDA 环境变量：

```bash
# 添加到 ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 重新加载
source ~/.bashrc

# 验证
nvcc --version
```

---

### 步骤 2：安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip
```

---

### 步骤 3：创建 Python 环境

#### 选项 A：使用 Conda（推荐）

```bash
# 如果没有 Conda，先安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 重启终端

# 创建环境
conda create -n gefm python=3.9 -y
conda activate gefm
```

#### 选项 B：使用 venv

```bash
# 创建虚拟环境
python3.9 -m venv ~/gefm_env

# 激活
source ~/gefm_env/bin/activate
```

---

### 步骤 4：安装 PyTorch（CUDA 11.8）

```bash
# 确保环境已激活
pip install --upgrade pip setuptools wheel

# 安装 PyTorch 1.12.1 + CUDA 11.3（兼容 CUDA 11.8）
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 验证 CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**期望输出**：
```
PyTorch: 1.12.1+cu113
CUDA available: True
```

---

### 步骤 5：安装 VLFM 依赖

```bash
# GroundingDINO
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67

# LAVIS (BLIP-2)
pip install salesforce-lavis==1.0.2

# 核心依赖
pip install numpy==1.26.4 \
    flask>=2.3.2 \
    seaborn>=0.12.2 \
    open3d>=0.17.0 \
    transformers==4.26.0 \
    timm==0.4.12 \
    opencv-python==4.5.5.64

# 额外模块
pip install git+https://github.com/naokiyokoyama/frontier_exploration.git
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install git+https://github.com/naokiyokoyama/depth_camera_filtering.git
```

---

### 步骤 6：安装 Habitat（可选，用于仿真）

```bash
# Habitat-Sim
pip install habitat-sim==0.2.4

# Habitat-Lab 和 Baselines
pip install habitat-lab==0.2.420230405
pip install habitat-baselines==0.2.420230405
```

**注意**：Habitat-Sim 编译可能需要 10-20 分钟。

---

### 步骤 7：安装 GEFM 依赖

```bash
# LLM 后端
pip install openai>=1.0.0      # OpenAI GPT
pip install anthropic>=0.25.0  # Anthropic Claude
pip install ollama>=0.1.0      # 本地 LLM

# 场景图和可视化
pip install networkx>=3.0
pip install matplotlib>=3.7.0

# 实验追踪
pip install wandb>=0.16.0
pip install tensorboard>=2.15.0

# 开发工具
pip install pytest>=7.4.0
pip install black>=23.0.0
pip install ruff>=0.1.0
```

---

### 步骤 8：安装 GEFM 包

```bash
# 复制 GEFM 目录到用户目录（如果还没有）
cp -r /home/user/GEFM ~/GEFM

# 安装
cd ~/GEFM
pip install -e .
```

---

### 步骤 9：安装 Ollama（本地 LLM，可选）

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动 Ollama 服务（在后台）
ollama serve &

# 下载 Llama3 模型（约 4.7 GB）
ollama pull llama3

# 测试
ollama run llama3 "Hello, how are you?"
```

---

### 步骤 10：配置 OpenAI API（可选）

如果使用 OpenAI GPT：

```bash
# 添加到 ~/.bashrc
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

获取 API Key：https://platform.openai.com/api-keys

---

## ✅ 验证安装

### 1. 测试 GEFM 组件

```bash
cd ~/GEFM
python test/test_gefm_components.py
```

**期望输出**：
```
============================================================
Testing SceneGraphMap...
============================================================
✓ SceneGraphMap created successfully
✓ Scene graph updated: 2 nodes created
✓ Semantic context: 'near chair (0.0m, conf=0.90), table (0.0m, conf=0.85)'
...

✅ SceneGraphMap tests PASSED
✅ LLMReasoner (Mock) tests PASSED
⚠  LLMReasoner (Real) tests SKIPPED
✅ GEFMPolicy Import tests PASSED

Total: 3 passed, 0 failed, 1 skipped
```

### 2. 测试 PyTorch CUDA

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

### 3. 测试 LLM（如果安装了 Ollama）

```bash
python -c "
from vlfm.vlm.llm_reasoner import LLMReasoner

reasoner = LLMReasoner(model_name='llama3', backend='ollama')
response = reasoner.query('Name 3 common objects in a kitchen')
print(f'LLM Response: {response}')
"
```

---

## 📁 目录结构

安装后的目录结构：

```
~/GEFM/                        # GEFM 主目录
├── vlfm/                      # 核心代码
├── config/                    # 配置文件
├── test/                      # 测试
└── docs/                      # 文档

~/gefm_env/                    # 虚拟环境（如果用 venv）
或
~/.conda/envs/gefm/            # Conda 环境（如果用 conda）

~/activate_gefm.sh             # 激活脚本（自动生成）
```

---

## 🎯 快速启动

创建激活脚本（方便每次使用）：

```bash
cat > ~/activate_gefm.sh <<'EOF'
#!/bin/bash
# 激活 GEFM 环境

# 激活 Python 环境（根据您的选择修改）
conda activate gefm
# 或者: source ~/gefm_env/bin/activate

# 设置 CUDA 环境变量
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# （可选）设置 OpenAI API Key
# export OPENAI_API_KEY="your-key"

echo "✓ GEFM environment activated!"
echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 切换到 GEFM 目录
cd ~/GEFM
EOF

chmod +x ~/activate_gefm.sh
```

每次使用时：

```bash
source ~/activate_gefm.sh
```

---

## 🐛 常见问题

### Q1: CUDA out of memory

**原因**：GPU 显存不足

**解决**：
```bash
# 查看 GPU 使用情况
nvidia-smi

# 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 或减少 batch size / 使用 CPU
```

### Q2: "No module named 'torch'"

**原因**：环境未激活或 PyTorch 未安装

**解决**：
```bash
# 激活环境
conda activate gefm  # 或 source ~/gefm_env/bin/activate

# 重新安装 PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

### Q3: "ImportError: libcudart.so.11.0: cannot open shared object file"

**原因**：CUDA 库路径未设置

**解决**：
```bash
# 添加到 ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

### Q4: GroundingDINO 安装失败

**原因**：编译依赖缺失

**解决**：
```bash
sudo apt-get install -y ninja-build
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### Q5: Ollama 连接失败

**原因**：Ollama 服务未启动

**解决**：
```bash
# 启动 Ollama 服务
ollama serve

# 或在后台运行
nohup ollama serve > ~/ollama.log 2>&1 &
```

---

## 📊 性能优化

### 1. 使用更快的 CUDA 版本

如果有 CUDA 12.1，可以尝试：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 启用混合精度

在配置中添加：
```yaml
use_mixed_precision: true
```

### 3. 减少 LLM 调用频率

修改 `config/experiments/gefm_objectnav_hm3d.yaml`：
```yaml
gefm:
  reasoning_interval: 10  # 从 5 增加到 10
```

---

## 📚 下一步

1. **阅读文档**：
   ```bash
   cd ~/GEFM
   cat README.md
   cat QUICKSTART.md
   ```

2. **运行测试**：
   ```bash
   python test/test_gefm_components.py
   ```

3. **配置 LLM**：
   - 使用 Ollama（免费）：已安装
   - 使用 OpenAI：设置 `OPENAI_API_KEY`

4. **开始实验**：
   ```bash
   # 查看实验指南
   cat docs/GEFM_RESEARCH_SUMMARY.md
   ```

---

## ✨ 完成！

您的 GEFM 环境已经配置完成！

**激活环境**：
```bash
source ~/activate_gefm.sh
```

**开始使用**：
```bash
cd ~/GEFM
python test/test_gefm_components.py
```

祝您研究顺利！🚀
