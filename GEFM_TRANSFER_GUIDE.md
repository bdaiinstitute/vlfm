# 将 GEFM 传输到 Windows 桌面的方法

## 📦 压缩包已创建

✅ **GEFM.zip** (51 KB) - Windows 推荐
✅ **GEFM.tar.gz** (109 KB) - Linux/Mac

文件位置：`/home/user/GEFM.zip`

---

## 🚀 传输方法

### 方法 1️⃣: 使用 SCP（如果是远程服务器）

在您的 **Windows PowerShell** 中运行：

```powershell
# 切换到桌面目录
cd C:\Users\yangz\Desktop

# 使用 SCP 下载（替换为实际服务器信息）
scp username@server-ip:/home/user/GEFM.zip .

# 解压
Expand-Archive -Path GEFM.zip -DestinationPath .
```

---

### 方法 2️⃣: 使用 Git Clone（推荐，如果已推送到 GitHub）

```powershell
# 在桌面打开 PowerShell
cd C:\Users\yangz\Desktop

# 克隆仓库
git clone https://github.com/YOUR_USERNAME/GEFM.git

# 完成！
cd GEFM
```

---

### 方法 3️⃣: 手动下载压缩包

#### 步骤 1：获取压缩包
根据您的环境：
- **如果是本地虚拟机**：直接复制 `/home/user/GEFM.zip` 到共享文件夹
- **如果是远程服务器**：使用 WinSCP、FileZilla 等工具下载

#### 步骤 2：解压到桌面

在 Windows PowerShell 中：
```powershell
cd C:\Users\yangz\Desktop
Expand-Archive -Path GEFM.zip -DestinationPath .
```

或者右键点击 `GEFM.zip` → "解压到此处"

---

### 方法 4️⃣: 直接在服务器上操作（推荐）

如果您可以直接访问服务器，在 **Linux 终端**运行：

```bash
# 方案 A：复制到共享文件夹（如果有）
cp -r /home/user/GEFM /mnt/shared/

# 方案 B：启动简单 HTTP 服务器
cd /home/user
python3 -m http.server 8000

# 然后在 Windows 浏览器访问：
# http://server-ip:8000/GEFM.zip
# 右键保存到桌面
```

---

### 方法 5️⃣: 使用 WSL（如果您使用 Windows Subsystem for Linux）

在 **Windows PowerShell** 中：

```powershell
# 从 WSL 复制到 Windows
wsl cp -r /home/user/GEFM /mnt/c/Users/yangz/Desktop/
```

或在 **WSL 终端**中：

```bash
cp -r /home/user/GEFM /mnt/c/Users/yangz/Desktop/
```

---

## 📂 解压后的目录结构

```
C:\Users\yangz\Desktop\GEFM\
├── README.md
├── QUICKSTART.md
├── setup.py
├── requirements.txt
│
├── vlfm\
│   ├── mapping\
│   │   └── scene_graph_map.py
│   ├── policy\
│   │   └── gefm_policy.py
│   └── vlm\
│       └── llm_reasoner.py
│
├── config\
│   └── experiments\
│       └── gefm_objectnav_hm3d.yaml
│
├── test\
│   └── test_gefm_components.py
│
└── docs\
    ├── GEFM_IMPLEMENTATION_PLAN.md
    ├── GEFM_INTEGRATION_GUIDE.md
    └── GEFM_RESEARCH_SUMMARY.md
```

---

## ✅ 验证传输成功

在 Windows PowerShell 中：

```powershell
# 进入 GEFM 目录
cd C:\Users\yangz\Desktop\GEFM

# 查看文件列表
ls

# 查看 README
cat README.md

# 初始化 Git（如果需要）
git init
git add .
git commit -m "Initial commit"
```

---

## 🔧 在 Windows 上使用 GEFM

### 1. 安装 Python 环境

```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 如果遇到执行策略错误，运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 运行测试

```powershell
python test\test_gefm_components.py
```

---

## 🆘 常见问题

### Q: "无法解压 .tar.gz 文件"
**A**: 使用 GEFM.zip 文件，或安装 7-Zip (https://www.7-zip.org/)

### Q: "SCP 命令不存在"
**A**:
- 安装 OpenSSH: `Settings → Apps → Optional Features → OpenSSH Client`
- 或使用 WinSCP 图形界面工具

### Q: "Git 命令不存在"
**A**: 下载安装 Git for Windows: https://git-scm.com/download/win

---

## 📧 需要帮助？

如果传输遇到问题，请告诉我：
1. 您的环境类型（本地虚拟机 / 远程服务器 / WSL）
2. 是否可以访问服务器的 IP 地址
3. 遇到的具体错误信息

我会提供针对性的解决方案！
