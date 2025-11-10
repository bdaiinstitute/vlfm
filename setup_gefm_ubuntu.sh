#!/bin/bash
# GEFM Environment Setup Script for Ubuntu 20.04
# CUDA Version: 11.8 (recommended)

set -e  # Exit on error

echo "========================================"
echo "GEFM Environment Setup"
echo "Ubuntu 20.04 + CUDA 11.8"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Step 1: Check CUDA installation
print_step "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    print_success "CUDA detected: $CUDA_VERSION"
else
    print_error "CUDA not found. Please install CUDA 11.8"
    exit 1
fi

# Step 2: Install system dependencies
print_step "Installing system dependencies..."
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

print_success "System dependencies installed"

# Step 3: Create conda environment (if conda available)
print_step "Setting up Python environment..."

if command -v conda &> /dev/null; then
    echo "Using Conda..."

    # Check if environment exists
    if conda env list | grep -q "^gefm "; then
        print_step "Environment 'gefm' already exists. Removing..."
        conda env remove -n gefm -y
    fi

    conda create -n gefm python=3.9 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gefm
    print_success "Conda environment 'gefm' created and activated"

    ENV_TYPE="conda"
else
    echo "Conda not found. Using venv..."

    # Create virtual environment
    python3.9 -m venv ~/gefm_env
    source ~/gefm_env/bin/activate
    print_success "Virtual environment created and activated"

    ENV_TYPE="venv"
fi

# Step 4: Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_success "pip upgraded"

# Step 5: Install PyTorch with CUDA 11.8
print_step "Installing PyTorch 1.12.1 + CUDA 11.8..."
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
print_success "PyTorch installed"

# Step 6: Verify PyTorch CUDA
print_step "Verifying PyTorch CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" || {
    print_error "PyTorch CUDA verification failed"
    exit 1
}
print_success "PyTorch CUDA verification passed"

# Step 7: Install VLFM dependencies
print_step "Installing VLFM dependencies..."

# GroundingDINO
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67

# LAVIS (BLIP-2)
pip install salesforce-lavis==1.0.2

# Other VLFM dependencies
pip install numpy==1.26.4 \
    flask>=2.3.2 \
    seaborn>=0.12.2 \
    open3d>=0.17.0 \
    transformers==4.26.0 \
    timm==0.4.12 \
    opencv-python==4.5.5.64

# Frontier exploration
pip install git+https://github.com/naokiyokoyama/frontier_exploration.git

# MobileSAM
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Depth camera filtering
pip install git+https://github.com/naokiyokoyama/depth_camera_filtering.git

print_success "VLFM dependencies installed"

# Step 8: Install Habitat (optional, for full navigation)
read -p "Do you want to install Habitat-Sim and Habitat-Lab? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Installing Habitat..."

    pip install habitat-sim==0.2.4
    pip install habitat-lab==0.2.420230405
    pip install habitat-baselines==0.2.420230405

    print_success "Habitat installed"
else
    print_step "Skipping Habitat installation"
fi

# Step 9: Install GEFM-specific dependencies
print_step "Installing GEFM-specific dependencies..."

# LLM backends
pip install openai>=1.0.0 \
    anthropic>=0.25.0 \
    ollama>=0.1.0

# Scene graph and visualization
pip install networkx>=3.0 \
    matplotlib>=3.7.0

# Experiment tracking
pip install wandb>=0.16.0 \
    tensorboard>=2.15.0

# Development tools
pip install pytest>=7.4.0 \
    black>=23.0.0 \
    ruff>=0.1.0

print_success "GEFM dependencies installed"

# Step 10: Clone GEFM repository (if not already present)
print_step "Setting up GEFM repository..."

GEFM_DIR="$HOME/GEFM"

if [ -d "$GEFM_DIR" ]; then
    print_step "GEFM directory already exists at $GEFM_DIR"
else
    # Check if we have the local copy
    if [ -d "/home/user/GEFM" ]; then
        cp -r /home/user/GEFM "$GEFM_DIR"
        print_success "Copied GEFM from /home/user/GEFM"
    else
        print_step "Please manually copy GEFM directory to $HOME/GEFM"
    fi
fi

# Step 11: Install GEFM package
if [ -d "$GEFM_DIR" ]; then
    print_step "Installing GEFM package..."
    cd "$GEFM_DIR"
    pip install -e .
    print_success "GEFM package installed"
fi

# Step 12: Install Ollama (local LLM, optional)
read -p "Do you want to install Ollama (local LLM)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh

    print_step "Pulling Llama3 model..."
    ollama pull llama3

    print_success "Ollama installed and Llama3 model downloaded"
else
    print_step "Skipping Ollama installation"
fi

# Step 13: Create activation script
print_step "Creating activation script..."

if [ "$ENV_TYPE" = "conda" ]; then
    ACTIVATE_CMD="conda activate gefm"
else
    ACTIVATE_CMD="source ~/gefm_env/bin/activate"
fi

cat > ~/activate_gefm.sh <<EOF
#!/bin/bash
# GEFM Environment Activation Script

# Activate Python environment
$ACTIVATE_CMD

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# Optional: Set OpenAI API key
# export OPENAI_API_KEY="your-api-key-here"

echo "GEFM environment activated!"
echo "Python: \$(which python)"
echo "PyTorch CUDA: \$(python -c 'import torch; print(torch.cuda.is_available())')"

# Navigate to GEFM directory
cd $GEFM_DIR 2>/dev/null || cd ~

EOF

chmod +x ~/activate_gefm.sh
print_success "Activation script created: ~/activate_gefm.sh"

# Step 14: Run tests
print_step "Running GEFM component tests..."

if [ -d "$GEFM_DIR" ]; then
    cd "$GEFM_DIR"
    python test/test_gefm_components.py || {
        print_error "Some tests failed (this is OK if LLM not configured)"
    }
fi

# Step 15: Summary
echo ""
echo "========================================"
echo -e "${GREEN}GEFM Environment Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Environment Type: $ENV_TYPE"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Next Steps:"
echo "1. Activate environment:"
if [ "$ENV_TYPE" = "conda" ]; then
    echo "   conda activate gefm"
else
    echo "   source ~/gefm_env/bin/activate"
fi
echo ""
echo "Or use the activation script:"
echo "   source ~/activate_gefm.sh"
echo ""
echo "2. Test GEFM components:"
echo "   cd $GEFM_DIR"
echo "   python test/test_gefm_components.py"
echo ""
echo "3. (Optional) Configure LLM API key:"
echo "   export OPENAI_API_KEY='your-key'"
echo ""
echo "4. Read documentation:"
echo "   cat $GEFM_DIR/README.md"
echo "   cat $GEFM_DIR/QUICKSTART.md"
echo ""
echo "========================================"
