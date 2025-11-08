# GEFM (Graph-Enhanced Frontier Maps) Implementation Plan

## Project Overview
融合 VLFM 和 UniGoal，构建基于场景图增强的前沿地图导航系统

## Phase 1: Environment Setup (Week 1)

### 1.1 Clone UniGoal Repository
```bash
cd /home/user
git clone https://github.com/bagh2178/UniGoal.git
```

### 1.2 Analyze UniGoal Dependencies
- 提取 scene graph 相关模块
- 识别需要的 LLM/VLM 接口
- 确定与 VLFM 的兼容性

### 1.3 Create GEFM Branch
```bash
cd frontier_map
git checkout -b gefm-development
```

## Phase 2: Core Module Development (Week 2-4)

### 2.1 Scene Graph Builder (Week 2)
**File**: `vlfm/mapping/scene_graph_map.py`

**Components**:
- [ ] Node representation (Object, Room, Frontier nodes)
- [ ] Edge representation (Spatial + Semantic relations)
- [ ] Incremental update mechanism
- [ ] Confidence tracking
- [ ] Visualization utilities

**Key Functions**:
```python
class SceneGraphMap:
    def __init__(self)
    def add_object_node(self, detection, position, confidence)
    def add_spatial_edge(self, node1, node2, relation_type)
    def update_from_observation(self, rgb, depth, detections, pose)
    def get_semantic_context(self, position, radius=2.0)
    def to_text(self) -> str  # For LLM input
    def visualize(self, save_path)
```

### 2.2 Graph-Enhanced Frontier Scorer (Week 3)
**File**: `vlfm/policy/gefm_policy.py`

**Components**:
- [ ] Inherit from ITMPolicyV2
- [ ] Integrate SceneGraphMap
- [ ] Implement three-way scoring (α, β, γ)
- [ ] Add LLM reasoner interface

**Key Functions**:
```python
class GEFMPolicy(ITMPolicyV2):
    def _score_frontiers(self, frontiers, rgb, depth, goal_text)
    def _get_blip2_itm_score(self, frontier, rgb, goal_text)
    def _get_graph_matching_score(self, frontier, goal_text)
    def _get_llm_reasoning_score(self, frontier, goal_text)
    def _parse_goal_object(self, goal_text) -> str
    def _get_related_objects(self, object_name) -> List[str]
```

### 2.3 LLM Reasoner (Week 3)
**File**: `vlfm/vlm/llm_reasoner.py`

**Components**:
- [ ] Support multiple backends (OpenAI, Ollama, Claude)
- [ ] Prompt engineering for spatial reasoning
- [ ] Response parsing and error handling
- [ ] Caching mechanism

**Key Functions**:
```python
class LLMReasoner:
    def __init__(self, model_name, api_key=None)
    def query(self, prompt, max_tokens=200) -> str
    def parse_frontier_scores(self, response) -> Dict[int, float]
    def explain_reasoning(self, response) -> str
```

### 2.4 Dual-Layer Controller (Week 4)
**File**: `vlfm/policy/utils/dual_layer_controller.py`

**Components**:
- [ ] Fast layer logic (every step)
- [ ] Reasoning layer logic (every N steps)
- [ ] Adaptive triggering (low confidence detection)
- [ ] Performance monitoring

**Key Functions**:
```python
class DualLayerController:
    def should_invoke_reasoning(self, step, confidence) -> bool
    def get_execution_mode(self) -> str  # "fast" or "reasoning"
    def update_statistics(self, layer, execution_time)
```

## Phase 3: Integration and Testing (Week 5-6)

### 3.1 Configuration
**File**: `config/experiments/gefm_objectnav_hm3d.yaml`

```yaml
# GEFM 特定配置
gefm:
  # Scoring weights
  alpha: 0.5  # BLIP2-ITM weight
  beta: 0.3   # Graph matching weight
  gamma: 0.2  # LLM reasoning weight

  # Dual-layer settings
  reasoning_interval: 5  # LLM 推理间隔
  confidence_threshold: 0.8  # 低置信度触发推理

  # Scene graph settings
  sg_max_nodes: 100
  sg_spatial_radius: 2.0  # meters
  sg_confidence_decay: 0.95

  # LLM settings
  llm_model: "gpt-4"  # or "ollama/llama3"
  llm_temperature: 0.3
  llm_max_tokens: 200
```

### 3.2 Unit Tests
**File**: `test/test_gefm.py`

- [ ] Test scene graph construction
- [ ] Test frontier scoring mechanism
- [ ] Test LLM integration
- [ ] Test dual-layer controller
- [ ] Mock tests (without actual LLM calls)

### 3.3 Integration with VLFM Pipeline
- [ ] Modify `vlfm/run.py` to support GEFM policy
- [ ] Update visualization to show scene graph
- [ ] Add logging for ablation studies

## Phase 4: Experiments (Week 7-10)

### 4.1 Baseline Comparison
**Datasets**: HM3D validation set (same as VLFM paper)

**Metrics**:
- Success Rate (SR)
- Success weighted by Path Length (SPL)
- Success weighted by Navigation Error (SNE)
- Average episode length
- Computational overhead

**Baselines**:
1. VLFM (ITMPolicyV2) - 原始方法
2. GEFM (α=1, β=0, γ=0) - 退化到 VLFM
3. GEFM (full) - 完整方法

### 4.2 Ablation Studies
**Experiments**:
1. **Effect of Scene Graph**
   - GEFM w/o graph matching (β=0)
   - GEFM w/o scene graph visualization

2. **Effect of LLM Reasoning**
   - GEFM w/o LLM (γ=0)
   - Different reasoning intervals (N=3, 5, 10, ∞)

3. **Weight Sensitivity**
   - Sweep α, β, γ (constraint: α+β+γ=1)
   - Find optimal configuration

4. **LLM Model Comparison**
   - GPT-4 vs GPT-3.5
   - Open-source (Llama 3, Mixtral)

### 4.3 Qualitative Analysis
- [ ] Visualize scene graph evolution
- [ ] Show frontier score comparison (VLFM vs GEFM)
- [ ] Collect LLM reasoning examples
- [ ] Record failure cases

## Phase 5: Paper Writing (Week 11-14)

### 5.1 Paper Structure

**Title**: *GEFM: Graph-Enhanced Frontier Maps for Zero-Shot Semantic Navigation*

**Abstract**:
- Problem: VLFM 缺乏结构化语义理解
- Solution: Scene graph + dual-layer exploration
- Results: X% improvement on HM3D

**Sections**:
1. Introduction
   - Motivation: Why scene graph matters
   - Contributions (3-4 bullet points)

2. Related Work
   - Frontier-based navigation (VLFM, classical methods)
   - Scene graph for robotics (UniGoal, Scene Graph Navigation)
   - Vision-Language models for navigation

3. Method
   - 3.1 Overview (架构图)
   - 3.2 Scene Graph Construction
   - 3.3 Graph-Enhanced Frontier Scoring
   - 3.4 Dual-Layer Exploration Strategy

4. Experiments
   - 4.1 Experimental Setup
   - 4.2 Baseline Comparison
   - 4.3 Ablation Studies
   - 4.4 Qualitative Analysis

5. Conclusion and Future Work

**Figures** (至少 6 个):
- Fig 1: System overview
- Fig 2: Scene graph example
- Fig 3: Frontier scoring comparison
- Fig 4: Quantitative results (bar charts)
- Fig 5: Ablation study results
- Fig 6: Qualitative examples (success cases)

### 5.2 Target Venues
**First Choice**: ICRA 2026 (Sep 2025 deadline)
**Backup**: IROS 2026, CoRL 2025, RA-L

## Phase 6: Code Release (Week 15-16)

### 6.1 Documentation
- [ ] README with installation instructions
- [ ] API documentation
- [ ] Tutorial notebook
- [ ] Pre-trained models (if applicable)

### 6.2 GitHub Repository
- [ ] Clean up code
- [ ] Add license
- [ ] Create issues template
- [ ] Add citation information

---

## Key Milestones

- **Week 2**: Scene graph module完成
- **Week 4**: GEFM policy 完成
- **Week 6**: 首次在 HM3D 上跑通
- **Week 10**: 所有实验完成
- **Week 14**: 论文初稿完成

## Dependencies to Add

```bash
# LLM 相关
pip install openai anthropic ollama-python

# 场景图可视化
pip install networkx pygraphviz

# 实验管理
pip install wandb tensorboard
```

## Potential Challenges

1. **LLM API 成本**: 使用 Ollama 本地部署作为备选
2. **Scene Graph 准确性**: 依赖 GroundingDINO 检测质量
3. **实时性**: 需要仔细平衡推理频率
4. **泛化性**: 在不同场景下表现可能不稳定

## Risk Mitigation

- 早期建立 baseline (Week 6)
- 频繁可视化和调试
- 准备 Plan B（如果 LLM 太慢，简化为规则推理）
- 保持与 VLFM 的兼容性（可以随时退回）
