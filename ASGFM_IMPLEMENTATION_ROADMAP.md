# ASGFM 实施路线图

## 🎯 项目概览

**目标**: 实现 ASGFM (Adaptive Scene Graph Frontier Maps)
**时间**: 20-24 周（5-6 个月）
**难度**: ⭐⭐⭐⭐⭐
**创新度**: ⭐⭐⭐⭐⭐
**发表目标**: ICRA 2026 / CoRL 2025 / RSS 2026

---

## 📅 时间线（24 周）

```
Phase 1: 基础架构 (Week 1-6)
  ├─ Week 1-2: 概率场景图
  ├─ Week 3-4: 语义 Frontier 生成
  └─ Week 5-6: 基础集成测试

Phase 2: 核心算法 (Week 7-12)
  ├─ Week 7-8: 多阶段控制器
  ├─ Week 9-10: 不确定性建模
  └─ Week 11-12: 初步实验

Phase 3: 高级功能 (Week 13-16)
  ├─ Week 13-14: 自我反思机制
  ├─ Week 15-16: 策略适应
  └─ Week 16: 完整系统测试

Phase 4: 实验与论文 (Week 17-24)
  ├─ Week 17-18: 完整实验（HM3D）
  ├─ Week 19-20: 消融实验
  ├─ Week 21-22: 论文撰写
  └─ Week 23-24: 内部审阅 + 修改
```

---

## 📦 Phase 1: 基础架构 (Week 1-6)

### Week 1-2: 概率场景图

#### 目标
- 实现带不确定性的场景图数据结构
- 支持概率更新和贝叶斯推理

#### 任务清单

**Task 1.1: 数据结构设计** (2 天)
```python
# 文件: vlfm/mapping/probabilistic_scene_graph.py

class ProbabilisticNode:
    """概率节点实现"""
    - 基础属性（id, type, label, position）
    - 不确定性建模（existence_prob, position_variance）
    - 时序信息（first_seen, last_updated, observation_count）
    - 预测来源（observation/inference/llm）

class ProbabilisticSceneGraph:
    """概率场景图实现"""
    - 节点和边管理
    - 不确定性传播
    - 查询接口
```

**Task 1.2: 贝叶斯更新** (3 天)
```python
def bayesian_update(node, observation, likelihood_model):
    """
    P(exists | obs) ∝ P(obs | exists) × P(exists)

    实现：
    - 正向观测（看到对象）
    - 负向观测（应看到但未看到）
    - 超时衰减（长时间未观测）
    """
```

**Task 1.3: 图推理算法** (3 天)
```python
def propagate_uncertainty(graph, inference_rules):
    """
    不确定性在图中传播

    规则：
    - 如果 A near B，且 A 存在，则 B 存在概率提升
    - 如果房间类型确定，推理可能的对象
    """
```

**Task 1.4: 单元测试** (2 天)
- 测试节点创建和更新
- 测试概率传播
- 测试图查询

#### 交付物
- ✅ `probabilistic_scene_graph.py` (500+ 行)
- ✅ `test_probabilistic_graph.py`
- ✅ 文档：概率模型说明

---

### Week 3-4: 语义 Frontier 生成

#### 目标
- 实现基于场景图的语义 frontier 生成
- 集成 LLM 进行房间和对象推理

#### 任务清单

**Task 2.1: 语义 Frontier 数据结构** (1 天)
```python
# 文件: vlfm/mapping/semantic_frontier.py

class SemanticFrontier:
    """语义 frontier"""
    - 位置（position）
    - 类型（geometric / semantic / virtual）
    - 语义分数（semantic_score）
    - 生成原因（reason）
    - 置信度（confidence）
```

**Task 2.2: 房间推理** (4 天)
```python
def infer_unvisited_rooms(scene_graph, goal_object, llm):
    """
    基于已观测对象推理未访问房间

    流程：
    1. 提取已观测对象
    2. LLM 推理当前房间类型
    3. LLM 推理相邻房间
    4. 估计房间位置
    5. 生成虚拟房间节点
    """
```

**Task 2.3: 对象共现** (3 天)
```python
def get_cooccurring_objects(scene_graph, goal_object):
    """
    获取与目标对象常共现的对象

    数据源：
    1. 预定义共现表（如 COCO-Stuff）
    2. LLM 查询
    3. 场景图历史统计
    """
```

**Task 2.4: 虚拟 Frontier 生成** (2 天)
```python
def create_virtual_frontiers(inferred_rooms, geometric_frontiers):
    """
    在推测的房间位置生成虚拟 frontier

    策略：
    - 在推测房间中心生成
    - 在推测房间入口生成
    - 在几何 frontier 附近增强
    """
```

**Task 2.5: 集成测试** (2 天)
- Mock LLM 测试
- 真实场景测试

#### 交付物
- ✅ `semantic_frontier.py` (400+ 行)
- ✅ `semantic_frontier_generator.py` (500+ 行)
- ✅ 测试和示例

---

### Week 5-6: 基础集成测试

#### 目标
- 将概率场景图和语义 frontier 集成到 VLFM

#### 任务清单

**Task 3.1: VLFM 集成** (5 天)
```python
# 文件: vlfm/policy/asgfm_policy.py

class ASGFMPolicy(ITMPolicyV2):
    """ASGFM 主策略"""

    def __init__(self, config):
        super().__init__(config)
        self.prob_scene_graph = ProbabilisticSceneGraph()
        self.semantic_frontier_gen = SemanticFrontierGenerator()

    def act(self, observations, ...):
        # 更新概率场景图
        # 生成语义 frontier
        # 选择 frontier
        # 导航
```

**Task 3.2: 端到端测试** (4 天)
- 在简化环境测试
- 检查场景图更新
- 验证语义 frontier 生成

**Task 3.3: 调试和优化** (3 天)

#### 交付物
- ✅ 基础 ASGFM 可运行
- ✅ 在 1-5 个 episode 上测试通过

---

## 📦 Phase 2: 核心算法 (Week 7-12)

### Week 7-8: 多阶段探索控制器

#### 目标
- 实现三阶段状态机（Zero/Partial/Perfect Match）
- 实现场景图匹配算法

#### 任务清单

**Task 4.1: 状态机实现** (3 天)
```python
# 文件: vlfm/policy/utils/multi_stage_controller.py

class MultiStageController:
    - 状态定义（ExplorationStage enum）
    - 状态转换逻辑
    - 策略参数配置（每个阶段不同）
```

**Task 4.2: 图匹配算法** (4 天)
```python
def compute_graph_matching(scene_graph, goal_graph):
    """
    匹配类型：
    - Exact: 找到目标对象
    - Partial: 找到相关对象
    - Zero: 无相关对象

    算法：
    - 子图同构检测
    - 语义相似度计算
    - 结构相似度计算
    """
```

**Task 4.3: 阶段特定策略** (3 天)
- Zero-match: 广泛探索参数
- Partial-match: 聚焦搜索参数
- Perfect-match: 直接导航

**Task 4.4: 测试** (2 天)

#### 交付物
- ✅ `multi_stage_controller.py`
- ✅ 阶段转换工作正常
- ✅ 可视化状态转换

---

### Week 9-10: 不确定性建模完善

#### 目标
- 完善概率更新机制
- 实现信念传播算法

#### 任务清单

**Task 5.1: 似然模型** (3 天)
```python
def likelihood_model(observation, object_type, distance, viewpoint):
    """
    P(obs | object exists, distance, viewpoint)

    考虑：
    - 距离（远处不容易看到）
    - 视角（侧面/背面不容易识别）
    - 遮挡（被其他物体遮挡）
    - 检测器性能（GroundingDINO 准确率）
    """
```

**Task 5.2: 信念传播** (4 天)
```python
def belief_propagation(graph, evidence):
    """
    在场景图上进行信念传播

    算法：
    - Loopy Belief Propagation
    - 或简化的迭代更新

    传播规则：
    - 空间相关性
    - 语义相关性
    - 时序一致性
    """
```

**Task 5.3: 集成和测试** (3 天)

#### 交付物
- ✅ 概率推理引擎完善
- ✅ 测试不同场景的概率更新

---

### Week 11-12: 初步实验

#### 目标
- 在 HM3D 上运行 20-50 个 episodes
- 收集初步数据

#### 任务清单

**Task 6.1: 配置文件** (1 天)
```yaml
# config/experiments/asgfm_objectnav_hm3d.yaml

asgfm:
  # 概率场景图
  prob_graph:
    initial_existence_prob: 0.5
    observation_boost: 0.3
    decay_rate: 0.98

  # 语义 frontier
  semantic_frontier:
    enable: true
    llm_inference: true
    virtual_frontier_threshold: 0.6

  # 多阶段控制
  multi_stage:
    enable: true
    partial_match_threshold: 0.3
    perfect_match_threshold: 0.8
```

**Task 6.2: 运行实验** (5 天)
- 20 episodes 用于调试
- 50 episodes 收集初步数据

**Task 6.3: 数据分析** (3 天)
- 成功率
- 平均步数
- 场景图质量
- 语义 frontier 准确率

**Task 6.4: 问题识别和修复** (3 天)

#### 交付物
- ✅ 初步实验结果
- ✅ 问题列表
- ✅ 改进计划

---

## 📦 Phase 3: 高级功能 (Week 13-16)

### Week 13-14: 自我反思机制

#### 目标
- 实现探索记忆
- 实现失败分析和信念修正

#### 任务清单

**Task 7.1: 探索记忆** (3 天)
```python
# 文件: vlfm/policy/utils/exploration_memory.py

class ExplorationMemory:
    - 记录访问历史
    - 记录失败案例
    - 提取失败模式
```

**Task 7.2: 失败分析** (4 天)
```python
def analyze_failures(memory, llm):
    """
    LLM 分析失败模式

    问题：
    1. 哪些假设错了？
    2. 应该搜索哪里？
    3. 调整什么策略？
    """
```

**Task 7.3: 信念修正** (3 天)
```python
def update_beliefs_from_reflection(analysis, scene_graph):
    """
    根据反思更新场景图中的概率

    操作：
    - 降低失败区域的对象概率
    - 提高推荐区域的对象概率
    - 更新空间先验
    """
```

**Task 7.4: 测试** (2 天)

#### 交付物
- ✅ `exploration_memory.py`
- ✅ `self_reflection.py`
- ✅ 失败案例分析示例

---

### Week 15-16: 策略适应

#### 目标
- 实现动态策略调整
- 实现性能监控

#### 任务清单

**Task 8.1: 性能指标** (2 天)
```python
def compute_performance_metrics(memory):
    """
    计算性能指标

    指标：
    - Frontier 选择成功率
    - LLM 预测准确率
    - 探索效率（steps/progress）
    - 时间效率
    """
```

**Task 8.2: 策略适应** (4 天)
```python
class AdaptiveStrategy:
    """动态调整策略参数"""

    def adapt(self, performance):
        # 调整权重（α, β, γ）
        # 调整探索 bonus
        # 调整 LLM 频率
        # 调整其他超参数
```

**Task 8.3: 在线学习** (3 天)
- 实现简单的 online learning
- 记录哪些策略有效

**Task 8.4: 测试** (3 天)

#### 交付物
- ✅ `adaptive_strategy.py`
- ✅ 策略适应工作正常
- ✅ 性能监控可视化

---

## 📦 Phase 4: 实验与论文 (Week 17-24)

### Week 17-18: 完整实验（HM3D）

#### 目标
- 运行 100+ episodes
- 收集完整数据

#### 任务清单

**Task 9.1: 实验配置** (1 天)
- 确定实验参数
- 准备计算资源

**Task 9.2: 运行实验** (8 天)
```bash
# Baseline: VLFM
python -m vlfm.run --config vlfm_hm3d num_episodes=100

# ASGFM (full)
python -m vlfm.run --config asgfm_hm3d num_episodes=100

# 并行运行（如果有多台机器）
```

**Task 9.3: 数据收集** (3 天)
- Success Rate
- SPL
- Steps to goal
- 语义 frontier 准确率
- 场景图质量
- 反思次数和效果

#### 交付物
- ✅ 完整实验数据
- ✅ 初步结果分析

---

### Week 19-20: 消融实验

#### 目标
- 证明每个组件的贡献

#### 任务清单

**Task 10.1: 消融配置** (1 天)
```yaml
# 1. ASGFM w/o semantic frontiers
# 2. ASGFM w/o uncertainty modeling
# 3. ASGFM w/o multi-stage
# 4. ASGFM w/o self-reflection
# 5. ASGFM w/o LLM
```

**Task 10.2: 运行消融实验** (8 天)
- 每个配置 50-100 episodes

**Task 10.3: 数据分析** (3 天)
- 对比表格
- 统计显著性测试
- 可视化

#### 交付物
- ✅ 消融实验结果
- ✅ 组件贡献分析

---

### Week 21-22: 论文撰写

#### 目标
- 完成论文初稿

#### 结构

**Abstract** (1 天)
**Introduction** (2 天)
- 问题动机
- 现有方法局限
- 我们的贡献

**Related Work** (2 天)
- Frontier-based navigation
- Scene graph navigation
- Zero-shot learning
- LLM for robotics

**Method** (4 天)
- 3.1 Overview
- 3.2 Probabilistic Scene Graph
- 3.3 Semantic Frontier Generation
- 3.4 Multi-Stage Exploration
- 3.5 Self-Reflective Learning

**Experiments** (3 天)
- 4.1 Setup
- 4.2 Main Results
- 4.3 Ablation Studies
- 4.4 Qualitative Analysis

**Conclusion** (1 天)

#### 交付物
- ✅ 论文初稿（8 页）
- ✅ 所有图表

---

### Week 23-24: 审阅和修改

#### 任务
- 内部审阅（导师/同事）
- 根据反馈修改
- 润色语言
- 准备补充材料（视频、代码）

#### 交付物
- ✅ 论文终稿
- ✅ 补充材料
- ✅ 代码开源准备

---

## 🎯 关键里程碑

| Week | 里程碑 | 检查点 |
|------|--------|--------|
| 2 | 概率场景图完成 | 单元测试通过 |
| 4 | 语义 frontier 生成完成 | Mock 测试通过 |
| 6 | 基础系统集成 | 1 episode 运行成功 |
| 8 | 多阶段控制器完成 | 阶段转换正常 |
| 10 | 不确定性建模完善 | 概率更新正确 |
| 12 | 初步实验完成 | 50 episodes 数据 |
| 14 | 自我反思完成 | 失败分析有效 |
| 16 | 完整系统测试 | 所有功能正常 |
| 18 | 完整实验完成 | 100 episodes 数据 |
| 20 | 消融实验完成 | 组件贡献明确 |
| 22 | 论文初稿完成 | 8 页完整论文 |
| 24 | 论文终稿 | 准备投稿 |

---

## 💰 资源需求

### 计算资源
- **GPU**: NVIDIA RTX 3090 / A6000 (24GB+)
- **数量**: 2-4 台（并行实验）
- **时间**: ~300 GPU 小时

### LLM API 成本
- **Ollama**: 免费（推荐开发）
- **OpenAI GPT-4**: ~$100-200（实验）
- **建议**: 开发用 Ollama，最终实验用 GPT-4

### 数据集
- HM3D: 免费（需申请）
- MP3D: 需 Matterport 账号

---

## ⚠️ 风险和缓解

### 风险 1: LLM 推理太慢
**缓解**:
- 减少 LLM 调用频率
- 使用更快的本地模型
- 实现激进的缓存

### 风险 2: 语义 frontier 准确率低
**缓解**:
- 降级到几何 frontier
- 调整语义分数权重
- 改进 LLM 提示词

### 风险 3: 实验结果不显著
**缓解**:
- 增加实验 episodes 数量
- 选择更困难的测试场景
- 强调定性分析和可解释性

### 风险 4: 时间不足
**缓解**:
- 优先实现核心功能
- 简化某些组件（如自我反思）
- 准备 Plan B（退回到 GEFM）

---

## 📈 成功标准

### 最低标准（可发表）:
- ✅ SR 提升 5%+
- ✅ SPL 提升 5%+
- ✅ 至少 1 个创新点明显有效
- ✅ 可解释性强于 baseline

### 理想标准（顶会）:
- ✅ SR 提升 10%+
- ✅ SPL 提升 10%+
- ✅ 所有 4 个创新点都有贡献
- ✅ 在长任务中表现出"智能涌现"

---

## 🤝 协作建议

如果这是团队项目：
- **成员 1**: 概率场景图 + 不确定性建模
- **成员 2**: 语义 frontier + LLM 集成
- **成员 3**: 多阶段控制 + 自我反思
- **导师**: 论文指导 + 实验设计

如果是个人项目：
- 严格按时间线
- 优先核心功能
- 寻求实验室帮助（GPU 资源、论文审阅）

---

**准备好开始了吗？从 Week 1 开始！** 🚀
