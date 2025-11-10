# ASGFM: Adaptive Scene Graph Frontier Maps
## 完整设计方案

---

## 🎯 核心创新架构

```
┌─────────────────────────────────────────────────────────────────┐
│              ASGFM: Adaptive Scene Graph Frontier Maps          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                        ┌──────────────┐      │
│  │  RGB-D Input │                        │  Exploration │      │
│  │   + History  │                        │   Memory     │      │
│  └──────┬───────┘                        └──────┬───────┘      │
│         │                                       │              │
│         ▼                                       ▼              │
│  ┌────────────────────────────────────────────────────┐       │
│  │     Incremental Scene Graph Constructor           │       │
│  │  ┌──────────────────────────────────────────┐     │       │
│  │  │ - Spatial nodes (objects, rooms)         │     │       │
│  │  │ - Uncertainty nodes (predicted objects)  │     │       │
│  │  │ - Temporal edges (observation sequence)  │     │       │
│  │  │ - Confidence propagation                 │     │       │
│  │  └──────────────────────────────────────────┘     │       │
│  └────────────────┬───────────────────────────────────┘       │
│                   │                                            │
│         ┌─────────┴──────────┐                                │
│         │                    │                                │
│         ▼                    ▼                                │
│  ┌──────────────┐    ┌──────────────────┐                    │
│  │  Geometric   │    │  Semantic        │                    │
│  │  Frontiers   │    │  Frontiers       │                    │
│  │  (VLFM)      │    │  (Graph-guided)  │                    │
│  └──────┬───────┘    └──────┬───────────┘                    │
│         │                   │                                 │
│         └──────────┬────────┘                                 │
│                    │                                           │
│                    ▼                                           │
│  ┌──────────────────────────────────────────┐                │
│  │  Multi-Stage Exploration Controller      │                │
│  │  ┌────────────────────────────────────┐  │                │
│  │  │ Stage 1: Zero-Match                │  │                │
│  │  │   → Broad exploration              │  │                │
│  │  │   → Value map + Graph priors       │  │                │
│  │  ├────────────────────────────────────┤  │                │
│  │  │ Stage 2: Partial-Match             │  │                │
│  │  │   → Focused search                 │  │                │
│  │  │   → Graph reasoning + LLM          │  │                │
│  │  ├────────────────────────────────────┤  │                │
│  │  │ Stage 3: Perfect-Match             │  │                │
│  │  │   → Object-centric navigation      │  │                │
│  │  │   → Direct approach                │  │                │
│  │  └────────────────────────────────────┘  │                │
│  └──────────────────┬───────────────────────┘                │
│                     │                                          │
│                     ▼                                          │
│  ┌──────────────────────────────────────────┐                │
│  │  Self-Reflective Learning                │                │
│  │  - Failure case analysis                 │                │
│  │  - Strategy adaptation                   │                │
│  │  - Belief updating                       │                │
│  └──────────────────┬───────────────────────┘                │
│                     │                                          │
│                     ▼                                          │
│              Navigation Action                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 四大核心创新详解

### **创新 1: 增量式场景图 + 不确定性建模**

#### 1.1 扩展的节点类型

```python
class NodeType(Enum):
    # 基础节点（来自观测）
    OBJECT = "object"              # 确认观测到的对象
    ROOM = "room"                  # 推理出的房间类型

    # 创新：不确定性节点
    PREDICTED_OBJECT = "predicted_object"   # 预测但未观测的对象
    VIRTUAL_ROOM = "virtual_room"           # 推理出的未访问房间
    FRONTIER_NODE = "frontier"              # 探索边界点
```

#### 1.2 概率场景图

```python
class ProbabilisticNode:
    """带有不确定性的节点"""
    node_id: int
    node_type: NodeType
    label: str
    position: np.ndarray  # 期望位置

    # 创新：不确定性建模
    existence_prob: float      # 存在概率 [0, 1]
    position_variance: float   # 位置不确定性
    confidence: float          # 观测置信度

    # 创新：时序信息
    first_seen: int           # 首次观测步数
    last_updated: int         # 最后更新步数
    observation_count: int    # 观测次数

    # 创新：预测来源
    prediction_source: str    # "observation" / "inference" / "llm"
```

#### 1.3 贝叶斯更新机制

```python
def update_existence_probability(self, observation_result, step):
    """
    贝叶斯更新对象存在概率

    P(exists | obs) ∝ P(obs | exists) × P(exists)
    """
    if observation_result == "confirmed":
        # 观测到对象
        self.existence_prob = min(1.0, self.existence_prob + 0.3)
        self.confidence = min(1.0, self.confidence + 0.2)

    elif observation_result == "not_found":
        # 应该看到但没看到
        self.existence_prob *= 0.7
        self.confidence *= 0.9

    elif observation_result == "out_of_view":
        # 不在视野内，缓慢衰减
        self.existence_prob *= 0.98

    self.last_updated = step
```

---

### **创新 2: 语义 Frontier 生成（Graph-Guided）**

#### 2.1 传统 Frontier vs 语义 Frontier

```
传统 Frontier（VLFM）:
  - 基于深度图的几何边界
  - "已探索 vs 未探索" 的分界线
  - 纯粹空间驱动

语义 Frontier（ASGFM）:
  - 基于场景图推理的高价值区域
  - "有可能存在目标的未知区域"
  - 语义 + 空间双重驱动
```

#### 2.2 语义 Frontier 生成算法

```python
class SemanticFrontierGenerator:
    """基于场景图生成语义 frontier"""

    def generate_semantic_frontiers(
        self,
        scene_graph: ProbabilisticSceneGraph,
        goal_object: str,
        geometric_frontiers: List[Frontier]
    ) -> List[SemanticFrontier]:
        """
        生成语义 frontier

        策略：
        1. Room-based reasoning（房间推理）
        2. Object co-occurrence（对象共现）
        3. Spatial layout priors（空间布局先验）
        """
        semantic_frontiers = []

        # 策略 1: 房间推理
        # "椅子通常在餐厅/客厅，如果看到餐桌 → 推测附近有餐厅"
        inferred_rooms = self._infer_unvisited_rooms(scene_graph, goal_object)

        for room in inferred_rooms:
            # 在推测的房间位置生成虚拟 frontier
            virtual_frontier = self._create_virtual_frontier(
                room_type=room.label,
                expected_position=room.position,
                confidence=room.existence_prob,
                reason=f"Inferred {room.label} may contain {goal_object}"
            )
            semantic_frontiers.append(virtual_frontier)

        # 策略 2: 对象共现
        # "看到沙发 → 附近可能有茶几/电视"
        cooccurring_objects = self._get_cooccurring_objects(
            scene_graph,
            goal_object
        )

        for obj in cooccurring_objects:
            # 在共现对象附近生成 frontier
            nearby_frontiers = self._find_nearby_geometric_frontiers(
                obj.position,
                geometric_frontiers,
                radius=3.0
            )

            for gf in nearby_frontiers:
                gf.semantic_score += 0.5  # 提升分数
                gf.reason = f"Near {obj.label}, likely to have {goal_object}"

        # 策略 3: 空间布局先验
        # "厨房通常在房子的某个方位"
        layout_priors = self._get_spatial_priors(scene_graph, goal_object)

        # 合并几何 + 语义 frontier
        all_frontiers = geometric_frontiers + semantic_frontiers

        return all_frontiers

    def _infer_unvisited_rooms(self, scene_graph, goal_object):
        """
        推理未访问的房间

        示例：
        - 观测到：sofa, TV, coffee_table
        - 推理：这是客厅
        - 进一步推理：客厅附近可能有厨房/餐厅
        - 目标：chair → 在餐厅的概率高
        """
        observed_objects = scene_graph.get_observed_objects()

        # 用 LLM 推理当前房间类型
        prompt = f"""
        Observed objects: {[obj.label for obj in observed_objects]}

        Question 1: What room type is this likely to be?
        Question 2: Given the goal is to find a {goal_object},
                    what adjacent rooms should we explore?

        Format:
        Current room: [room type]
        Target rooms: [room1, room2, ...]
        Reasoning: [brief explanation]
        """

        llm_response = self.llm.query(prompt)

        # 解析响应，生成虚拟房间节点
        target_rooms = self._parse_room_inference(llm_response)

        # 为每个推测的房间估计位置
        virtual_rooms = []
        for room_type in target_rooms:
            position = self._estimate_room_position(
                room_type,
                observed_objects,
                scene_graph
            )

            virtual_room = ProbabilisticNode(
                node_type=NodeType.VIRTUAL_ROOM,
                label=room_type,
                position=position,
                existence_prob=0.6,  # 初始概率
                prediction_source="llm"
            )
            virtual_rooms.append(virtual_room)

        return virtual_rooms
```

#### 2.3 虚拟 Frontier 示例

```
场景：
  目标：找到 "microwave"
  当前观测：table, chair (推理为餐厅)

推理链：
  1. 观测到 table + chair → 当前在餐厅
  2. microwave 通常在厨房
  3. 厨房通常与餐厅相邻
  4. 估计厨房在餐厅的 [方向]
  5. 在该方向生成虚拟 frontier

结果：
  - Geometric frontier A: 普通未探索区域 (score=0.5)
  - Semantic frontier B: 推测的厨房位置 (score=0.9) ⭐

选择 frontier B 优先探索！
```

---

### **创新 3: 三阶段探索匹配**

#### 3.1 探索状态机

```python
class ExplorationStage(Enum):
    ZERO_MATCH = 0      # 完全未匹配，广泛探索
    PARTIAL_MATCH = 1   # 部分匹配，聚焦搜索
    PERFECT_MATCH = 2   # 完美匹配，直接导航

class MultiStageController:
    """三阶段探索控制器"""

    def __init__(self):
        self.current_stage = ExplorationStage.ZERO_MATCH
        self.goal_graph = None  # 目标的场景图表示
        self.match_score = 0.0

    def update_stage(self, scene_graph, goal_object):
        """
        更新探索阶段

        阶段转换条件：
        Zero → Partial: 检测到与目标相关的对象
        Partial → Perfect: 检测到目标对象本身
        """
        # 计算场景图匹配度
        match_result = self._compute_graph_matching(
            scene_graph,
            self.goal_graph
        )

        self.match_score = match_result.score

        # 阶段转换逻辑
        if match_result.exact_match:
            # 找到目标对象
            self.transition_to(ExplorationStage.PERFECT_MATCH)

        elif match_result.related_objects_found:
            # 找到相关对象（如：找椅子，看到桌子）
            if self.current_stage == ExplorationStage.ZERO_MATCH:
                self.transition_to(ExplorationStage.PARTIAL_MATCH)

        elif match_result.no_progress:
            # 长时间无进展，可能需要回退
            if self.steps_without_progress > 50:
                self.transition_to(ExplorationStage.ZERO_MATCH)

        return self.current_stage

    def get_exploration_strategy(self):
        """根据当前阶段返回探索策略"""

        if self.current_stage == ExplorationStage.ZERO_MATCH:
            # 阶段 1: 广泛探索
            return {
                "frontier_selection": "max_coverage",  # 最大化覆盖面积
                "llm_frequency": 10,                   # 低频 LLM
                "use_semantic_frontiers": True,        # 使用语义 frontier
                "exploration_bonus": 0.3,              # 探索奖励
                "focus": "broad"
            }

        elif self.current_stage == ExplorationStage.PARTIAL_MATCH:
            # 阶段 2: 聚焦搜索
            return {
                "frontier_selection": "semantic_guided",  # 语义引导
                "llm_frequency": 5,                       # 中频 LLM
                "use_semantic_frontiers": True,
                "exploration_bonus": 0.1,
                "focus": "narrow",
                "search_radius": 5.0  # 在匹配区域附近搜索
            }

        else:  # PERFECT_MATCH
            # 阶段 3: 直接导航
            return {
                "frontier_selection": "none",      # 不用 frontier
                "navigation_mode": "direct",       # 直接导航到目标
                "use_pointnav": True,
                "focus": "target"
            }
```

#### 3.2 场景图匹配算法

```python
def _compute_graph_matching(self, scene_graph, goal_graph):
    """
    计算场景图与目标图的匹配度

    匹配类型：
    1. Exact match: 找到目标对象
    2. Partial match: 找到相关对象
    3. Zero match: 没有相关对象
    """
    # 1. 检查精确匹配
    target_object = goal_graph.get_target_object()
    if scene_graph.contains(target_object):
        return MatchResult(
            score=1.0,
            exact_match=True,
            matched_node=scene_graph.get_node(target_object)
        )

    # 2. 检查部分匹配（相关对象）
    related_objects = goal_graph.get_related_objects()
    matched_related = []

    for obj in scene_graph.get_all_objects():
        if obj.label in related_objects:
            matched_related.append(obj)

    if len(matched_related) > 0:
        # 计算匹配分数（基于相关对象数量和置信度）
        score = min(0.8, len(matched_related) * 0.2)

        return MatchResult(
            score=score,
            exact_match=False,
            related_objects_found=True,
            matched_objects=matched_related
        )

    # 3. 零匹配
    return MatchResult(
        score=0.0,
        exact_match=False,
        related_objects_found=False,
        no_progress=True
    )
```

---

### **创新 4: 自我反思与策略适应**

#### 4.1 探索记忆

```python
class ExplorationMemory:
    """记录探索历史和失败案例"""

    def __init__(self):
        self.visited_regions = []
        self.failed_frontiers = []  # 访问过但无收获的 frontier
        self.belief_updates = []    # 信念更新历史

    def record_frontier_visit(self, frontier, outcome):
        """
        记录 frontier 访问结果

        outcome:
        - "found_target": 找到目标 ✓
        - "found_related": 找到相关对象
        - "nothing": 什么都没找到 ✗
        """
        visit_record = {
            "frontier": frontier,
            "position": frontier.position,
            "semantic_context": frontier.semantic_context,
            "outcome": outcome,
            "step": self.current_step,
            "expected_value": frontier.score,
            "actual_value": self._compute_actual_value(outcome)
        }

        self.visited_regions.append(visit_record)

        if outcome == "nothing":
            self.failed_frontiers.append(visit_record)

    def analyze_failures(self):
        """
        分析失败案例，提取教训

        问题：
        1. 为什么这些 frontier 失败了？
        2. 我们的预测哪里出错了？
        3. 应该调整什么策略？
        """
        if len(self.failed_frontiers) < 3:
            return None  # 样本不足

        # 用 LLM 分析失败模式
        prompt = f"""
        Goal: Find {self.goal_object}

        Failed exploration attempts:
        """

        for i, failure in enumerate(self.failed_frontiers[-5:]):
            prompt += f"""
        Attempt {i+1}:
        - Location: {failure['position']}
        - Context: {failure['semantic_context']}
        - Expected: {failure['expected_value']:.2f}
        - Result: Found nothing
        """

        prompt += """

        Questions:
        1. What pattern do you see in these failures?
        2. What assumptions were wrong?
        3. Where should we search instead?

        Provide:
        - Analysis: [pattern description]
        - Recommendation: [where to search next]
        - Confidence: [0-1]
        """

        analysis = self.llm.query(prompt)

        return self._parse_failure_analysis(analysis)
```

#### 4.2 信念修正

```python
class BeliefUpdater:
    """基于探索反馈更新信念"""

    def update_beliefs(self, failure_analysis, scene_graph):
        """
        根据失败分析更新场景图中的信念

        示例：
        失败分析：
        "在客厅探索了 3 次都没找到椅子，
         可能椅子在餐厅而不是客厅"

        更新：
        - 降低客厅中椅子的存在概率
        - 提高餐厅中椅子的存在概率
        """
        # 解析失败分析中提到的区域
        failed_rooms = failure_analysis.get("failed_locations")
        recommended_rooms = failure_analysis.get("recommended_locations")

        # 降低失败区域的预测对象概率
        for room in failed_rooms:
            predicted_objects = scene_graph.get_predicted_objects_in_room(room)
            for obj in predicted_objects:
                obj.existence_prob *= 0.5  # 大幅降低

        # 提高推荐区域的预测对象概率
        for room in recommended_rooms:
            # 创建或更新预测对象
            predicted_obj = scene_graph.get_or_create_predicted_object(
                label=self.goal_object,
                room=room
            )
            predicted_obj.existence_prob = min(1.0, predicted_obj.existence_prob + 0.3)
            predicted_obj.prediction_source = "llm_reflection"
```

#### 4.3 策略适应

```python
class AdaptiveStrategy:
    """动态调整探索策略"""

    def __init__(self):
        self.strategy_params = {
            "alpha": 0.5,  # BLIP2-ITM 权重
            "beta": 0.3,   # Graph matching 权重
            "gamma": 0.2,  # LLM reasoning 权重
            "exploration_bonus": 0.2,
            "llm_frequency": 5
        }

    def adapt_strategy(self, performance_metrics):
        """
        根据性能指标调整策略

        指标：
        - success_rate: 最近 N 次 frontier 选择的成功率
        - llm_accuracy: LLM 预测的准确率
        - time_efficiency: 时间效率
        """
        # 如果 LLM 预测经常错误，降低其权重
        if performance_metrics["llm_accuracy"] < 0.4:
            self.strategy_params["gamma"] *= 0.8
            self.strategy_params["alpha"] += 0.1  # 更依赖视觉

        # 如果 graph matching 效果好，提高其权重
        if performance_metrics["graph_match_accuracy"] > 0.7:
            self.strategy_params["beta"] *= 1.2

        # 如果探索效率低，增加探索奖励
        if performance_metrics["time_efficiency"] < 0.5:
            self.strategy_params["exploration_bonus"] += 0.1

        # 归一化权重
        total = (self.strategy_params["alpha"] +
                self.strategy_params["beta"] +
                self.strategy_params["gamma"])

        self.strategy_params["alpha"] /= total
        self.strategy_params["beta"] /= total
        self.strategy_params["gamma"] /= total
```

---

## 📊 完整算法流程

### Main Loop

```python
def asgfm_navigation_loop(self):
    """ASGFM 主导航循环"""

    # 初始化
    scene_graph = ProbabilisticSceneGraph()
    memory = ExplorationMemory()
    stage_controller = MultiStageController()
    belief_updater = BeliefUpdater()

    step = 0
    max_steps = 500

    while step < max_steps:
        # 1. 获取观测
        obs = self.get_observation()

        # 2. 更新场景图（增量式）
        detections = self.detect_objects(obs.rgb)
        scene_graph.update_from_observation(
            obs.rgb, obs.depth, detections, obs.pose, step
        )

        # 3. 检查是否找到目标
        if scene_graph.contains(self.goal_object):
            stage_controller.transition_to(ExplorationStage.PERFECT_MATCH)
            # 直接导航到目标
            target_pos = scene_graph.get_object_position(self.goal_object)
            action = self.navigate_to(target_pos)
            return action

        # 4. 更新探索阶段
        current_stage = stage_controller.update_stage(
            scene_graph,
            self.goal_object
        )

        # 5. 根据阶段获取策略
        strategy = stage_controller.get_exploration_strategy()

        # 6. 生成 frontiers
        geometric_frontiers = self.detect_geometric_frontiers(obs.depth)

        if strategy["use_semantic_frontiers"]:
            semantic_frontiers = self.generate_semantic_frontiers(
                scene_graph,
                self.goal_object,
                geometric_frontiers
            )
            all_frontiers = geometric_frontiers + semantic_frontiers
        else:
            all_frontiers = geometric_frontiers

        # 7. 评分和选择 frontier
        frontier_scores = self.score_frontiers(
            all_frontiers,
            scene_graph,
            strategy
        )

        best_frontier = self.select_best_frontier(
            all_frontiers,
            frontier_scores,
            memory  # 避免重复访问
        )

        # 8. 执行导航
        action = self.navigate_to_frontier(best_frontier)

        # 9. 记录结果
        outcome = self.evaluate_frontier_outcome(best_frontier)
        memory.record_frontier_visit(best_frontier, outcome)

        # 10. 自我反思（每 N 步）
        if step % 20 == 0:
            failure_analysis = memory.analyze_failures()
            if failure_analysis:
                belief_updater.update_beliefs(failure_analysis, scene_graph)

        # 11. 策略适应（每 M 步）
        if step % 50 == 0:
            performance = self.compute_performance_metrics(memory)
            self.adapt_strategy(performance)

        step += 1

    return STOP  # 超时
```

---

## 🎓 论文写作重点

### Abstract 模板

```
Zero-shot semantic navigation requires robots to efficiently locate
unseen objects in novel environments. Existing methods face a fundamental
trade-off: frontier-based approaches prioritize coverage but lack semantic
understanding, while graph-based methods leverage rich semantics but suffer
from exploration inefficiency.

We introduce ASGFM (Adaptive Scene Graph Frontier Maps), which transcends
this trade-off through four key innovations:

(1) Probabilistic scene graph construction with uncertainty modeling,
    enabling belief propagation over unobserved regions

(2) Semantic frontier generation via graph-guided reasoning, creating
    "virtual frontiers" in predicted high-value areas even before observation

(3) Multi-stage exploration matching that adapts between broad exploration,
    focused search, and direct navigation based on partial goal matching

(4) Self-reflective learning that analyzes failed attempts and dynamically
    adjusts exploration strategies

ASGFM achieves X% improvement in SPL on HM3D while reducing exploration
steps by Y%, and demonstrates emergent intelligent behaviors such as room
inference and failure-driven strategy adaptation.
```

### 核心贡献（Introduction）

```markdown
This paper makes the following contributions:

1. **Probabilistic Scene Graphs**: First method to model uncertainty
   in scene graphs for navigation, enabling reasoning about unobserved
   regions

2. **Semantic Frontier Generation**: Novel graph-guided frontier
   generation that predicts high-value exploration targets before
   direct observation

3. **Adaptive Multi-Stage Exploration**: Dynamic stage transitions
   (zero/partial/perfect match) that optimize exploration efficiency

4. **Self-Reflective Navigation**: First navigation system to
   analyze failures and adapt strategies during deployment

5. **State-of-the-Art Performance**: Achieves new SOTA on HM3D,
   MP3D, and Gibson benchmarks while providing interpretable
   decision-making
```

---

## 🧪 实验设计

### 实验 1: Main Results

**Baselines:**
- VLFM
- UniGoal
- PIRLNav
- ZSON

**Metrics:**
- Success Rate (SR)
- SPL
- Navigation Efficiency (steps to goal)
- **New**: Prediction Accuracy (语义 frontier 的准确率)

### 实验 2: Ablation Study

| Configuration | SR | SPL | Notes |
|--------------|----|----|-------|
| ASGFM (full) | 75% | 55% | 完整方法 |
| w/o semantic frontiers | 70% | 50% | 退化到几何 frontier |
| w/o uncertainty modeling | 72% | 52% | 确定性场景图 |
| w/o multi-stage | 68% | 48% | 固定探索策略 |
| w/o self-reflection | 71% | 51% | 无失败学习 |

### 实验 3: Semantic Frontier Evaluation

**问题**: 语义 frontier 的准确率如何？

**方法**:
- 记录所有生成的语义 frontier
- 检查后续是否在该区域找到目标或相关对象
- 计算精确率和召回率

**预期**:
- 精确率: 60-70%（语义 frontier 确实有用）
- 召回率: 80%+（能覆盖大部分有价值区域）

### 实验 4: Self-Reflection Case Study

**展示**:
- Episode 开始: 盲目探索
- Episode 中期: 3 次失败后的反思
- LLM 分析: "椅子不在客厅，应该搜索餐厅"
- 策略调整: 提高餐厅区域 frontier 分数
- Episode 结束: 在餐厅找到椅子 ✓

**可视化**:
- 场景图演化 GIF
- Frontier 分数热力图变化
- 探索轨迹 + 反思点标注

---

## 📈 预期性能提升

基于 VLFM (65% SR, 45% SPL):

```
ASGFM 预期:
- Success Rate: 72-75% (+7-10%)
- SPL: 52-55% (+7-10%)
- Avg Steps to Goal: 200 (-20%)
- 语义 Frontier 准确率: 65%
```

**Why?**
1. 语义 frontier 减少无效探索
2. 多阶段匹配加速目标定位
3. 自我反思避免重复错误
4. 不确定性建模提高推理质量

---

这就是 ASGFM 的完整设计！准备好开始实施了吗？
