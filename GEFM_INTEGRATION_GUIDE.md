# GEFM Integration Guide

## 🚀 Quick Start

### 1. Install Additional Dependencies

```bash
# LLM backends
pip install openai anthropic  # For cloud LLMs
pip install ollama             # For local LLMs

# Scene graph visualization
pip install networkx matplotlib

# Experiment tracking
pip install wandb tensorboard
```

### 2. Set Up API Keys (for Cloud LLMs)

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-api-key-here"

# Option 2: In config file
# Edit config/experiments/gefm_objectnav_hm3d.yaml
# Set gefm.llm_api_key: "your-api-key-here"
```

**Cost-Saving Option**: Use local LLMs with Ollama (free):
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3

# Update config:
# gefm.llm_backend: "ollama"
# gefm.llm_model: "llama3"
```

### 3. Test GEFM Components

```bash
# Test scene graph
python test/test_scene_graph.py

# Test LLM reasoner
python test/test_llm_reasoner.py

# Test GEFM policy (dry run)
python test/test_gefm_policy.py
```

### 4. Run GEFM on HM3D

```bash
# Make sure VLM servers are running (from VLFM)
./scripts/launch_vlm_servers.sh

# Run GEFM
python -m vlfm.run \
  --config-name gefm_objectnav_hm3d \
  num_episodes=10
```

---

## 🔧 Integration Details

### Modified Files

**New Files Created**:
- `vlfm/mapping/scene_graph_map.py` - Scene graph construction
- `vlfm/vlm/llm_reasoner.py` - LLM integration
- `vlfm/policy/gefm_policy.py` - GEFM policy
- `config/experiments/gefm_objectnav_hm3d.yaml` - GEFM config

**Files to Modify** (TODO):
1. `vlfm/run.py` - Add GEFM policy option
2. `vlfm/policy/__init__.py` - Export GEFMPolicy
3. `vlfm/mapping/__init__.py` - Export SceneGraphMap
4. `vlfm/vlm/__init__.py` - Export LLMReasoner

### Integration Checklist

- [ ] Import GEFM modules in `__init__.py` files
- [ ] Update `vlfm/run.py` to support GEFM policy
- [ ] Implement `_get_frontier_position()` in GEFMPolicy
- [ ] Implement `_get_camera_pose()` using VLFM's mapping
- [ ] Connect VLFM's detections to scene graph
- [ ] Add GEFM-specific logging
- [ ] Create visualization pipeline

---

## 📝 Step-by-Step Integration

### Step 1: Update Module Imports

**File**: `vlfm/mapping/__init__.py`
```python
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.value_map import ValueMap
from vlfm.mapping.scene_graph_map import SceneGraphMap  # Add this

__all__ = ["ObstacleMap", "ValueMap", "SceneGraphMap"]
```

**File**: `vlfm/vlm/__init__.py`
```python
# ... existing imports ...
from vlfm.vlm.llm_reasoner import LLMReasoner  # Add this

__all__ = [
    # ... existing exports ...
    "LLMReasoner"
]
```

**File**: `vlfm/policy/__init__.py`
```python
from vlfm.policy.itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3
from vlfm.policy.gefm_policy import GEFMPolicy  # Add this

__all__ = ["ITMPolicy", "ITMPolicyV2", "ITMPolicyV3", "GEFMPolicy"]
```

### Step 2: Update vlfm/run.py

**File**: `vlfm/run.py`

Find the policy initialization section and add:

```python
from vlfm.policy.gefm_policy import GEFMPolicy

# In the policy selection logic (around line 50-70):
def create_policy(config):
    policy_name = config.policy.name

    if policy_name == "ITMPolicy":
        return ITMPolicy(config)
    elif policy_name == "ITMPolicyV2":
        return ITMPolicyV2(config)
    elif policy_name == "ITMPolicyV3":
        return ITMPolicyV3(config)
    elif policy_name == "GEFMPolicy":  # Add this
        return GEFMPolicy(config)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
```

### Step 3: Connect to VLFM's Detection Pipeline

**File**: `vlfm/policy/gefm_policy.py`

Modify `_update_scene_graph()` to properly extract detections:

```python
def _update_scene_graph(self, observations) -> None:
    """Update scene graph from current observations"""
    rgb = observations.get("rgb", None)
    depth = observations.get("depth", None)

    if rgb is None or depth is None:
        return

    # Access parent's object detection results
    # VLFM stores detections in self.object_map or similar
    # You need to find where parent stores GroundingDINO results

    # Option 1: If parent has object_map
    if hasattr(self, 'object_map'):
        detections = self._convert_object_map_to_detections()

    # Option 2: Run detection here (expensive)
    else:
        detections = self._run_object_detection(rgb)

    # Get camera pose from parent's mapping
    if hasattr(self, 'obstacle_map'):
        camera_pose = self.obstacle_map.get_camera_pose()
    else:
        camera_pose = np.eye(4)

    self.scene_graph.update_from_observation(
        rgb=rgb,
        depth=depth,
        detections=detections,
        camera_pose=camera_pose,
        step=self.step_counter
    )
```

### Step 4: Implement Helper Methods

**Critical TODOs in gefm_policy.py**:

1. **Extract frontier positions**:
```python
def _get_frontier_position(self, frontier) -> np.ndarray:
    """Get 3D position of frontier"""
    # Frontiers from VLFM are 2D grid coordinates
    # Convert to meters using map resolution
    if hasattr(self, 'obstacle_map'):
        position_2d = frontier  # (x, y) in grid coordinates
        resolution = self.obstacle_map.resolution
        position_meters = position_2d * resolution

        # Add z=0 for 3D
        return np.array([position_meters[0], position_meters[1], 0.0])

    return np.array([0.0, 0.0, 0.0])
```

2. **Get BLIP2-ITM scores from value map**:
```python
def _get_blip2_itm_scores(self, frontiers, rgb, goal_text) -> np.ndarray:
    """Extract scores from parent's value map"""
    num_frontiers = len(frontiers)
    scores = np.zeros(num_frontiers)

    # Access parent's value_map
    if hasattr(self, 'value_map'):
        for i, frontier in enumerate(frontiers):
            # Query value at frontier position
            x, y = frontier  # Grid coordinates
            scores[i] = self.value_map.map[y, x]  # Note: may need to flip x/y

    return scores
```

### Step 5: Add Visualization

**File**: `vlfm/policy/gefm_policy.py`

Implement `visualize_gefm()` properly:

```python
def visualize_gefm(self, save_dir: str) -> None:
    """Save GEFM visualizations"""
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # 1. Scene graph visualization
    if hasattr(self.scene_graph, 'visualize'):
        sg_path = os.path.join(save_dir, f"scene_graph_{self.step_counter:04d}.png")

        # Simple text-based visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, self.scene_graph.to_text(),
                ha='center', va='center', fontfamily='monospace')
        ax.axis('off')
        plt.savefig(sg_path)
        plt.close()

    # 2. Frontier scores breakdown
    # TODO: Implement bar chart comparing α, β, γ components
```

---

## 🧪 Testing Strategy

### Unit Tests

**File**: `test/test_scene_graph.py`
```python
import numpy as np
from vlfm.mapping.scene_graph_map import SceneGraphMap, NodeType

def test_scene_graph_creation():
    sg = SceneGraphMap()

    # Mock detection
    detection = {
        "label": "chair",
        "score": 0.9,
        "box": np.array([100, 100, 200, 200]),
        "mask": None
    }

    depth = np.ones((480, 640)) * 2.0  # 2 meters
    camera_pose = np.eye(4)

    # Update scene graph
    sg.update_from_observation(
        rgb=np.zeros((480, 640, 3)),
        depth=depth,
        detections=[detection],
        camera_pose=camera_pose,
        step=0
    )

    assert len(sg.nodes) == 1
    print("✓ Scene graph creation test passed")

if __name__ == "__main__":
    test_scene_graph_creation()
```

**File**: `test/test_llm_reasoner.py`
```python
from vlfm.vlm.llm_reasoner import LLMReasoner

def test_llm_reasoner():
    # Use mock backend for testing
    class MockBackend:
        def query(self, prompt, max_tokens=200, temperature=0.3):
            return '{"frontier_0": {"score": 0.8, "reason": "Test"}}'

    reasoner = LLMReasoner(model_name="test", backend="openai")
    reasoner.backend = MockBackend()

    frontiers = [{"id": 0, "position": [0, 0], "context": "near chair"}]
    scores = reasoner.score_frontiers(frontiers, "Scene: chair", "chair")

    assert 0 in scores
    print(f"✓ LLM reasoner test passed: {scores}")

if __name__ == "__main__":
    test_llm_reasoner()
```

### Integration Test

```bash
# Test on a single episode
python -m vlfm.run \
  --config-name gefm_objectnav_hm3d \
  num_episodes=1 \
  gefm.llm_backend=ollama \
  gefm.llm_model=llama3
```

---

## 📊 Running Experiments

### Baseline Comparison

```bash
# 1. VLFM baseline
python -m vlfm.run \
  --config-name vlfm_objectnav_hm3d \
  num_episodes=100

# 2. GEFM (full)
python -m vlfm.run \
  --config-name gefm_objectnav_hm3d \
  num_episodes=100

# 3. GEFM w/o scene graph (ablation)
python -m vlfm.run \
  --config-name gefm_objectnav_hm3d \
  gefm.disable_graph_matching=True \
  num_episodes=100

# 4. GEFM w/o LLM (ablation)
python -m vlfm.run \
  --config-name gefm_objectnav_hm3d \
  gefm.disable_llm_reasoning=True \
  num_episodes=100
```

### Parameter Sweep

```bash
# Sweep α, β, γ weights
for alpha in 0.3 0.5 0.7; do
  beta=$(python -c "print(round((1.0 - $alpha) * 0.6, 2))")
  gamma=$(python -c "print(round(1.0 - $alpha - $beta, 2))")

  python -m vlfm.run \
    --config-name gefm_objectnav_hm3d \
    gefm.alpha=$alpha \
    gefm.beta=$beta \
    gefm.gamma=$gamma \
    num_episodes=50
done
```

---

## 📈 Evaluation Metrics

### Primary Metrics (from VLFM)
- **Success Rate (SR)**: % of episodes reaching goal
- **SPL**: Success weighted by Path Length
- **Distance to Goal**: Average final distance

### GEFM-Specific Metrics
- **LLM Call Frequency**: Actual vs configured
- **Scene Graph Size**: Nodes/edges over time
- **Component Contribution**: Correlation of α, β, γ scores with success

### Performance Metrics
- **Average Episode Time**: Total time per episode
- **LLM Latency**: Average time per LLM call
- **Graph Update Time**: Time to update scene graph

---

## 🐛 Troubleshooting

### Common Issues

**1. LLM API errors**
```
Error: OpenAI API rate limit exceeded
```
**Solution**: Use Ollama locally or reduce `reasoning_interval`

**2. Scene graph grows too large**
```
Warning: Scene graph has 200 nodes
```
**Solution**: Reduce `sg_max_nodes` or increase `sg_min_confidence`

**3. Low graph matching scores**
```
All frontiers have β score = 0.3
```
**Solution**: Check object detection quality, verify scene graph is being updated

**4. Import errors**
```
ModuleNotFoundError: No module named 'vlfm.policy.gefm_policy'
```
**Solution**: Update `__init__.py` files as described in Step 1

---

## 🎯 Next Steps

1. **Complete integration** (Steps 1-4 above)
2. **Run unit tests** to verify components work
3. **Test on 1 episode** to debug
4. **Run baseline comparison** (10 episodes)
5. **Full evaluation** (100 episodes)
6. **Ablation studies**
7. **Parameter tuning**
8. **Write paper** 🎉

---

## 📚 Additional Resources

- **VLFM Paper**: https://arxiv.org/abs/2312.03275
- **UniGoal Paper**: https://arxiv.org/abs/2503.10630
- **Habitat Docs**: https://aihabitat.org/docs/
- **BLIP-2 Paper**: https://arxiv.org/abs/2301.12597

---

## 💡 Tips

- Start with Ollama (free) for development
- Use small `num_episodes` initially (1-10)
- Monitor LLM costs if using OpenAI
- Visualize scene graph frequently to debug
- Keep `reasoning_interval` high (10+) initially
- Consider caching LLM responses aggressively
