# Copyright [2024] Your Institution
"""
GEFM Policy: Graph-Enhanced Frontier Maps Policy

Extends VLFM's ITMPolicyV2 with scene graph reasoning
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from vlfm.policy.itm_policy import ITMPolicyV2
from vlfm.mapping.scene_graph_map import SceneGraphMap
from vlfm.vlm.llm_reasoner import LLMReasoner


class GEFMPolicy(ITMPolicyV2):
    """
    Graph-Enhanced Frontier Maps Policy

    Combines:
    1. VLFM's BLIP2-ITM for fast frontier scoring
    2. Scene graph for structured semantic understanding
    3. LLM for spatial reasoning

    Scoring formula:
        score = α × BLIP2_score + β × graph_score + γ × LLM_score
    """

    def __init__(self, config):
        """
        Args:
            config: Hydra config object
        """
        super().__init__(config)

        # GEFM-specific components
        self.scene_graph = SceneGraphMap(
            max_nodes=config.gefm.sg_max_nodes,
            spatial_radius=config.gefm.sg_spatial_radius,
            confidence_decay=config.gefm.sg_confidence_decay
        )

        self.llm_reasoner = LLMReasoner(
            model_name=config.gefm.llm_model,
            backend=config.gefm.get("llm_backend", "openai"),
            api_key=config.gefm.get("llm_api_key", None),
            enable_cache=True
        )

        # Scoring weights (should sum to 1.0)
        self.alpha = config.gefm.alpha  # BLIP2-ITM weight
        self.beta = config.gefm.beta    # Graph matching weight
        self.gamma = config.gefm.gamma  # LLM reasoning weight

        assert abs(self.alpha + self.beta + self.gamma - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {self.alpha + self.beta + self.gamma}"

        # Dual-layer settings
        self.reasoning_interval = config.gefm.reasoning_interval
        self.confidence_threshold = config.gefm.confidence_threshold

        # Tracking
        self.step_counter = 0
        self.last_llm_scores: Dict[int, float] = {}
        self.last_llm_reasoning: Dict[int, str] = {}

        # Related objects cache (for graph matching)
        self.related_objects_cache: Dict[str, List[str]] = {}

        # Statistics
        self.gefm_stats = {
            "llm_calls": 0,
            "graph_updates": 0,
            "fast_layer_steps": 0,
            "reasoning_layer_steps": 0
        }

        print(f"[GEFM] Initialized with weights: α={self.alpha}, β={self.beta}, γ={self.gamma}")
        print(f"[GEFM] LLM reasoning every {self.reasoning_interval} steps")

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False
    ):
        """
        Main action selection method (called by Habitat)

        Extends parent's act() with scene graph updates
        """
        # Update scene graph from current observation
        self._update_scene_graph(observations)

        # Call parent's act method
        action, rnn_hidden_states = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic
        )

        self.step_counter += 1

        return action, rnn_hidden_states

    def _update_scene_graph(self, observations) -> None:
        """
        Update scene graph from current observations

        Args:
            observations: Habitat observations dict
        """
        # Extract RGB-D from observations
        rgb = observations.get("rgb", None)
        depth = observations.get("depth", None)

        if rgb is None or depth is None:
            return

        # Get object detections (from parent's detection pipeline)
        # This assumes parent class has run GroundingDINO/YOLOv7
        detections = getattr(self, "_last_detections", [])

        # Get camera pose (from parent's mapping)
        camera_pose = self._get_camera_pose()

        # Update scene graph
        self.scene_graph.update_from_observation(
            rgb=rgb,
            depth=depth,
            detections=detections,
            camera_pose=camera_pose,
            step=self.step_counter
        )

        self.gefm_stats["graph_updates"] += 1

    def _get_best_frontier(
        self,
        frontiers,
        frontier_values=None
    ):
        """
        Override parent's frontier selection with GEFM scoring

        Args:
            frontiers: List of frontier waypoints
            frontier_values: (Optional) Pre-computed values from parent

        Returns:
            Best frontier index
        """
        if len(frontiers) == 0:
            return None

        # Get RGB and goal text
        rgb = self._get_current_rgb()
        goal_text = self._get_goal_text()

        # Compute GEFM scores
        gefm_scores = self._score_frontiers_gefm(
            frontiers,
            rgb,
            goal_text
        )

        # Select best frontier (highest score)
        best_idx = np.argmax(gefm_scores)

        return best_idx

    def _score_frontiers_gefm(
        self,
        frontiers: List,
        rgb: np.ndarray,
        goal_text: str
    ) -> np.ndarray:
        """
        GEFM three-way scoring mechanism

        Args:
            frontiers: List of frontier waypoints
            rgb: Current RGB image
            goal_text: Target object description

        Returns:
            Array of scores for each frontier
        """
        num_frontiers = len(frontiers)
        scores = np.zeros(num_frontiers)

        # 1. BLIP2-ITM scores (α component) - Fast, every step
        blip2_scores = self._get_blip2_itm_scores(frontiers, rgb, goal_text)

        # 2. Graph matching scores (β component) - Fast, every step
        graph_scores = self._get_graph_matching_scores(frontiers, goal_text)

        # 3. LLM reasoning scores (γ component) - Slow, periodic
        llm_scores = self._get_llm_reasoning_scores(frontiers, goal_text)

        # Weighted combination
        for i in range(num_frontiers):
            scores[i] = (
                self.alpha * blip2_scores[i] +
                self.beta * graph_scores[i] +
                self.gamma * llm_scores[i]
            )

        # Track execution layer
        if self.step_counter % self.reasoning_interval == 0:
            self.gefm_stats["reasoning_layer_steps"] += 1
        else:
            self.gefm_stats["fast_layer_steps"] += 1

        return scores

    def _get_blip2_itm_scores(
        self,
        frontiers: List,
        rgb: np.ndarray,
        goal_text: str
    ) -> np.ndarray:
        """
        Get BLIP2-ITM scores (original VLFM approach)

        Args:
            frontiers: List of frontiers
            rgb: RGB image
            goal_text: Target object

        Returns:
            Array of BLIP2-ITM scores
        """
        # Use parent's value map scoring
        # This is a simplified version; actual implementation should
        # extract scores from parent's value map
        num_frontiers = len(frontiers)
        scores = np.zeros(num_frontiers)

        for i, frontier in enumerate(frontiers):
            # Get value from parent's value map at frontier position
            # Placeholder: need to implement proper value map query
            scores[i] = 0.5  # TODO: Query actual value map

        return scores

    def _get_graph_matching_scores(
        self,
        frontiers: List,
        goal_text: str
    ) -> np.ndarray:
        """
        Score frontiers based on scene graph semantic context

        Args:
            frontiers: List of frontiers
            goal_text: Target object

        Returns:
            Array of graph-based scores
        """
        num_frontiers = len(frontiers)
        scores = np.zeros(num_frontiers)

        # Parse target object from goal text
        goal_object = self._parse_goal_object(goal_text)

        # Get related objects (with caching)
        related_objects = self._get_related_objects(goal_object)

        for i, frontier in enumerate(frontiers):
            # Get semantic context around frontier
            frontier_pos = self._get_frontier_position(frontier)
            context = self.scene_graph.get_semantic_context(
                frontier_pos,
                radius=self.scene_graph.spatial_radius
            )

            # Scoring logic:
            # - High score if target object mentioned in context
            # - Medium score if related objects present
            # - Low score otherwise

            if goal_object.lower() in context.lower():
                scores[i] = 1.0
            else:
                # Check for related objects
                match_count = sum(
                    1 for obj in related_objects
                    if obj.lower() in context.lower()
                )

                if match_count > 0:
                    scores[i] = 0.5 + 0.3 * min(match_count / len(related_objects), 1.0)
                else:
                    scores[i] = 0.3  # Unexplored or unrelated area

        return scores

    def _get_llm_reasoning_scores(
        self,
        frontiers: List,
        goal_text: str
    ) -> np.ndarray:
        """
        Get LLM-based reasoning scores (periodic, expensive)

        Args:
            frontiers: List of frontiers
            goal_text: Target object

        Returns:
            Array of LLM reasoning scores
        """
        num_frontiers = len(frontiers)

        # Check if we should invoke LLM
        should_reason = self._should_invoke_llm_reasoning()

        if should_reason:
            # Prepare frontier data for LLM
            frontier_data = []
            for i, frontier in enumerate(frontiers):
                pos = self._get_frontier_position(frontier)
                context = self.scene_graph.get_semantic_context(pos)

                frontier_data.append({
                    "id": i,
                    "position": pos[:2].tolist(),  # [x, y]
                    "context": context
                })

            # Get scene graph text
            sg_text = self.scene_graph.to_text()

            # Query LLM
            llm_results = self.llm_reasoner.score_frontiers(
                frontiers=frontier_data,
                scene_graph_text=sg_text,
                goal_text=goal_text
            )

            # Update cache
            self.last_llm_scores = {
                fid: score for fid, (score, _) in llm_results.items()
            }
            self.last_llm_reasoning = {
                fid: reason for fid, (_, reason) in llm_results.items()
            }

            self.gefm_stats["llm_calls"] += 1

        # Return scores from cache
        scores = np.array([
            self.last_llm_scores.get(i, 0.5)
            for i in range(num_frontiers)
        ])

        return scores

    def _should_invoke_llm_reasoning(self) -> bool:
        """
        Decide whether to invoke LLM reasoning

        Conditions:
        1. Reached reasoning interval
        2. Low confidence in fast layer
        3. Stuck (cyclic exploration detected)
        """
        # Condition 1: Fixed interval
        if self.step_counter % self.reasoning_interval == 0:
            return True

        # Condition 2: Low confidence
        # (Need to track fast layer confidence from previous step)
        # TODO: Implement low-confidence detection

        return False

    def _parse_goal_object(self, goal_text: str) -> str:
        """
        Extract target object from goal text

        Args:
            goal_text: e.g., "chair", "red chair", "chair near table"

        Returns:
            Primary object name
        """
        # Simple parsing: take first word
        # TODO: Use NLP for better parsing
        return goal_text.split()[0].lower()

    def _get_related_objects(self, object_name: str) -> List[str]:
        """
        Get objects commonly found near target object

        Uses LLM with caching

        Args:
            object_name: Target object

        Returns:
            List of related object names
        """
        if object_name in self.related_objects_cache:
            return self.related_objects_cache[object_name]

        # Query LLM for related objects
        related = self.llm_reasoner.get_related_objects(object_name, max_count=5)

        # Extract object names
        related_names = [obj for obj, _ in related]

        # Cache result
        self.related_objects_cache[object_name] = related_names

        return related_names

    def _get_frontier_position(self, frontier) -> np.ndarray:
        """
        Get 3D position of frontier

        Args:
            frontier: Frontier waypoint

        Returns:
            Position array [x, y, z]
        """
        # TODO: Extract position from frontier object
        # This depends on frontier data structure from parent
        return np.array([0.0, 0.0, 0.0])  # Placeholder

    def _get_camera_pose(self) -> np.ndarray:
        """
        Get current camera pose

        Returns:
            4x4 transformation matrix
        """
        # TODO: Get from parent's mapping system
        return np.eye(4)  # Placeholder

    def _get_current_rgb(self) -> np.ndarray:
        """Get current RGB observation"""
        # TODO: Store from latest observation
        return np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder

    def _get_goal_text(self) -> str:
        """Get current goal text"""
        # TODO: Extract from episode info
        return "chair"  # Placeholder

    def get_gefm_statistics(self) -> Dict:
        """Get GEFM-specific statistics"""
        return {
            **self.gefm_stats,
            "scene_graph_stats": self.scene_graph.get_statistics(),
            "llm_stats": self.llm_reasoner.get_statistics()
        }

    def visualize_gefm(self, save_dir: str) -> None:
        """
        Save GEFM visualizations

        Args:
            save_dir: Directory to save visualizations
        """
        import os

        # 1. Visualize scene graph
        sg_path = os.path.join(save_dir, f"scene_graph_step_{self.step_counter}.png")
        self.scene_graph.visualize(sg_path)

        # 2. Visualize frontier scores breakdown
        # TODO: Create visualization comparing α, β, γ components

        # 3. Save LLM reasoning log
        reasoning_path = os.path.join(save_dir, "llm_reasoning.txt")
        with open(reasoning_path, 'a') as f:
            f.write(f"\n=== Step {self.step_counter} ===\n")
            for fid, reason in self.last_llm_reasoning.items():
                f.write(f"Frontier {fid}: {reason}\n")
