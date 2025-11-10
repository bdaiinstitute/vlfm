# Copyright [2024] Your Institution
"""
Probabilistic Scene Graph for ASGFM
Maintains scene graph with uncertainty modeling and belief propagation
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Extended node types with uncertainty support"""
    # Observed nodes
    OBJECT = "object"
    ROOM = "room"

    # Predicted nodes (with uncertainty)
    PREDICTED_OBJECT = "predicted_object"
    VIRTUAL_ROOM = "virtual_room"
    FRONTIER = "frontier"


class PredictionSource(Enum):
    """Source of node prediction"""
    OBSERVATION = "observation"  # Directly observed
    INFERENCE = "inference"      # Inferred from spatial relations
    LLM = "llm"                  # Predicted by LLM
    PRIOR = "prior"              # From prior knowledge


@dataclass
class ProbabilisticNode:
    """
    Node with uncertainty modeling

    Key Innovation: Track existence probability and position uncertainty
    """
    node_id: int
    node_type: NodeType
    label: str  # e.g., "chair", "kitchen"
    position: np.ndarray  # Expected position [x, y, z]

    # Uncertainty modeling
    existence_prob: float = 0.5    # P(node exists) ∈ [0, 1]
    position_variance: float = 1.0  # Uncertainty in position (meters²)
    confidence: float = 0.5         # Observation confidence

    # Temporal information
    first_seen: int = 0
    last_updated: int = 0
    observation_count: int = 0

    # Prediction metadata
    prediction_source: PredictionSource = PredictionSource.OBSERVATION
    prediction_reason: str = ""

    def __post_init__(self):
        """Validate probability values"""
        assert 0 <= self.existence_prob <= 1, "existence_prob must be in [0, 1]"
        assert 0 <= self.confidence <= 1, "confidence must be in [0, 1]"
        assert self.position_variance >= 0, "variance must be non-negative"


@dataclass
class ProbabilisticEdge:
    """Edge with confidence"""
    source_id: int
    target_id: int
    relation: str  # "near", "in_room", "left_of", etc.
    confidence: float = 1.0
    distance: float = 0.0  # For spatial relations


class ProbabilisticSceneGraph:
    """
    Scene graph with uncertainty modeling and belief propagation

    Key Features:
    1. Probabilistic nodes (existence probability)
    2. Bayesian updates from observations
    3. Belief propagation through graph structure
    4. Temporal consistency
    """

    def __init__(
        self,
        max_nodes: int = 200,
        initial_existence_prob: float = 0.5,
        observation_boost: float = 0.3,
        decay_rate: float = 0.98,
        min_existence_prob: float = 0.1
    ):
        """
        Args:
            max_nodes: Maximum number of nodes
            initial_existence_prob: Initial P(exists) for predicted nodes
            observation_boost: Probability boost when observed
            decay_rate: Decay rate per step for unobserved nodes
            min_existence_prob: Minimum probability before pruning
        """
        self.max_nodes = max_nodes
        self.initial_existence_prob = initial_existence_prob
        self.observation_boost = observation_boost
        self.decay_rate = decay_rate
        self.min_existence_prob = min_existence_prob

        # Core data structures
        self.nodes: Dict[int, ProbabilisticNode] = {}
        self.edges: Dict[Tuple[int, int], ProbabilisticEdge] = {}
        self.graph = nx.DiGraph()

        # Tracking
        self.next_node_id = 0
        self.current_step = 0

        # Statistics
        self.stats = {
            "total_nodes_created": 0,
            "predicted_nodes": 0,
            "observed_nodes": 0,
            "belief_updates": 0,
            "nodes_pruned": 0
        }

    def add_observed_node(
        self,
        label: str,
        position: np.ndarray,
        confidence: float,
        node_type: NodeType = NodeType.OBJECT
    ) -> int:
        """
        Add a node from direct observation

        Args:
            label: Object/room label
            position: 3D position
            confidence: Detection confidence

        Returns:
            Node ID
        """
        # Check if similar node exists nearby
        existing_id = self._find_nearby_node(position, label, threshold=0.5)

        if existing_id is not None:
            # Update existing node
            return self._update_observed_node(existing_id, position, confidence)
        else:
            # Create new node
            return self._create_new_node(
                label=label,
                position=position,
                node_type=node_type,
                existence_prob=min(1.0, 0.7 + confidence * 0.3),
                confidence=confidence,
                prediction_source=PredictionSource.OBSERVATION
            )

    def add_predicted_node(
        self,
        label: str,
        position: np.ndarray,
        existence_prob: float,
        node_type: NodeType = NodeType.PREDICTED_OBJECT,
        source: PredictionSource = PredictionSource.INFERENCE,
        reason: str = ""
    ) -> int:
        """
        Add a predicted node (not directly observed)

        This is a key innovation: reasoning about unobserved objects
        """
        node_id = self._create_new_node(
            label=label,
            position=position,
            node_type=node_type,
            existence_prob=existence_prob,
            confidence=0.0,  # Not yet confirmed
            prediction_source=source,
            position_variance=2.0,  # High uncertainty
            observation_count=0,
            prediction_reason=reason
        )

        self.stats["predicted_nodes"] += 1

        return node_id

    def bayesian_update(
        self,
        node_id: int,
        observation_result: str,
        likelihood_model: Optional[Dict] = None
    ):
        """
        Bayesian update of node existence probability

        P(exists | obs) ∝ P(obs | exists) × P(exists)

        Args:
            node_id: Node to update
            observation_result: "confirmed" / "not_found" / "out_of_view"
            likelihood_model: Optional custom likelihood model
        """
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        if observation_result == "confirmed":
            # Observed the object → strong evidence for existence
            node.existence_prob = min(1.0, node.existence_prob + self.observation_boost)
            node.confidence = min(1.0, node.confidence + 0.2)
            node.observation_count += 1

            # Reduce position uncertainty
            node.position_variance *= 0.5

            # Upgrade predicted node to observed
            if node.node_type == NodeType.PREDICTED_OBJECT:
                node.node_type = NodeType.OBJECT
                node.prediction_source = PredictionSource.OBSERVATION

        elif observation_result == "not_found":
            # Should have seen it but didn't → evidence against existence
            node.existence_prob *= 0.6  # Strong penalty
            node.confidence *= 0.8

        elif observation_result == "out_of_view":
            # Not in current view → slow decay
            node.existence_prob *= self.decay_rate
            node.confidence *= 0.98

        node.last_updated = self.current_step
        self.stats["belief_updates"] += 1

        # Prune if probability too low
        if node.existence_prob < self.min_existence_prob:
            self._prune_node(node_id)

    def propagate_beliefs(self):
        """
        Propagate beliefs through graph structure

        Rules:
        1. If A is near B and A has high existence prob, boost B
        2. If room type is confirmed, boost typical objects in room
        3. Temporal consistency: object positions don't jump
        """
        # Rule 1: Spatial correlation
        for (source_id, target_id), edge in self.edges.items():
            if edge.relation == "near":
                source = self.nodes[source_id]
                target = self.nodes[target_id]

                # If source highly likely to exist, boost target
                if source.existence_prob > 0.8:
                    boost = 0.05 * source.existence_prob * edge.confidence
                    target.existence_prob = min(1.0, target.existence_prob + boost)

        # Rule 2: Room-object correlations
        for node in self.nodes.values():
            if node.node_type == NodeType.ROOM and node.existence_prob > 0.7:
                # Boost typical objects in this room
                self._boost_typical_room_objects(node)

    def get_nodes_by_type(self, node_type: NodeType) -> List[ProbabilisticNode]:
        """Get all nodes of a specific type"""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_predicted_objects(self) -> List[ProbabilisticNode]:
        """Get all predicted (unconfirmed) objects"""
        return [
            n for n in self.nodes.values()
            if n.node_type == NodeType.PREDICTED_OBJECT
            and n.existence_prob > self.min_existence_prob
        ]

    def get_high_confidence_predictions(
        self,
        threshold: float = 0.7
    ) -> List[ProbabilisticNode]:
        """Get predicted nodes with high existence probability"""
        return [
            n for n in self.nodes.values()
            if n.node_type in [NodeType.PREDICTED_OBJECT, NodeType.VIRTUAL_ROOM]
            and n.existence_prob >= threshold
        ]

    def contains(self, label: str, min_prob: float = 0.7) -> bool:
        """Check if object with label exists (with high probability)"""
        for node in self.nodes.values():
            if node.label == label and node.existence_prob >= min_prob:
                return True
        return False

    def get_object_position(
        self,
        label: str,
        min_prob: float = 0.7
    ) -> Optional[np.ndarray]:
        """Get position of object (if exists with sufficient probability)"""
        for node in self.nodes.values():
            if node.label == label and node.existence_prob >= min_prob:
                return node.position
        return None

    def to_text(self) -> str:
        """
        Convert to text for LLM input

        Format optimized for LLM reasoning
        """
        text = "Probabilistic Scene Graph:\n\n"

        # Observed objects
        text += "Confirmed Objects:\n"
        confirmed = [n for n in self.nodes.values()
                    if n.node_type == NodeType.OBJECT
                    and n.existence_prob > 0.8]

        for node in confirmed:
            text += f"  - {node.label} at ({node.position[0]:.1f}, {node.position[1]:.1f}), "
            text += f"confidence={node.existence_prob:.2f}\n"

        # Predicted objects
        text += "\nPredicted Objects (unconfirmed):\n"
        predicted = [n for n in self.nodes.values()
                    if n.node_type == NodeType.PREDICTED_OBJECT
                    and n.existence_prob > 0.5]

        for node in predicted[:5]:  # Limit to top 5
            text += f"  - {node.label} (P={node.existence_prob:.2f}), "
            text += f"reason: {node.prediction_reason}\n"

        # Rooms
        text += "\nInferred Rooms:\n"
        rooms = [n for n in self.nodes.values()
                if n.node_type in [NodeType.ROOM, NodeType.VIRTUAL_ROOM]]

        for room in rooms:
            text += f"  - {room.label} (P={room.existence_prob:.2f})\n"

        return text

    # ===== Private methods =====

    def _create_new_node(
        self,
        label: str,
        position: np.ndarray,
        node_type: NodeType,
        existence_prob: float,
        confidence: float,
        prediction_source: PredictionSource,
        position_variance: float = 0.5,
        observation_count: int = 1,
        prediction_reason: str = ""
    ) -> int:
        """Create a new node"""
        node_id = self.next_node_id
        self.next_node_id += 1

        node = ProbabilisticNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            position=position.copy(),
            existence_prob=existence_prob,
            position_variance=position_variance,
            confidence=confidence,
            first_seen=self.current_step,
            last_updated=self.current_step,
            observation_count=observation_count,
            prediction_source=prediction_source,
            prediction_reason=prediction_reason
        )

        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.__dict__)

        self.stats["total_nodes_created"] += 1
        if prediction_source == PredictionSource.OBSERVATION:
            self.stats["observed_nodes"] += 1

        return node_id

    def _update_observed_node(
        self,
        node_id: int,
        position: np.ndarray,
        confidence: float
    ) -> int:
        """Update an existing observed node"""
        node = self.nodes[node_id]

        # Bayesian position update (weighted average)
        weight_old = node.observation_count
        weight_new = 1
        total_weight = weight_old + weight_new

        node.position = (weight_old * node.position + weight_new * position) / total_weight

        # Update probability and confidence
        node.existence_prob = min(1.0, node.existence_prob + 0.1)
        node.confidence = max(node.confidence, confidence)
        node.observation_count += 1
        node.last_updated = self.current_step

        # Reduce uncertainty
        node.position_variance *= 0.8

        return node_id

    def _find_nearby_node(
        self,
        position: np.ndarray,
        label: str,
        threshold: float = 0.5
    ) -> Optional[int]:
        """Find node with same label within threshold distance"""
        for node_id, node in self.nodes.items():
            if node.label == label:
                dist = np.linalg.norm(node.position[:2] - position[:2])  # 2D distance
                if dist < threshold:
                    return node_id
        return None

    def _prune_node(self, node_id: int):
        """Remove node with low existence probability"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.graph.remove_node(node_id)
            self.stats["nodes_pruned"] += 1

            # Remove associated edges
            edges_to_remove = [
                (s, t) for (s, t) in self.edges.keys()
                if s == node_id or t == node_id
            ]
            for edge_key in edges_to_remove:
                del self.edges[edge_key]

    def _boost_typical_room_objects(self, room_node: ProbabilisticNode):
        """Boost probability of objects typically found in this room"""
        # TODO: Implement room-object associations
        # E.g., kitchen → microwave, refrigerator, sink
        #       bedroom → bed, nightstand, lamp
        pass

    def step(self):
        """Advance time step (for temporal decay)"""
        self.current_step += 1

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            **self.stats,
            "current_nodes": len(self.nodes),
            "current_edges": len(self.edges),
            "observed_ratio": self.stats["observed_nodes"] / max(1, self.stats["total_nodes_created"]),
            "predicted_ratio": self.stats["predicted_nodes"] / max(1, self.stats["total_nodes_created"])
        }
