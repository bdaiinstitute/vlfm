# Copyright [2024] Your Institution
"""
Scene Graph Map for GEFM
Maintains an online scene graph from RGB-D observations
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the scene graph"""
    OBJECT = "object"
    ROOM = "room"
    FRONTIER = "frontier"


class RelationType(Enum):
    """Types of edges in the scene graph"""
    # Spatial relations
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    NEAR = "near"
    FAR = "far"

    # Semantic relations
    PART_OF = "part_of"
    TYPICALLY_NEAR = "typically_near"
    IN_ROOM = "in_room"


@dataclass
class GraphNode:
    """Represents a node in the scene graph"""
    node_id: int
    node_type: NodeType
    position: np.ndarray  # [x, y, z] in global frame
    label: str  # e.g., "chair", "living_room", "frontier_3"
    confidence: float  # 0-1
    attributes: Dict  # Additional attributes (color, size, etc.)
    timestamp: int  # When first observed


@dataclass
class GraphEdge:
    """Represents an edge in the scene graph"""
    source_id: int
    target_id: int
    relation: RelationType
    confidence: float
    timestamp: int


class SceneGraphMap:
    """
    Online scene graph construction and maintenance

    Integrates with VLFM's mapping system to add semantic structure
    """

    def __init__(
        self,
        max_nodes: int = 100,
        spatial_radius: float = 2.0,
        confidence_decay: float = 0.95,
        min_confidence: float = 0.3
    ):
        """
        Args:
            max_nodes: Maximum number of nodes to maintain
            spatial_radius: Radius for spatial relation detection (meters)
            confidence_decay: Decay factor for node confidence over time
            min_confidence: Minimum confidence to keep a node
        """
        self.max_nodes = max_nodes
        self.spatial_radius = spatial_radius
        self.confidence_decay = confidence_decay
        self.min_confidence = min_confidence

        # Core data structures
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: Dict[Tuple[int, int], GraphEdge] = {}
        self.graph = nx.DiGraph()  # NetworkX graph for advanced algorithms

        # Tracking
        self.next_node_id = 0
        self.current_step = 0

        # Statistics
        self.stats = {
            "total_nodes_created": 0,
            "total_edges_created": 0,
            "nodes_pruned": 0
        }

    def update_from_observation(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        detections: List,  # From GroundingDINO/YOLOv7
        camera_pose: np.ndarray,
        step: int
    ) -> None:
        """
        Update scene graph from current observation

        Args:
            rgb: RGB image (H, W, 3)
            depth: Depth image (H, W)
            detections: List of object detections with boxes, labels, scores
            camera_pose: 4x4 transformation matrix (camera to world)
            step: Current timestep
        """
        self.current_step = step

        # 1. Add/update object nodes from detections
        new_node_ids = []
        for det in detections:
            node_id = self._add_or_update_object_node(det, depth, camera_pose)
            if node_id is not None:
                new_node_ids.append(node_id)

        # 2. Infer spatial relations between nodes
        self._infer_spatial_relations(new_node_ids)

        # 3. Prune low-confidence nodes
        self._prune_nodes()

        # 4. Apply confidence decay to old observations
        self._apply_confidence_decay()

    def _add_or_update_object_node(
        self,
        detection: Dict,
        depth: np.ndarray,
        camera_pose: np.ndarray
    ) -> Optional[int]:
        """
        Add a new object node or update existing one

        Args:
            detection: {label, box, score, mask}
            depth: Depth image
            camera_pose: Camera pose

        Returns:
            Node ID if created/updated, None otherwise
        """
        label = detection["label"]
        score = detection["score"]
        box = detection["box"]  # [x1, y1, x2, y2]

        # Estimate 3D position from depth
        position = self._estimate_3d_position(box, depth, camera_pose)

        if position is None:
            return None

        # Check if this object already exists (within spatial threshold)
        existing_node_id = self._find_nearby_node(position, label, threshold=0.5)

        if existing_node_id is not None:
            # Update existing node
            node = self.nodes[existing_node_id]
            node.confidence = min(1.0, node.confidence + 0.1 * score)
            node.position = 0.7 * node.position + 0.3 * position  # Smooth update
            return existing_node_id
        else:
            # Create new node
            node_id = self.next_node_id
            self.next_node_id += 1

            new_node = GraphNode(
                node_id=node_id,
                node_type=NodeType.OBJECT,
                position=position,
                label=label,
                confidence=score,
                attributes={},
                timestamp=self.current_step
            )

            self.nodes[node_id] = new_node
            self.graph.add_node(node_id, **new_node.__dict__)

            self.stats["total_nodes_created"] += 1
            return node_id

    def _estimate_3d_position(
        self,
        box: np.ndarray,
        depth: np.ndarray,
        camera_pose: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estimate 3D position of detected object

        Args:
            box: Bounding box [x1, y1, x2, y2]
            depth: Depth image
            camera_pose: 4x4 transformation matrix

        Returns:
            3D position [x, y, z] in global frame, or None if invalid
        """
        # TODO: Implement proper depth extraction and transformation
        # For now, use center of box
        x1, y1, x2, y2 = box.astype(int)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Get depth at center
        if 0 <= center_y < depth.shape[0] and 0 <= center_x < depth.shape[1]:
            d = depth[center_y, center_x]

            if d > 0.1 and d < 10.0:  # Valid depth range
                # Simple projection (needs proper camera intrinsics)
                # This is a placeholder
                position_camera = np.array([center_x / 100.0, center_y / 100.0, d])
                position_world = camera_pose[:3, :3] @ position_camera + camera_pose[:3, 3]
                return position_world

        return None

    def _find_nearby_node(
        self,
        position: np.ndarray,
        label: str,
        threshold: float = 0.5
    ) -> Optional[int]:
        """
        Find existing node near this position with same label

        Args:
            position: 3D position
            label: Object label
            threshold: Distance threshold (meters)

        Returns:
            Node ID if found, None otherwise
        """
        for node_id, node in self.nodes.items():
            if node.label == label and node.node_type == NodeType.OBJECT:
                dist = np.linalg.norm(node.position - position)
                if dist < threshold:
                    return node_id
        return None

    def _infer_spatial_relations(self, node_ids: List[int]) -> None:
        """
        Infer spatial relations between nodes

        Args:
            node_ids: List of node IDs to process
        """
        for node_id in node_ids:
            node = self.nodes[node_id]

            # Find nearby nodes
            for other_id, other_node in self.nodes.items():
                if other_id == node_id:
                    continue

                dist = np.linalg.norm(node.position - other_node.position)

                # Add "near" relation if within radius
                if dist < self.spatial_radius:
                    self._add_edge(
                        node_id,
                        other_id,
                        RelationType.NEAR,
                        confidence=1.0 - dist / self.spatial_radius
                    )

    def _add_edge(
        self,
        source_id: int,
        target_id: int,
        relation: RelationType,
        confidence: float
    ) -> None:
        """Add or update an edge in the scene graph"""
        edge_key = (source_id, target_id)

        if edge_key in self.edges:
            # Update confidence
            self.edges[edge_key].confidence = max(
                self.edges[edge_key].confidence,
                confidence
            )
        else:
            # Create new edge
            edge = GraphEdge(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                confidence=confidence,
                timestamp=self.current_step
            )
            self.edges[edge_key] = edge
            self.graph.add_edge(
                source_id,
                target_id,
                relation=relation.value,
                confidence=confidence
            )
            self.stats["total_edges_created"] += 1

    def _apply_confidence_decay(self) -> None:
        """Apply confidence decay to nodes not recently observed"""
        for node in self.nodes.values():
            if node.timestamp < self.current_step:
                node.confidence *= self.confidence_decay

    def _prune_nodes(self) -> None:
        """Remove nodes with low confidence"""
        to_remove = []

        for node_id, node in self.nodes.items():
            if node.confidence < self.min_confidence:
                to_remove.append(node_id)

        for node_id in to_remove:
            del self.nodes[node_id]
            self.graph.remove_node(node_id)

            # Remove associated edges
            edges_to_remove = [
                (s, t) for (s, t) in self.edges.keys()
                if s == node_id or t == node_id
            ]
            for edge_key in edges_to_remove:
                del self.edges[edge_key]

            self.stats["nodes_pruned"] += 1

    def get_semantic_context(
        self,
        position: np.ndarray,
        radius: float = 2.0
    ) -> str:
        """
        Get semantic context around a position

        Args:
            position: Query position [x, y, z]
            radius: Search radius (meters)

        Returns:
            Text description of nearby objects
        """
        nearby_objects = []

        for node in self.nodes.values():
            if node.node_type == NodeType.OBJECT:
                dist = np.linalg.norm(node.position[:2] - position[:2])  # 2D distance
                if dist < radius:
                    nearby_objects.append((node.label, dist, node.confidence))

        if not nearby_objects:
            return "unexplored area"

        # Sort by distance
        nearby_objects.sort(key=lambda x: x[1])

        # Format as text
        context_parts = []
        for label, dist, conf in nearby_objects[:3]:  # Top 3
            context_parts.append(f"{label} ({dist:.1f}m, conf={conf:.2f})")

        return "near " + ", ".join(context_parts)

    def to_text(self) -> str:
        """
        Convert scene graph to text representation for LLM

        Returns:
            Text description of scene graph
        """
        text = "Scene Graph:\n"
        text += f"Nodes ({len(self.nodes)}):\n"

        for node in self.nodes.values():
            text += f"  - {node.label} at ({node.position[0]:.1f}, {node.position[1]:.1f}), "
            text += f"confidence={node.confidence:.2f}\n"

        text += f"\nSpatial Relations ({len(self.edges)}):\n"
        for edge in list(self.edges.values())[:10]:  # Limit to avoid too long
            source_label = self.nodes[edge.source_id].label
            target_label = self.nodes[edge.target_id].label
            text += f"  - {source_label} {edge.relation.value} {target_label}\n"

        return text

    def visualize(self, save_path: str) -> None:
        """
        Visualize scene graph

        Args:
            save_path: Path to save visualization
        """
        # TODO: Implement visualization using networkx + matplotlib
        pass

    def get_statistics(self) -> Dict:
        """Get statistics about the scene graph"""
        return {
            **self.stats,
            "current_nodes": len(self.nodes),
            "current_edges": len(self.edges)
        }
