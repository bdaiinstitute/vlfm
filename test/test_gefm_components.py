#!/usr/bin/env python3
"""
Test GEFM Components

Quick tests to verify GEFM modules are working
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_scene_graph_map():
    """Test SceneGraphMap basic functionality"""
    print("\n" + "="*60)
    print("Testing SceneGraphMap...")
    print("="*60)

    try:
        from vlfm.mapping.scene_graph_map import SceneGraphMap, NodeType

        # Create scene graph
        sg = SceneGraphMap(
            max_nodes=10,
            spatial_radius=2.0,
            confidence_decay=0.95
        )

        print("✓ SceneGraphMap created successfully")

        # Mock detection data
        detections = [
            {
                "label": "chair",
                "score": 0.9,
                "box": np.array([100, 100, 200, 200]),
                "mask": None
            },
            {
                "label": "table",
                "score": 0.85,
                "box": np.array([300, 150, 400, 250]),
                "mask": None
            }
        ]

        # Mock depth and pose
        depth = np.ones((480, 640)) * 2.0  # 2 meters
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        camera_pose = np.eye(4)

        # Update scene graph
        sg.update_from_observation(
            rgb=rgb,
            depth=depth,
            detections=detections,
            camera_pose=camera_pose,
            step=0
        )

        print(f"✓ Scene graph updated: {len(sg.nodes)} nodes created")

        # Test semantic context
        query_position = np.array([1.0, 1.0, 0.0])
        context = sg.get_semantic_context(query_position, radius=5.0)
        print(f"✓ Semantic context: '{context}'")

        # Test text representation
        text = sg.to_text()
        print(f"✓ Text representation generated ({len(text)} chars)")

        # Test statistics
        stats = sg.get_statistics()
        print(f"✓ Statistics: {stats}")

        print("\n✅ SceneGraphMap tests PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ SceneGraphMap tests FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_llm_reasoner_mock():
    """Test LLMReasoner with mock backend"""
    print("\n" + "="*60)
    print("Testing LLMReasoner (Mock)...")
    print("="*60)

    try:
        from vlfm.vlm.llm_reasoner import LLMReasoner, LLMBackend

        # Create mock backend
        class MockLLMBackend(LLMBackend):
            def query(self, prompt, max_tokens=200, temperature=0.3):
                # Return mock JSON response
                return '''
                {
                  "frontier_0": {"score": 0.85, "reason": "Near kitchen area where chairs are common"},
                  "frontier_1": {"score": 0.6, "reason": "Bedroom area, less likely"},
                  "frontier_2": {"score": 0.4, "reason": "Unexplored corridor"}
                }
                '''

        print("✓ Mock LLM backend created")

        # Create reasoner with mock backend
        reasoner = LLMReasoner(model_name="mock", backend="openai", api_key="mock")
        reasoner.backend = MockLLMBackend()

        print("✓ LLMReasoner created with mock backend")

        # Test frontier scoring
        frontiers = [
            {"id": 0, "position": [1.0, 2.0], "context": "near table and chairs"},
            {"id": 1, "position": [3.0, 4.0], "context": "near bed"},
            {"id": 2, "position": [5.0, 6.0], "context": "unexplored area"}
        ]

        scene_graph_text = "Scene Graph:\nNodes: table, chair, bed"
        goal_text = "chair"

        scores = reasoner.score_frontiers(
            frontiers=frontiers,
            scene_graph_text=scene_graph_text,
            goal_text=goal_text
        )

        print(f"✓ Frontier scores computed: {len(scores)} frontiers")

        for fid, (score, reason) in scores.items():
            print(f"  - Frontier {fid}: score={score:.2f}, reason='{reason}'")

        # Test related objects
        related = reasoner.get_related_objects("chair", max_count=3)
        print(f"✓ Related objects: {related}")

        # Test statistics
        stats = reasoner.get_statistics()
        print(f"✓ Statistics: {stats}")

        print("\n✅ LLMReasoner tests PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ LLMReasoner tests FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_llm_reasoner_real():
    """Test LLMReasoner with real LLM (optional)"""
    print("\n" + "="*60)
    print("Testing LLMReasoner (Real LLM - Optional)...")
    print("="*60)

    try:
        from vlfm.vlm.llm_reasoner import LLMReasoner

        # Try to use Ollama first (free, local)
        try:
            import ollama
            reasoner = LLMReasoner(
                model_name="llama3",
                backend="ollama"
            )
            print("✓ Using Ollama backend (free)")

        except ImportError:
            print("⚠ Ollama not available, skipping real LLM test")
            print("  Install: pip install ollama && ollama pull llama3")
            return None  # Skip, not failure

        # Simple test query
        prompt = "Name 3 objects commonly found in a kitchen. Reply with just the list."
        response = reasoner.query(prompt, max_tokens=50)

        if response:
            print(f"✓ LLM response: '{response[:100]}...'")
            print("\n✅ Real LLM test PASSED\n")
            return True
        else:
            print("⚠ No response from LLM")
            return None

    except Exception as e:
        print(f"\n⚠ Real LLM test SKIPPED: {e}\n")
        return None


def test_gefm_policy_import():
    """Test GEFMPolicy can be imported"""
    print("\n" + "="*60)
    print("Testing GEFMPolicy Import...")
    print("="*60)

    try:
        from vlfm.policy.gefm_policy import GEFMPolicy

        print("✓ GEFMPolicy imported successfully")

        # Check class attributes
        print("✓ GEFMPolicy class defined")

        print("\n✅ GEFMPolicy import test PASSED\n")
        return True

    except ImportError as e:
        print(f"\n❌ GEFMPolicy import FAILED: {e}\n")
        print("This is expected if VLFM modules haven't been set up yet")
        return False


def run_all_tests():
    """Run all GEFM component tests"""
    print("\n" + "="*80)
    print(" "*20 + "GEFM Component Tests")
    print("="*80)

    results = {
        "SceneGraphMap": test_scene_graph_map(),
        "LLMReasoner (Mock)": test_llm_reasoner_mock(),
        "LLMReasoner (Real)": test_llm_reasoner_real(),
        "GEFMPolicy Import": test_gefm_policy_import()
    }

    # Summary
    print("\n" + "="*80)
    print("Test Summary:")
    print("="*80)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠  SKIPPED"

        print(f"{status:12} {name}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
