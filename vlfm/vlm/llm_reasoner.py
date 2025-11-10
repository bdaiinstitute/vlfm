# Copyright [2024] Your Institution
"""
LLM Reasoner for GEFM
Provides spatial reasoning capabilities using Large Language Models
"""

from typing import Dict, List, Optional, Tuple
import re
import json
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""

    @abstractmethod
    def query(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        """
        Query the LLM

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend (GPT-3.5, GPT-4, etc.)"""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def query(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for robot navigation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""


class OllamaBackend(LLMBackend):
    """Local LLM backend using Ollama"""

    def __init__(self, model_name: str = "llama3"):
        try:
            import ollama
            self.client = ollama
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install ollama-python: pip install ollama")

    def query(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend"""

    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    def query(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return ""


class LLMReasoner:
    """
    LLM-based spatial reasoner for frontier selection

    Provides:
    1. Frontier scoring based on spatial reasoning
    2. Object relation inference
    3. Explanation generation
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        backend: str = "openai",
        api_key: Optional[str] = None,
        enable_cache: bool = True
    ):
        """
        Args:
            model_name: Name of the LLM model
            backend: Backend type ("openai", "ollama", "anthropic")
            api_key: API key (for cloud backends)
            enable_cache: Whether to cache responses
        """
        self.model_name = model_name
        self.backend = self._create_backend(backend, model_name, api_key)
        self.enable_cache = enable_cache
        self.cache: Dict[str, str] = {}

        # Statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "failed_queries": 0
        }

    def _create_backend(
        self,
        backend: str,
        model_name: str,
        api_key: Optional[str]
    ) -> LLMBackend:
        """Create appropriate LLM backend"""
        if backend == "openai":
            return OpenAIBackend(model_name, api_key)
        elif backend == "ollama":
            return OllamaBackend(model_name)
        elif backend == "anthropic":
            return AnthropicBackend(model_name, api_key)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def query(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.3
    ) -> str:
        """
        Query the LLM

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        self.stats["total_queries"] += 1

        # Check cache
        if self.enable_cache and prompt in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[prompt]

        # Query LLM
        response = self.backend.query(prompt, max_tokens, temperature)

        if response:
            # Cache response
            if self.enable_cache:
                self.cache[prompt] = response
            return response
        else:
            self.stats["failed_queries"] += 1
            return ""

    def score_frontiers(
        self,
        frontiers: List[Dict],
        scene_graph_text: str,
        goal_text: str
    ) -> Dict[int, Tuple[float, str]]:
        """
        Score frontiers using LLM reasoning

        Args:
            frontiers: List of frontiers with {id, position, context}
            scene_graph_text: Text representation of scene graph
            goal_text: Target object description

        Returns:
            Dict mapping frontier_id -> (score, reasoning)
        """
        # Construct prompt
        prompt = self._build_frontier_scoring_prompt(
            frontiers,
            scene_graph_text,
            goal_text
        )

        # Query LLM
        response = self.query(prompt, max_tokens=300, temperature=0.3)

        # Parse response
        scores = self._parse_frontier_scores(response, frontiers)

        return scores

    def _build_frontier_scoring_prompt(
        self,
        frontiers: List[Dict],
        scene_graph_text: str,
        goal_text: str
    ) -> str:
        """Build prompt for frontier scoring"""
        prompt = f"""You are a robot navigation assistant. Your task is to score frontier locations based on how likely they are to lead to the target object.

Goal: Find a {goal_text}

Current Scene Graph:
{scene_graph_text}

Frontier Candidates:
"""

        for frontier in frontiers:
            fid = frontier["id"]
            context = frontier.get("context", "unknown")
            pos = frontier.get("position", [0, 0])
            prompt += f"\nFrontier {fid}:\n"
            prompt += f"  - Position: ({pos[0]:.1f}, {pos[1]:.1f})\n"
            prompt += f"  - Nearby objects: {context}\n"

        prompt += f"""
Task: For each frontier, provide:
1. A confidence score (0.0-1.0) indicating likelihood of finding {goal_text}
2. Brief reasoning (1 sentence)

Think about:
- Which rooms typically contain {goal_text}?
- What objects are usually near {goal_text}?
- Which frontier is in the most relevant spatial context?

Output format (JSON):
{{
  "frontier_1": {{"score": 0.85, "reason": "Near kitchen area where {goal_text} are common"}},
  "frontier_2": {{"score": 0.3, "reason": "Bedroom area, unlikely to have {goal_text}"}},
  ...
}}

Your response (JSON only):
"""

        return prompt

    def _parse_frontier_scores(
        self,
        response: str,
        frontiers: List[Dict]
    ) -> Dict[int, Tuple[float, str]]:
        """
        Parse LLM response to extract frontier scores

        Args:
            response: LLM response text
            frontiers: Original frontier list

        Returns:
            Dict mapping frontier_id -> (score, reasoning)
        """
        scores = {}

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                for frontier in frontiers:
                    fid = frontier["id"]
                    key = f"frontier_{fid}"

                    if key in data:
                        score = float(data[key].get("score", 0.5))
                        reason = data[key].get("reason", "No reason provided")
                        scores[fid] = (score, reason)
                    else:
                        # Default score if not found
                        scores[fid] = (0.5, "LLM did not score this frontier")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response: {response}")

            # Fallback: assign default scores
            for frontier in frontiers:
                scores[frontier["id"]] = (0.5, "Parsing failed")

        return scores

    def get_related_objects(
        self,
        object_name: str,
        max_count: int = 5
    ) -> List[Tuple[str, str]]:
        """
        Get objects commonly found near the target object

        Args:
            object_name: Target object
            max_count: Maximum number of related objects

        Returns:
            List of (object_name, relation_type) tuples
        """
        prompt = f"""List {max_count} objects that are commonly found near a {object_name}.

For each object, specify the spatial relation.

Example format:
- table: typically_near
- lamp: on_top_of
- rug: underneath

Your response (list only):
"""

        response = self.query(prompt, max_tokens=150, temperature=0.5)

        # Parse response
        related = []
        for line in response.strip().split('\n'):
            match = re.match(r'-\s*(\w+):\s*(\w+)', line.strip())
            if match:
                obj, relation = match.groups()
                related.append((obj, relation))

        return related[:max_count]

    def explain_reasoning(
        self,
        frontier_id: int,
        score: float,
        context: str,
        goal: str
    ) -> str:
        """
        Generate human-readable explanation for frontier selection

        Args:
            frontier_id: Selected frontier ID
            score: Frontier score
            context: Semantic context
            goal: Target object

        Returns:
            Explanation text
        """
        prompt = f"""Explain why frontier {frontier_id} was selected for finding {goal}.

Context: {context}
Score: {score:.2f}

Provide a 1-2 sentence explanation suitable for a user interface.
"""

        explanation = self.query(prompt, max_tokens=100, temperature=0.7)
        return explanation.strip()

    def clear_cache(self) -> None:
        """Clear the response cache"""
        self.cache.clear()

    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        return {
            **self.stats,
            "cache_size": len(self.cache)
        }
