r"""Grounded Training Trace Formatter and Curriculum Noise Scheduler.

Implements Step 4 of AutoProxy-CoT:
- Grounded Target Format:
  <grounding>[c_k_s_j]</grounding> <thought>...</thought> <answer>...</answer>
- Curriculum Noise Schedule:
  C_k = P* U N_k, where |N_k| = alpha_k * |C \ P*|
  alpha_k in {0.0, 0.25, 0.50, 0.75, 1.0}
- Grounded SFT Loss Formulation:
  L_Grounded(theta) = - Sum_{t=1}^{|tau|} log pi_theta( tau_t | C, q, tau_{<t} )
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .structurer import StructuredProxy
from .extractor import EvidenceSpan
from .chunker import Chunk


@dataclass
class GroundedTrainingExample:
    """A formatted training sample ready for Stage 1 RLVR or Stage 2 Grounded SFT."""
    example_id: str
    query: str
    context: str                  # P* (Stage 1) or C_k / Full C (Stage 2)
    stage: str                    # "Stage1_ProxyRLVR" or "Stage2_GroundedSFT"
    curriculum_level: float       # alpha_k in [0.0, 1.0]
    target_grounding: str         # e.g., "[c1_s04], [c2_s02]"
    target_thought: str           # Step-by-step chain of thought
    target_answer: str            # Final answer
    formatted_target: str         # Full trace string <grounding>...</grounding> <thought>...</thought> <answer>...</answer>
    cited_coordinates: List[str]


class GroundedTraceFormatter:
    """Formats training targets requiring explicit evidence citations before reasoning."""

    @staticmethod
    def format_trace(
        cited_coordinates: List[str],
        reasoning_steps: List[str],
        final_answer: str,
    ) -> str:
        """Constructs target string: <grounding>[coords]</grounding> <thought>...</thought> <answer>...</answer>"""
        coords_str = ", ".join(f"[{c}]" if not c.startswith("[") else c for c in cited_coordinates)
        grounding_tag = f"<grounding>{coords_str}</grounding>"
        
        thought_body = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(reasoning_steps))
        thought_tag = f"<thought>\n{thought_body}\n</thought>"
        
        answer_tag = f"<answer>{final_answer}</answer>"
        
        return f"{grounding_tag} {thought_tag} {answer_tag}"


class CurriculumNoiseScheduler:
    """Generates noise-augmented contexts P* U N_k across stages: 0.0 -> 0.25 -> 0.50 -> 0.75 -> 1.0 (Full C)."""

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def generate_curriculum_context(
        self,
        proxy: StructuredProxy,
        full_chunks: List[Chunk],
        alpha: float = 0.0,
    ) -> str:
        """
        Creates context C_k = P* U N_k.
        alpha = 0.0: Pure compact proxy P* (Stage 1 RLVR)
        alpha in (0, 1): Proxy interspersed with noise distractors (Intermediate Curriculum)
        alpha = 1.0: Full Document Context C (Stage 2 Final SFT)
        """
        if alpha <= 0.0:
            return proxy.render_context()

        if alpha >= 1.0:
            # Full context with coordinate annotations
            all_text_chunks = []
            for ch in full_chunks:
                chunk_header = f"\n--- Section Chunk {ch.chunk_index} ---\n"
                chunk_body = " ".join(f"[{s.coordinate}] {s.text}" for s in ch.sentences)
                all_text_chunks.append(chunk_header + chunk_body)
            return "".join(all_text_chunks)

        # Interspersed noise: select distractor sentences from full chunks that are not in proxy
        proxy_coords = {s.coordinate for s in proxy.spans}
        distractor_sentences = []
        for ch in full_chunks:
            for s in ch.sentences:
                if s.coordinate not in proxy_coords:
                    distractor_sentences.append(s)

        num_distractors_to_add = int(len(distractor_sentences) * alpha)
        selected_distractors = self.random.sample(
            distractor_sentences, min(num_distractors_to_add, len(distractor_sentences))
        )

        # Merge proxy spans and noise distractors, sorted by global order
        combined_items = []
        for s in proxy.spans:
            combined_items.append((s.chunk_index, s.sentence_index, f"[{s.coordinate}] {s.verbatim_text}"))
        for d in selected_distractors:
            combined_items.append((d.chunk_index, d.sentence_index, f"[{d.coordinate}] {d.text}"))

        combined_items.sort(key=lambda x: (x[0], x[1]))
        interspersed_text = " ".join(item[2] for item in combined_items)

        header = f"### Curriculum Context (Noise Ratio alpha = {alpha:.2f})\n\n"
        return header + interspersed_text
