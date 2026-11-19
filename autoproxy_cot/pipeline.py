"""AutoProxy-CoT End-to-End Orchestrator Pipeline.

Ties together all 4 phases of the AutoProxy-CoT framework:
1. Overlapping Chunking & Coordinate Tagging
2. Answer-Masked Evidence Extraction
3. Schema-Driven Structuring (Entity-Relation Table)
4. Sufficiency Verification Gate
5. Grounded SFT Formatting & Noise Curriculum
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
from .chunker import DocumentChunker, Chunk
from .extractor import AnswerMaskedExtractor, EvidenceSpan
from .structurer import SchemaStructurer, StructuredProxy
from .verifier import SufficiencyVerifier, VerificationResult
from .grounded_trainer import (
    GroundedTraceFormatter,
    CurriculumNoiseScheduler,
    GroundedTrainingExample,
)
from .metrics import Evaluator, EvaluationMetrics


@dataclass
class PipelineOutput:
    """End-to-end outcome for a single document-query instance."""
    chunks: List[Chunk]
    raw_spans: List[EvidenceSpan]
    structured_proxy: StructuredProxy
    verification: VerificationResult
    is_admitted: bool
    grounded_example: Optional[GroundedTrainingExample] = None


class AutoProxyCoTPipeline:
    """Main pipeline class implementing the complete AutoProxy-CoT methodology."""

    def __init__(
        self,
        chunk_size: int = 2048,
        stride: int = 1792,
        max_proxy_tokens: int = 2048,
        extractor_llm: Optional[Callable[[str], str]] = None,
        verifier_llm: Optional[Callable[[str, str], str]] = None,
        sufficiency_threshold: float = 0.5,
        random_seed: int = 42,
    ):
        self.chunker = DocumentChunker(chunk_size=chunk_size, stride=stride)
        self.extractor = AnswerMaskedExtractor(llm_callable=extractor_llm)
        self.structurer = SchemaStructurer(max_token_budget=max_proxy_tokens)
        self.verifier = SufficiencyVerifier(
            verifier_callable=verifier_llm,
            match_threshold=sufficiency_threshold,
        )
        self.noise_scheduler = CurriculumNoiseScheduler(seed=random_seed)
        self.evaluator = Evaluator()

    def process(
        self,
        document_text: str,
        query: str,
        gold_answer: str,
        reasoning_steps: Optional[List[str]] = None,
        example_id: str = "example_01",
    ) -> PipelineOutput:
        """
        Executes the full AutoProxy-CoT pipeline on an input (C, q, a*):
        Step 1: Chunk document with overlap and coordinate tags
        Step 2: Answer-masked extraction over chunks
        Step 3: Structure union of spans into entity-relation schema
        Step 4: Sufficiency check gate(P*, q) == a*
        Step 5: Format grounded target if admitted
        """
        # Step 1: Chunking
        chunks = self.chunker.chunk_document(document_text)

        # Step 2: Chunk-level Extraction (Answer-Masked: gold_answer is NOT passed)
        all_spans: List[EvidenceSpan] = []
        for ch in chunks:
            spans = self.extractor.extract_from_chunk(ch, query)
            all_spans.extend(spans)

        # Step 3: Schema Structuring
        structured_proxy = self.structurer.structure(all_spans)

        # Step 4: Sufficiency Verification Gate
        verification = self.verifier.verify(structured_proxy, query, gold_answer)
        is_admitted = verification.is_accepted

        # Step 5: Grounded Trace Formatting (Only for admitted proxies)
        grounded_example = None
        if is_admitted:
            cited_coords = [s.coordinate for s in structured_proxy.spans]
            steps = reasoning_steps or [
                f"Identify premises from {', '.join(cited_coords[:2])}",
                f"Connect attributes to deduce target: {gold_answer}",
            ]
            formatted_trace = GroundedTraceFormatter.format_trace(
                cited_coordinates=cited_coords,
                reasoning_steps=steps,
                final_answer=gold_answer,
            )

            grounded_example = GroundedTrainingExample(
                example_id=example_id,
                query=query,
                context=structured_proxy.render_context(),
                stage="Stage1_ProxyRLVR",
                curriculum_level=0.0,
                target_grounding=", ".join(cited_coords),
                target_thought="\n".join(steps),
                target_answer=gold_answer,
                formatted_target=formatted_trace,
                cited_coordinates=cited_coords,
            )

        return PipelineOutput(
            chunks=chunks,
            raw_spans=all_spans,
            structured_proxy=structured_proxy,
            verification=verification,
            is_admitted=is_admitted,
            grounded_example=grounded_example,
        )

    def generate_curriculum_batch(
        self,
        output: PipelineOutput,
        alpha_levels: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0],
    ) -> List[GroundedTrainingExample]:
        """Generates multi-stage curriculum examples from pure proxy to full context."""
        if not output.is_admitted or not output.grounded_example:
            return []

        curriculum_examples = []
        base_ex = output.grounded_example

        for alpha in alpha_levels:
            curr_context = self.noise_scheduler.generate_curriculum_context(
                proxy=output.structured_proxy,
                full_chunks=output.chunks,
                alpha=alpha,
            )
            stage_name = "Stage1_ProxyRLVR" if alpha == 0.0 else f"Stage2_Noise_{int(alpha*100)}"
            curriculum_examples.append(
                GroundedTrainingExample(
                    example_id=f"{base_ex.example_id}_alpha_{alpha:.2f}",
                    query=base_ex.query,
                    context=curr_context,
                    stage=stage_name,
                    curriculum_level=alpha,
                    target_grounding=base_ex.target_grounding,
                    target_thought=base_ex.target_thought,
                    target_answer=base_ex.target_answer,
                    formatted_target=base_ex.formatted_target,
                    cited_coordinates=base_ex.cited_coordinates,
                )
            )

        return curriculum_examples
