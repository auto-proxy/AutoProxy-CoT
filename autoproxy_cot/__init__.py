"""AutoProxy-CoT: Automated Proxy Synthesis and Evidence-Grounded Chain-of-Thought Tuning.
Reference Implementation and Formal Pipeline Model.
"""

from .chunker import DocumentChunker, Chunk, SentenceSpan
from .extractor import AnswerMaskedExtractor, EvidenceSpan
from .structurer import SchemaStructurer, StructuredProxy
from .verifier import SufficiencyVerifier, VerificationResult
from .grounded_trainer import (
    GroundedTraceFormatter,
    CurriculumNoiseScheduler,
    GroundedTrainingExample,
)
from .metrics import Evaluator, EvaluationMetrics
from .pipeline import AutoProxyCoTPipeline

__all__ = [
    "DocumentChunker",
    "Chunk",
    "SentenceSpan",
    "AnswerMaskedExtractor",
    "EvidenceSpan",
    "SchemaStructurer",
    "StructuredProxy",
    "SufficiencyVerifier",
    "VerificationResult",
    "GroundedTraceFormatter",
    "CurriculumNoiseScheduler",
    "GroundedTrainingExample",
    "Evaluator",
    "EvaluationMetrics",
    "AutoProxyCoTPipeline",
]
