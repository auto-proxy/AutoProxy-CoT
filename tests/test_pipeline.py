"""Unit and integration tests for AutoProxy-CoT modules."""

import unittest
from autoproxy_cot.chunker import DocumentChunker, Chunk, SentenceSpan
from autoproxy_cot.extractor import AnswerMaskedExtractor, EvidenceSpan
from autoproxy_cot.structurer import SchemaStructurer
from autoproxy_cot.verifier import SufficiencyVerifier
from autoproxy_cot.grounded_trainer import GroundedTraceFormatter, CurriculumNoiseScheduler
from autoproxy_cot.metrics import Evaluator
from autoproxy_cot.pipeline import AutoProxyCoTPipeline


class TestAutoProxyCoT(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "Arthur's Magazine was founded in 1844 in Philadelphia. "
            "It was edited by Timothy Shay Arthur. "
            "Timothy Shay Arthur later wrote the novel Ten Nights in a Bar-Room. "
            "The novel became a major 19th-century success."
        )
        self.query = "What novel was written by the editor of Arthur's Magazine?"
        self.gold_answer = "Ten Nights in a Bar-Room"

    def test_chunking_and_coordinate_tagging(self):
        chunker = DocumentChunker(chunk_size=30, stride=20)
        chunks = chunker.chunk_document(self.sample_text)
        self.assertGreater(len(chunks), 0)
        first_sentence = chunks[0].sentences[0]
        self.assertEqual(first_sentence.coordinate, "c1_s01")
        self.assertTrue(first_sentence.text.startswith("Arthur's Magazine"))

    def test_answer_masked_extraction(self):
        chunker = DocumentChunker(chunk_size=100, stride=80)
        chunks = chunker.chunk_document(self.sample_text)
        extractor = AnswerMaskedExtractor()
        
        # Verify that extraction works without gold answer in prompt
        spans = extractor.extract_from_chunk(chunks[0], self.query)
        self.assertGreater(len(spans), 0)
        for s in spans:
            self.assertTrue(s.verbatim_text in self.sample_text)

    def test_schema_structuring(self):
        spans = [
            EvidenceSpan("c1_s01", "Arthur's Magazine was edited by T. S. Arthur", "Arthur's Magazine", "edited by", 1, 1),
            EvidenceSpan("c1_s02", "T. S. Arthur wrote Ten Nights in a Bar-Room", "T. S. Arthur", "wrote", 1, 2),
        ]
        structurer = SchemaStructurer(max_token_budget=500)
        proxy = structurer.structure(spans)
        self.assertIn("| Coordinate |", proxy.markdown_table)
        self.assertIn("[c1_s01]", proxy.markdown_table)
        self.assertEqual(len(proxy.key_value_records), 2)

    def test_sufficiency_verifier_gate(self):
        verifier = SufficiencyVerifier()
        # Mock structured proxy with valid records
        structurer = SchemaStructurer(max_token_budget=500)
        spans = [
            EvidenceSpan("c1_s01", "It was edited by Timothy Shay Arthur.", "Arthur's Magazine", "edited by", 1, 1),
            EvidenceSpan("c1_s02", "He became famous for his temperance novel Ten Nights in a Bar-Room.", "Timothy Shay Arthur", "famous for", 1, 2),
        ]
        proxy = structurer.structure(spans)
        result = verifier.verify(proxy, self.query, self.gold_answer)
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.predicted_answer, "Ten Nights in a Bar-Room")

    def test_grounded_trace_formatting(self):
        coords = ["c1_s01", "c1_s02"]
        steps = ["Identify editor as Timothy Shay Arthur", "Identify novel as Ten Nights in a Bar-Room"]
        trace = GroundedTraceFormatter.format_trace(coords, steps, self.gold_answer)
        self.assertIn("<grounding>[c1_s01], [c1_s02]</grounding>", trace)
        self.assertIn("<thought>", trace)
        self.assertIn(f"<answer>{self.gold_answer}</answer>", trace)

    def test_metrics_evaluation(self):
        sample = {
            "predicted_output": f"<grounding>[c1_s01]</grounding> <thought>...</thought> <answer>{self.gold_answer}</answer>",
            "gold_answer": self.gold_answer,
            "proxy_coordinates": ["c1_s01"],
            "gold_coordinates": ["c1_s01"],
            "evidence_depth": 0.5,
        }
        metrics = Evaluator.evaluate_benchmark([sample])
        self.assertEqual(metrics.exact_match, 100.0)
        self.assertEqual(metrics.citation_faithfulness, 100.0)


if __name__ == "__main__":
    unittest.main()
