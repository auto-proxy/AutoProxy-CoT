"""Evaluation Metrics and Diagnostic Engine.

Implements all metrics specified in AutoProxy-CoT research report:
1. Exact Match (EM) accuracy
2. Token-level F1 score
3. Evidence Recall: |P cap S*| / |S*|
4. Evidence Precision: |P cap S*| / |P|
5. Citation Faithfulness: Precision of cited span coordinates in <grounding>
6. Robustness to Evidence Position: Lost-in-the-Middle accuracy across normalized depths [0.0, 1.0]
"""

import re
import string
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional


@dataclass
class EvaluationMetrics:
    """Aggregated benchmark scores."""
    exact_match: float
    token_f1: float
    evidence_recall: float
    evidence_precision: float
    citation_faithfulness: float
    mid_context_drop: float
    depth_accuracies: Dict[str, float] = field(default_factory=dict)
    total_samples: int = 0


class Evaluator:
    """Calculates research metrics for multi-hop QA, grounding, and positional robustness."""

    @staticmethod
    def normalize_answer(text: str) -> str:
        """Standard HotpotQA normalization."""
        def remove_articles(t):
            return re.sub(r'\b(a|an|the)\b', ' ', t)

        def white_space_fix(t):
            return ' '.join(t.split())

        def remove_punc(t):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in t if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(text.lower())))

    @classmethod
    def compute_em(cls, prediction: str, ground_truth: str) -> float:
        return float(cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth))

    @classmethod
    def compute_f1(cls, prediction: str, ground_truth: str) -> float:
        pred_tokens = cls.normalize_answer(prediction).split()
        gold_tokens = cls.normalize_answer(ground_truth).split()
        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)
        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def extract_tags(text: str) -> Tuple[List[str], str, str]:
        """Extracts cited coordinates, thought, and answer from a generated or formatted trace."""
        grounding_match = re.search(r'<grounding>(.*?)</grounding>', text, re.DOTALL)
        thought_match = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

        raw_grounding = grounding_match.group(1).strip() if grounding_match else ""
        raw_thought = thought_match.group(1).strip() if thought_match else ""
        raw_answer = answer_match.group(1).strip() if answer_match else text.strip()

        # Extract coordinates like [c1_s04] or c1_s04
        coords = re.findall(r'c\d+_s\d+', raw_grounding)
        return coords, raw_thought, raw_answer

    @classmethod
    def compute_evidence_metrics(
        cls, proxy_coordinates: Set[str], gold_supporting_coordinates: Set[str]
    ) -> Tuple[float, float]:
        """Computes Evidence Recall and Evidence Precision."""
        if not gold_supporting_coordinates:
            return 1.0, 1.0 if not proxy_coordinates else 0.0
        if not proxy_coordinates:
            return 0.0, 0.0

        intersection = proxy_coordinates.intersection(gold_supporting_coordinates)
        recall = len(intersection) / len(gold_supporting_coordinates)
        precision = len(intersection) / len(proxy_coordinates)
        return recall, precision

    @classmethod
    def compute_citation_faithfulness(
        cls, cited_coordinates: List[str], gold_supporting_coordinates: Set[str]
    ) -> float:
        """Precision of cited evidence coordinates relative to true gold supporting facts."""
        if not cited_coordinates:
            return 0.0
        cited_set = set(cited_coordinates)
        correct_citations = cited_set.intersection(gold_supporting_coordinates)
        return len(correct_citations) / len(cited_set)

    @classmethod
    def evaluate_benchmark(
        cls,
        samples: List[Dict[str, Any]],
    ) -> EvaluationMetrics:
        """
        Evaluates a batch of test predictions.
        Each sample dict expects:
            - 'predicted_output': raw string or trace
            - 'gold_answer': reference answer string
            - 'proxy_coordinates': set/list of coordinates retained in proxy
            - 'gold_coordinates': set/list of true supporting fact coordinates
            - 'evidence_depth': float between 0.0 (start) and 1.0 (end)
        """
        total = len(samples)
        if total == 0:
            return EvaluationMetrics(0, 0, 0, 0, 0, 0, {}, 0)

        em_total = 0.0
        f1_total = 0.0
        rec_total = 0.0
        prec_total = 0.0
        faith_total = 0.0
        faith_count = 0

        # Depth buckets: Start (0.0), 0.25, Middle (0.50), 0.75, End (1.0)
        depth_buckets: Dict[str, List[float]] = {
            "start": [],
            "0.25": [],
            "middle": [],
            "0.75": [],
            "end": [],
        }

        for s in samples:
            raw_out = s.get("predicted_output", "")
            gold_ans = s.get("gold_answer", "")
            proxy_coords = set(s.get("proxy_coordinates", []))
            gold_coords = set(s.get("gold_coordinates", []))
            depth = float(s.get("evidence_depth", 0.5))

            cited_coords, thought, pred_ans = cls.extract_tags(raw_out)
            if not pred_ans:
                pred_ans = raw_out

            em = cls.compute_em(pred_ans, gold_ans)
            f1 = cls.compute_f1(pred_ans, gold_ans)
            em_total += em
            f1_total += f1

            rec, prec = cls.compute_evidence_metrics(proxy_coords, gold_coords)
            rec_total += rec
            prec_total += prec

            if cited_coords:
                faith = cls.compute_citation_faithfulness(cited_coords, gold_coords)
                faith_total += faith
                faith_count += 1

            # Depth categorization
            if depth <= 0.125:
                depth_buckets["start"].append(em)
            elif depth <= 0.375:
                depth_buckets["0.25"].append(em)
            elif depth <= 0.625:
                depth_buckets["middle"].append(em)
            elif depth <= 0.875:
                depth_buckets["0.75"].append(em)
            else:
                depth_buckets["end"].append(em)

        avg_em = (em_total / total) * 100
        avg_f1 = (f1_total / total) * 100
        avg_rec = (rec_total / total) * 100
        avg_prec = (prec_total / total) * 100
        avg_faith = (faith_total / max(1, faith_count)) * 100

        depth_accs = {}
        for k, v in depth_buckets.items():
            depth_accs[k] = (sum(v) / len(v) * 100) if v else avg_em

        start_acc = depth_accs.get("start", avg_em)
        mid_acc = depth_accs.get("middle", avg_em)
        mid_drop = mid_acc - start_acc  # Drop from start to middle

        return EvaluationMetrics(
            exact_match=round(avg_em, 1),
            token_f1=round(avg_f1, 1),
            evidence_recall=round(avg_rec, 1),
            evidence_precision=round(avg_prec, 1),
            citation_faithfulness=round(avg_faith, 1),
            mid_context_drop=round(mid_drop, 1),
            depth_accuracies={k: round(v, 1) for k, v in depth_accs.items()},
            total_samples=total,
        )
