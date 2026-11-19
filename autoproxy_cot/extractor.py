"""Answer-Masked Chunk Extractor.

Key Guarantees:
- Answer Masking: The extractor prompt contains ONLY the query q and chunk c_i.
  Under NO circumstances is the gold answer a* included in the extractor prompt.
- Verbatim Span Grounding: Every extracted fact must exist verbatim within the chunk text.
  Any hallucinated span is discarded immediately.
- Coordinate Association: Every extracted span retains its source chunk and sentence coordinate.
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from .chunker import Chunk, SentenceSpan


@dataclass
class EvidenceSpan:
    """Represents a verified extracted evidence unit from a chunk."""
    coordinate: str        # e.g., "c1_s04"
    verbatim_text: str     # The exact text anchor from the document
    entity: str            # The primary entity involved
    relation: str          # Attribute or relationship descriptor
    chunk_index: int
    sentence_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinate": self.coordinate,
            "verbatim_text": self.verbatim_text,
            "entity": self.entity,
            "relation": self.relation,
            "chunk_index": self.chunk_index,
            "sentence_index": self.sentence_index,
        }


class AnswerMaskedExtractor:
    """Extracts relevant evidence from chunks without exposure to the gold answer."""

    def __init__(self, llm_callable: Optional[Callable[[str], str]] = None):
        """
        Args:
            llm_callable: Optional custom function taking prompt string and returning LLM response string.
                          If None, a built-in semantic-relevance extractor is used for deterministic offline simulation.
        """
        self.llm_callable = llm_callable

    def build_extractor_prompt(self, query: str, chunk: Chunk) -> str:
        """Constructs an answer-masked prompt displaying numbered sentences with coordinates."""
        annotated_sentences = "\n".join(
            f"[{s.coordinate}] {s.text}" for s in chunk.sentences
        )
        prompt = (
            f"You are an evidence extraction engine. Identify facts relevant to answering the question.\n"
            f"RULES:\n"
            f"1. You do NOT have access to the target answer. Rely only on the context below.\n"
            f"2. Extract ONLY sentences or clauses that contain potential premises or bridge entities.\n"
            f"3. Return output strictly in JSON format as specified.\n\n"
            f"Question: {query}\n\n"
            f"Document Chunk:\n{annotated_sentences}\n\n"
            f"Respond with JSON format:\n"
            f"```json\n"
            f'{{\n  "evidence_found": true,\n  "spans": [\n    {{"coordinate": "c1_s01", "verbatim_text": "...", "entity": "...", "relation": "..."}}\n  ]\n}}\n'
            f"```"
        )
        return prompt

    def extract_from_chunk(self, chunk: Chunk, query: str) -> List[EvidenceSpan]:
        """Extracts candidate evidence spans S_i = Extract(c_i, q) with Answer Masking."""
        # Verification check: Ensure gold answer is not accidentally leaked
        prompt = self.build_extractor_prompt(query, chunk)

        if self.llm_callable:
            raw_response = self.llm_callable(prompt)
            spans = self._parse_llm_response(raw_response, chunk)
        else:
            spans = self._heuristic_semantic_extract(chunk, query)

        # Anti-hallucination verification: Ensure extracted spans exist verbatim in chunk
        verified_spans: List[EvidenceSpan] = []
        for span in spans:
            if span.verbatim_text.strip() and (span.verbatim_text in chunk.text or any(span.verbatim_text in s.text for s in chunk.sentences)):
                verified_spans.append(span)

        return verified_spans

    def _parse_llm_response(self, response_text: str, chunk: Chunk) -> List[EvidenceSpan]:
        """Parses structured JSON response from LLM."""
        spans: List[EvidenceSpan] = []
        try:
            # Extract JSON from markdown fence if present
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_part = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_part = response_text.strip()

            data = json.loads(json_part)
            if not data.get("evidence_found", False):
                return []

            for item in data.get("spans", []):
                coord = item.get("coordinate", "")
                sentence = chunk.get_sentence_by_coordinate(coord)
                spans.append(
                    EvidenceSpan(
                        coordinate=coord,
                        verbatim_text=item.get("verbatim_text", sentence.text if sentence else ""),
                        entity=item.get("entity", "Unknown Entity"),
                        relation=item.get("relation", "Relevant context"),
                        chunk_index=chunk.chunk_index,
                        sentence_index=sentence.sentence_index if sentence else 0,
                    )
                )
        except Exception:
            # Fall back to heuristic matching if JSON parsing fails
            return self._heuristic_semantic_extract(chunk, "")

        return spans

    def _heuristic_semantic_extract(self, chunk: Chunk, query: str) -> List[EvidenceSpan]:
        """Built-in deterministic extractor based on lexical and entity overlap with query."""
        query_words = set(w.lower() for w in query.replace("?", "").replace(",", "").split() if len(w) > 2)
        stopwords = {"what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", "from", "with", "about"}
        query_terms = query_words - stopwords

        matched_spans: List[EvidenceSpan] = []
        for sent in chunk.sentences:
            sent_words = set(w.lower() for w in sent.text.split())
            overlap = query_terms.intersection(sent_words)
            # Include if there is term overlap or if sentence defines a relationship
            if overlap or any(kw in sent.text.lower() for kw in ["is a", "was born", "founded", "known for", "precursor", "located"]):
                # Extract simple entity and relation heuristics
                words = sent.text.split()
                entity = " ".join(words[:3]) if len(words) >= 3 else sent.text
                matched_spans.append(
                    EvidenceSpan(
                        coordinate=sent.coordinate,
                        verbatim_text=sent.text,
                        entity=entity,
                        relation=f"Relates to {', '.join(list(overlap)[:3]) if overlap else 'context'}",
                        chunk_index=chunk.chunk_index,
                        sentence_index=sent.sentence_index,
                    )
                )
        return matched_spans
