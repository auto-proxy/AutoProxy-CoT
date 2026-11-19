"""Schema-Driven Structurer.

Implements Step 2 of AutoProxy-CoT:
P* = Structure(Union_{i=1}^m S_i)

Key Purpose:
- Transforms disparate extracted text spans into a compact, indexable schema (entity-attribute & relation table).
- Directly leverages the empirical finding that structured representations provide a +66.9 point boost
  over raw unstructured prose (Li et al., 2026; AutoProxy-CoT report Section 4.2).
- Enforces a strict budget constraint (typically <= 2,048 tokens).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from .extractor import EvidenceSpan


@dataclass
class StructuredProxy:
    """The synthesized structured proxy representation P*."""
    markdown_table: str
    key_value_records: List[Dict[str, str]]
    spans: List[EvidenceSpan]
    token_count: int
    is_truncated: bool = False

    def render_context(self) -> str:
        """Returns the full prompt-ready structured proxy text."""
        header = "### Synthesized Evidence Proxy (Structured Schema)\n\n"
        return header + self.markdown_table


class SchemaStructurer:
    """Compiles extracted evidence spans into structured tables and entity maps."""

    def __init__(self, max_token_budget: int = 2048):
        self.max_token_budget = max_token_budget

    def structure(self, spans: List[EvidenceSpan]) -> StructuredProxy:
        """Combines and organizes the union of extracted spans into a structured schema."""
        # 1. Deduplicate spans by coordinate and verbatim text
        seen_coords: Set[str] = set()
        seen_texts: Set[str] = set()
        unique_spans: List[EvidenceSpan] = []

        for span in spans:
            clean_text = span.verbatim_text.strip().lower()
            if span.coordinate in seen_coords or clean_text in seen_texts:
                continue
            seen_coords.add(span.coordinate)
            seen_texts.add(clean_text)
            unique_spans.append(span)

        # 2. Sort spans by source chunk and sentence order to preserve document logical progression
        unique_spans.sort(key=lambda s: (s.chunk_index, s.sentence_index))

        # 3. Compile into Markdown Table and Key-Value records
        records: List[Dict[str, str]] = []
        table_lines: List[str] = [
            "| Coordinate | Entity / Subject | Relationship / Attribute | Verbatim Anchor |",
            "| :--- | :--- | :--- | :--- |",
        ]

        current_tokens = 30  # Table header approximate tokens
        admitted_spans: List[EvidenceSpan] = []
        is_truncated = False

        for s in unique_spans:
            # Clean text for table cell display
            safe_text = s.verbatim_text.replace("|", "-").strip()
            safe_entity = s.entity.replace("|", "-").strip()
            safe_relation = s.relation.replace("|", "-").strip()

            line = f"| [{s.coordinate}] | {safe_entity} | {safe_relation} | \"{safe_text}\" |"
            line_tokens = len(line.split())

            if current_tokens + line_tokens > self.max_token_budget:
                is_truncated = True
                break

            table_lines.append(line)
            current_tokens += line_tokens
            admitted_spans.append(s)
            records.append({
                "coordinate": s.coordinate,
                "entity": safe_entity,
                "relation": safe_relation,
                "anchor": safe_text,
            })

        table_markdown = "\n".join(table_lines)

        return StructuredProxy(
            markdown_table=table_markdown,
            key_value_records=records,
            spans=admitted_spans,
            token_count=current_tokens,
            is_truncated=is_truncated,
        )
