"""Document Chunker with Boundary Overlap and Coordinate Tagging.

According to AutoProxy-CoT specification:
- Documents are split into overlapping windows of L_c = 2,048 tokens.
- Window stride S = 1,792 tokens (overlap delta = 256 tokens).
- Each sentence within each chunk is assigned an explicit coordinate tag: [c_k_s_j].
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class SentenceSpan:
    """Represents a single sentence with chunk and sentence-level coordinates."""
    coordinate: str       # e.g. "c1_s01"
    chunk_index: int      # 1-based index of chunk
    sentence_index: int   # 1-based index within chunk
    text: str             # Verbatim sentence text
    start_char: int       # Character offset in original document
    end_char: int         # End character offset in original document
    global_index: int     # Global sentence index across entire document


@dataclass
class Chunk:
    """An overlapping document chunk with tagged sentences."""
    chunk_index: int
    text: str
    sentences: List[SentenceSpan] = field(default_factory=list)
    start_token: int = 0
    end_token: int = 0

    @property
    def token_length(self) -> int:
        return len(self.text.split())

    def get_sentence_by_coordinate(self, coord: str) -> Optional[SentenceSpan]:
        for s in self.sentences:
            if s.coordinate == coord:
                return s
        return None


class DocumentChunker:
    """Partitions a long document C into overlapping chunks with coordinate tagging."""

    def __init__(self, chunk_size: int = 2048, stride: int = 1792):
        self.chunk_size = chunk_size
        self.stride = stride
        self.overlap = chunk_size - stride
        # Sentence splitting pattern (supporting punctuation and newlines)
        self.sentence_regex = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])|\n\n+')

    def split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences while tracking exact character offsets."""
        sentences = []
        raw_parts = self.sentence_regex.split(text)
        current_pos = 0

        for part in raw_parts:
            part_str = part.strip()
            if not part_str:
                continue
            idx = text.find(part_str, current_pos)
            if idx == -1:
                idx = current_pos
            start = idx
            end = start + len(part_str)
            sentences.append((part_str, start, end))
            current_pos = end

        return sentences

    def chunk_document(self, document_text: str) -> List[Chunk]:
        """Divides long context C into overlapping chunks c_1 through c_m with coordinate tags."""
        raw_sentences = self.split_sentences(document_text)
        if not raw_sentences:
            return []

        # Convert sentences to token counts (approximate token count via whitespace/words)
        sent_tokens = []
        for text, start, end in raw_sentences:
            tokens = text.split()
            sent_tokens.append((len(tokens), text, start, end))

        chunks: List[Chunk] = []
        global_sent_count = len(raw_sentences)
        
        chunk_start_idx = 0
        chunk_num = 1

        while chunk_start_idx < global_sent_count:
            accumulated_tokens = 0
            chunk_sent_spans: List[SentenceSpan] = []
            chunk_texts: List[str] = []
            curr_idx = chunk_start_idx
            
            while curr_idx < global_sent_count:
                num_tok, s_text, s_start, s_end = sent_tokens[curr_idx]
                if accumulated_tokens + num_tok > self.chunk_size and chunk_sent_spans:
                    break
                
                coord = f"c{chunk_num}_s{len(chunk_sent_spans) + 1:02d}"
                span = SentenceSpan(
                    coordinate=coord,
                    chunk_index=chunk_num,
                    sentence_index=len(chunk_sent_spans) + 1,
                    text=s_text,
                    start_char=s_start,
                    end_char=s_end,
                    global_index=curr_idx + 1,
                )
                chunk_sent_spans.append(span)
                chunk_texts.append(s_text)
                accumulated_tokens += num_tok
                curr_idx += 1

            chunk_body = " ".join(chunk_texts)
            chunk = Chunk(
                chunk_index=chunk_num,
                text=chunk_body,
                sentences=chunk_sent_spans,
                start_token=chunk_start_idx,
                end_token=curr_idx,
            )
            chunks.append(chunk)

            if curr_idx >= global_sent_count:
                break

            # Calculate next chunk_start_idx based on stride tokens
            stride_tokens = 0
            next_start = chunk_start_idx
            while next_start < curr_idx:
                stride_tokens += sent_tokens[next_start][0]
                next_start += 1
                if stride_tokens >= self.stride:
                    break
            
            # Ensure progress
            if next_start <= chunk_start_idx:
                next_start = chunk_start_idx + 1

            chunk_start_idx = next_start
            chunk_num += 1

        return chunks
