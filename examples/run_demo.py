"""AutoProxy-CoT Demo and Usage Example."""

import sys
import os

# Ensure package root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from autoproxy_cot.pipeline import AutoProxyCoTPipeline
from autoproxy_cot.metrics import Evaluator


def main():
    print("=" * 80)
    print(" Running AutoProxy-CoT Example Pipeline")
    print("=" * 80)

    document = (
        "Arthur's Magazine was an American literary magazine published in Philadelphia in the 19th century. "
        "The publication was founded in 1844 and catered to family reading circles. "
        "It was edited by Timothy Shay Arthur, who also contributed numerous moral essays. "
        "Philadelphia was a major hub of publishing and commercial printing during the antebellum period. "
        "Timothy Shay Arthur was a prolific American author and temperance advocate. "
        "He became famous throughout the United States for his temperance novel Ten Nights in a Bar-Room and What I Saw There. "
        "The novel was a major commercial sensation and was later adapted into several successful stage plays."
    )
    query = "What was the famous novel written by the editor of Arthur's Magazine?"
    gold_answer = "Ten Nights in a Bar-Room"

    pipeline = AutoProxyCoTPipeline(
        chunk_size=150,
        stride=100,
        max_proxy_tokens=500,
        sufficiency_threshold=0.6,
    )

    output = pipeline.process(
        document_text=document,
        query=query,
        gold_answer=gold_answer,
        example_id="demo_001",
    )

    print(f"\nQuery: {query}")
    print(f"Chunks: {len(output.chunks)}")
    print(f"Extracted Spans: {len(output.raw_spans)}")
    print(f"Proxy Admitted: {output.is_admitted}")
    print(f"Verification Match Score: {output.verification.match_score:.2f}")

    if output.grounded_example:
        print("\nGenerated Grounded Trace Target:")
        print(output.grounded_example.formatted_target)


if __name__ == "__main__":
    main()
