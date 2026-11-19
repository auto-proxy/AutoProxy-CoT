# AutoProxy-CoT

Automated Proxy Synthesis and Evidence-Grounded Chain-of-Thought Tuning for Long-Context Reasoning in Compact Language Models.

---

## Overview

AutoProxy-CoT introduces an automated four-stage pipeline to enable effective multi-step reasoning over long contexts (32K–128K+ tokens) in compact open-weight models (7B–9B parameters):

1. **Answer-Masked Chunk Extraction:** Extracts verbatim factual spans across overlapping document windows without exposing target answers.
2. **Schema-Driven Structuring:** Compiles extracted facts into compact entity-attribute and relation tables.
3. **Sufficiency Verification Gate:** Validates candidate proxies via closed-loop checking before admitting them to training.
4. **Grounded Policy Optimization:** Fine-tunes reasoning policies using explicit source citation tags `<grounding>[c_k_s_j]</grounding>` under a noise curriculum schedule.

---

## Repository Structure

```text
AutoProxy-CoT/
├── autoproxy_cot/          # Core Python framework package
│   ├── chunker.py          # Sliding-window chunking & coordinate tagging
│   ├── extractor.py        # Answer-masked evidence span extractor
│   ├── structurer.py       # Entity-relation table compiler
│   ├── verifier.py         # Sufficiency verification gate
│   ├── grounded_trainer.py # Citation formatter & noise curriculum scheduler
│   ├── metrics.py          # QA, evidence, and lost-in-the-middle evaluation
│   └── pipeline.py         # End-to-end pipeline orchestrator
├── docs/                   # Research proposal & technical report documentation
├── examples/               # Standalone execution examples
├── scripts/                # Utility scripts for document compiling
├── tests/                  # Automated unit and integration tests
├── pyproject.toml          # Packaging and dependency configuration
└── requirements.txt        # Python package dependencies
```

---

## Installation & Setup

```bash
git clone https://github.com/danwin-byte/AutoProxy-CoT.git
cd AutoProxy-CoT
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart

Run the demonstration pipeline:

```bash
python examples/run_demo.py
```

Run test suite:

```bash
python -m unittest discover -s tests -v
```
