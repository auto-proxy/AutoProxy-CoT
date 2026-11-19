# AutoProxy-CoT: Automated Proxy Synthesis and Evidence-Grounded Chain-of-Thought Tuning for Long-Context Reasoning in Compact Language Models

**Authors:** Bashik Rajan, Danwin Sajith, Adith Sajan  
**Faculty Supervisor:** Prof. Divyanshu Prakash  
**Affiliation:** Department of Artificial Intelligence and Machine Learning (AIML), JAIN University, Bengaluru, India  
**Type:** Technical Research Report  

---

## Abstract
Compact language models can perform multi-step reasoning, but their accuracy often falls when the supporting information is spread across very long documents. Processing these documents directly is also expensive, and models may overlook evidence placed in the middle of a context. Proxy-based Chain-of-Thought (ProxyCoT) addresses this by first training on shorter, information-rich versions of documents and then transferring that reasoning ability to the full context. However, the original approach depends on manually prepared proxies and does not explicitly teach the model where its evidence appears in the full document.

This report presents **AutoProxy-CoT**, a pipeline designed to automate proxy creation and make later reasoning more closely tied to source evidence. It extracts information from overlapping document chunks without showing the target answer to the extractor, organizes the extracted information into a structured proxy, and checks whether that proxy is sufficient before using it for training. During the second training stage, the model is taught to identify the source sentences that support its reasoning. The reported evaluation uses 200 held-out questions from HotpotQA and SciTrek with Qwen2.5-7B-Instruct and Gemma-2-9B-It. The results show improved answer accuracy, stronger evidence alignment, better performance when evidence appears in the middle of a long context, and lower training compute than direct long-context reinforcement learning.

---

## 1. Introduction and Background

### 1.1 The problem
Answering a question about a long document can require connecting facts that appear far apart. This is common in multi-hop question answering and in scientific literature review. For compact open-weight models with 7B–9B parameters, directly training on contexts of 32K to 128K or more tokens creates two main difficulties:
* **Compute and memory costs:** Long inputs increase attention computation, KV-cache use, and the time needed to process a prompt.
* **Evidence position:** Models can be less reliable when important information appears in the middle of a long context. This is commonly called the "lost-in-the-middle" effect.

### 1.2 ProxyCoT and its limitations
ProxyCoT is based on the Reasoning Invariance Hypothesis: the key reasoning steps needed to answer a question should remain the same when irrelevant passages are removed. It therefore trains a reasoning policy on short proxy contexts (about 1K–2K tokens) and then transfers the learned reasoning to full documents through supervised fine-tuning.

The approach has two practical limitations:
1. Strong results depend on proxies curated by people; replacing them with standard retrieval methods causes a large accuracy drop.
2. The later fine-tuning stage does not explicitly connect each reasoning step to the location of its supporting evidence in the full document.

### 1.3 Research objectives
* **Automate proxy creation:** Extract useful evidence from document chunks without exposing the target answer during extraction.
* **Check proxy quality:** Verify that a candidate proxy contains enough information to answer the question before using it for training.
* **Ground reasoning in evidence:** Train the model to cite source sentence locations during the full-context fine-tuning stage.
* **Evaluate the approach:** Measure answer quality, evidence quality, robustness to evidence position, and computational cost.

---

## 2. Literature Review and Research Gap

| Reference | Main idea | How it relates to this work |
| :--- | :--- | :--- |
| **Li et al. (2026) [1]** | ProxyCoT trains on short proxy contexts and transfers reasoning to long documents. | Provides the main baseline. AutoProxy-CoT focuses on automating proxy creation and adding evidence citations. |
| **Yang et al. (2018) [2]** | Introduced HotpotQA, a multi-hop question-answering dataset with supporting-fact annotations. | Provides sentence-level evidence labels for evaluation. |
| **Trivedi et al. (2022) [3]** | Introduced MuSiQue, designed to test compositional multi-hop reasoning. | Highlights the need to preserve dependencies across reasoning steps, not just matching keywords. |
| **Liu et al. (2024) [4]** | Studied how evidence position affects language-model performance in long contexts. | Motivates explicit evidence grounding and tests for mid-context performance. |
| **Wang et al. (2023) [5]** | Introduced Self-Consistency, which combines multiple reasoning paths. | Offers a test-time reasoning baseline for comparison. |

### 2.1 Research gap
ProxyCoT shows that reasoning learned from compact contexts can transfer to longer documents, but its reliance on manually prepared proxies makes large-scale use difficult. Standard lexical or dense retrieval can miss bridge facts that only become relevant after an earlier reasoning step. In addition, the original transfer stage does not directly supervise the model to locate and cite evidence in the full document. AutoProxy-CoT addresses these issues through answer-masked chunk extraction, structured proxy creation, a sufficiency check, and evidence-grounded fine-tuning.

---

## 3. Datasets and Evaluation Protocol

### 3.1 Datasets
* **HotpotQA:** The distractor and full-wiki configurations. Each example contains a multi-hop question, supporting documents, distractor paragraphs, sentence-level supporting-fact indices, and a reference answer.
* **SciTrek:** A scientific question-answering benchmark based on full-text research articles. It includes long contexts of up to 128K tokens and requires the model to connect information about methods and empirical findings.

### 3.2 Preprocessing and evaluation
Documents are divided into overlapping windows of 2,048 tokens, using a stride of 1,792 tokens (overlap = 256 tokens). Each sentence is assigned a coordinate tag that identifies its chunk and sentence position (`[c_k_s_j]`).

Evaluation is carried out on a fixed, non-repeating subset of 200 questions sampled from the validation splits of HotpotQA and SciTrek, using random seed 42. Sampling settings are temperature 0.7 and top-p 0.8; deterministic greedy decoding uses temperature 0.0.

---

## 4. Proposed Model: AutoProxy-CoT

| Step | What happens | Purpose |
| :--- | :--- | :--- |
| **1. Chunk extraction** | Split full document into overlapping 2,048-token chunks. An LLM extracts relevant evidence from each chunk using the question, but not the gold answer. | Avoid answer leakage and retain useful evidence across the document. |
| **2. Structured proxy** | Combine extracted spans and organize them into an entity–attribute and relation table, with a target size of no more than 2,048 tokens. | Create a compact representation that preserves links between facts. |
| **3. Sufficiency check** | A verifier tries to answer the question using only the candidate proxy. The gold answer is used afterward as an acceptance check. | Reject incomplete or corrupted proxies before training. |
| **4. Grounded training** | Train on verified proxies first, then fine-tune with increasingly noisy contexts and finally the full document. The target includes source sentence coordinates. | Transfer reasoning while teaching the model to point to supporting evidence. |

### 4.1 Answer-masked chunk extraction
The full document is represented as overlapping chunks $c_1$ through $c_m$. Each chunk is processed independently with query $q$. The gold answer is not included in the extractor prompt ($a^* \notin \text{Prompt}$).

### 4.2 Structuring and verification
Extracted spans are combined into an entity-attribute table. A lightweight verifier attempts to answer $q$ using only candidate proxy $P^*$. Candidate is admitted if and only if $\mathcal{M}_{\text{ver}}(P^*, q) == a^*$.

### 4.3 Evidence-grounded fine-tuning
* **Stage 1:** Reasoning policy optimized on verified compact proxies $P^*$.
* **Stage 2:** Policy fine-tuned using curriculum noise schedule moving from $P^*$ to noisy mixture to full document $C$.
* **Grounding target format:**
  `<grounding>[c_k_s_j]</grounding> <thought>...</thought> <answer>...</answer>`

---

## 5. Experimental Results

### 5.1 Main benchmark results (200-question evaluation subset)

| Model and method | HotpotQA EM | HotpotQA F1 | SciTrek EM | SciTrek F1 | Evidence recall | Evidence precision | Citation faithfulness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen: Zero-shot direct | 41.5 | 52.8 | 28.0 | 36.4 | — | — | — |
| Qwen: Standard long-context SFT | 64.0 | 73.2 | 43.5 | 51.9 | — | — | — |
| Qwen: Heuristic ProxyCoT | 35.1 | 44.6 | 24.5 | 31.8 | 41.2 | 38.5 | — |
| **Qwen: AutoProxy-CoT** | **86.5** | **91.4** | **61.0** | **68.7** | **88.4** | **84.2** | **93.5** |
| Qwen: Human ProxyCoT upper bound | 92.0 | 95.1 | 65.5 | 72.4 | 94.0 | 91.5 | — |
| Gemma: Zero-shot direct | 44.0 | 55.1 | 30.5 | 38.9 | — | — | — |
| Gemma: Standard long-context SFT | 66.5 | 75.4 | 46.0 | 54.2 | — | — | — |
| Gemma: Heuristic ProxyCoT | 36.8 | 46.2 | 26.0 | 33.4 | 43.0 | 39.8 | — |
| **Gemma: AutoProxy-CoT** | **88.0** | **92.6** | **63.5** | **70.8** | **89.8** | **85.6** | **94.8** |
| Gemma: Human ProxyCoT upper bound | 92.5 | 95.6 | 67.0 | 73.9 | 94.5 | 92.0 | — |

### 5.2 Ablation study

| Configuration | HotpotQA EM | Change from full model | Evidence precision | Citation faithfulness |
| :--- | :--- | :--- | :--- | :--- |
| **Full AutoProxy-CoT** | **86.5** | — | **84.2** | **93.5** |
| Raw text proxy; no schema structuring | 81.0 | −5.5 | 79.4 | 90.1 |
| No sufficiency gate | 74.5 | −12.0 | 71.8 | 86.4 |
| No citation grounding | 79.5 | −7.0 | — | — |

### 5.3 Robustness to evidence position (64K–128K tokens)

| Method | Start (0.0) | 0.25 | Middle (0.50) | 0.75 | End (1.0) | Mid-context drop |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard long-context SFT | 68.0 | 48.5 | 45.0 | 49.0 | 67.0 | **−23.0** |
| No citation grounding | 88.0 | 74.0 | 72.5 | 75.0 | 87.5 | **−15.5** |
| **AutoProxy-CoT** | **89.0** | **86.0** | **85.5** | **86.5** | **88.5** | **−3.5** |

### 5.4 Training compute and inference time

| Training and inference setup | GPU-hours | Prefill at 64K (ms) | Prefill at 128K (ms) | Avg. generation (s) |
| :--- | :--- | :--- | :--- | :--- |
| Direct long-context RLVR | 142.0 | 1,840 | 4,620 | 3.45 |
| Standard long-context SFT | 48.5 | 1,840 | 4,620 | 3.10 |
| **AutoProxy-CoT** | **34.0** | **1,855** | **4,650** | **2.65** |

---

## 6. References
1. X. Li, A. Saparina, C. Gurung, and M. Lapata, "Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning," arXiv preprint arXiv:2605.20201v2, 2026.
2. Z. Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering," EMNLP, 2018.
3. H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, "MuSiQue: Multihop Questions via Single-hop Question Inversion," TACL, 2022.
4. N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," TACL, 2024.
5. X. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models," ICLR, 2023.
