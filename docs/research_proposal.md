# Research Proposal

## AutoProxy-CoT: Automated Proxy Synthesis and Evidence-Grounded Chain-of-Thought Tuning for Long-Context Reasoning

---

**Author:** [Ashik Rajan / Candidate Name]  
**Target Program / Course:** Advanced Natural Language Processing / Research Project  
**Affiliation:** Department of Computer Science & Artificial Intelligence  
**Date:** November 2026  
**Primary Reference:** Li, Saparina, Gurung, & Lapata (2026). *Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning* (arXiv:2605.20201v2).

---

### Executive Summary

Large language models (LLMs) frequently falter when performing multi-step reasoning over long textual inputs (32K–128K+ tokens), succumbing to attention dilution, quadratic computational bottlenecks, and the well-documented "lost-in-the-middle" phenomenon. Recent work on *Proxy-based Chain-of-Thought Tuning* (ProxyCoT; Li et al., 2026) introduced an appealing thesis: multi-step reasoning chains are largely invariant to irrelevant context, meaning policy optimization (RLVR) can occur on condensed "proxy contexts" rather than massive full documents. 

However, ProxyCoT suffers from a crippling bottleneck: **its empirical success hinges entirely on high-quality, human-annotated proxies.** When the authors attempted heuristic automation in their study, downstream performance collapsed precipitously—yielding a dismal 35.1% accuracy on HotpotQA using semantic retrieval compared to ~92% with human-annotated proxies, and falling to 3.4%–15.8% with random sentence selection. Moreover, ProxyCoT's Stage 2 Supervised Fine-Tuning (SFT) trains models exclusively on traces generated over short proxies, failing to teach the model how to locate, extract, and ground evidence within noisy, needle-in-a-haystack contexts.

This proposal presents **AutoProxy-CoT**, a scalable, automated framework that eliminates the human-in-the-loop requirement while reinforcing evidence grounding. We design a four-stage pipeline:
1. **Chunk-Level Span Extraction:** A cost-efficient LLM API parses full contexts in segmented chunks under strict zero-answer-leakage protocols to isolate verbatim factual spans.
2. **Schema-Driven Structuring:** Extracted spans are synthesized into structured key-value/tabular representations—directly exploiting the empirical finding that structured representations provide a +66.9 point boost over unstructured text.
3. **Answer-Free Sufficiency Verification:** A lightweight verifier checks whether the proxy contains adequate signal to answer the query before admitting it to the training curriculum.
4. **Grounded Policy Optimization:** Stage 2 SFT is redesigned with an explicit *cite-before-reason* mechanism, followed by curriculum reinforcement learning across graduated noise ratios ($P \to P \cup \text{Noise} \to C$).

We propose an exhaustive evaluation across HotpotQA, SciTrek, MuSiQue, 2WikiMQA, and LongBench v2, scaling sequence lengths up to 512K tokens. By rigorously isolating proxy quality, grounding fidelity, and end-to-end computational costs, this project aims to turn proxy-based tuning into an accessible, fully autonomous training paradigm for long-context reasoning.

---

### 1. Introduction and Background

Recent advancements in transformer architectures—such as RingAttention, FlashAttention-3, and rotary position embedding (RoPE) extrapolation—have expanded context windows from 8K to over 1M tokens. Yet, scaling the context window does not automatically confer long-context reasoning capabilities. In multi-hop question answering and scientific literature synthesis, models must isolate disparate factual premises dispersed across dozens of pages, ignore plausible distractors, and execute coherent deduction.

Under standard Supervised Fine-Tuning (SFT) or Reinforcement Learning with Verifiable Rewards (RLVR) on long sequences, practitioners confront two prohibitive hurdles:
1. **Computational Overhead:** Generating long chain-of-thought (CoT) rollouts across sequence lengths of $128\text{K}$ tokens is dominated by prefill and KV-cache memory pressure, making exploration in RL intractable.
2. **Attention Dispersion:** Models exposed to massive distraction struggle to associate relevant premises, leading to hallucinatory reasoning steps and position-dependent performance drops ("lost-in-the-middle").

To bypass these hurdles, Li et al. (2026) introduced **ProxyCoT**. Their method rests on the hypothesis of **reasoning invariance**: the logical deduction steps needed to resolve a query $q$ from a long context $C$ remain identical whether reasoning over $C$ or over an idealized, condensed proxy $P \subset C$. ProxyCoT trains in two stages:
- **Stage 1 (Proxy RLVR):** Reinforce reasoning policies on concise proxies $P$ where search space and generation length are constrained.
- **Stage 2 (Long-Context SFT):** Transfer the learned reasoning behavior to full contexts $C$ by fine-tuning the model on input pairs $(C, q)$ with target proxy-generated traces $\tau$.

```
Standard Long-Context RLVR:
  [Context C (128k)] + [Query q] ---> Huge Prefill & Memory ---> Long Exploratory Traces (Expensive)

ProxyCoT Paradigm:
  Stage 1: [Proxy P (1k)]   + [Query q] ---> Fast RLVR Exploration ---> Dense Reasoning Policy
  Stage 2: [Context C (128k)] + [Query q] ---> Standard SFT with Trace tau ---> Context Transfer
```

While conceptually elegant, ProxyCoT leaves critical open questions regarding real-world feasibility, data generation dependencies, and evidence grounding.

---

### 2. Critical Problem Statement and Motivation

A granular examination of the empirical results in Li et al. (2026) reveals three structural vulnerabilities:

#### 2.1. The Fatal Dependency on Human Annotations
ProxyCoT's performance is strictly bounded by human-constructed proxy quality. Table 9 of Li et al. demonstrates that replacing human proxies with off-the-shelf automated retrieval causes catastrophic degradation:
- **Human-Annotated Proxies:** ~92.0% accuracy on HotpotQA.
- **BM25 / Dense Semantic Retrieval:** **35.1%** accuracy.
- **Random Sentence Selection:** **3.4% – 15.8%** accuracy.

Standard retrieval mechanisms fail because multi-hop queries contain bridge entities and relational steps that semantic search cannot capture in isolation without iterative query reformulation. If proxy generation cannot be automated with high precision and recall, ProxyCoT remains an academic curiosity rather than a viable production pipeline.

#### 2.2. The Ungrounded Reasoning Gap in Stage 2
In ProxyCoT, Stage 2 SFT presents the model with the full context $C$ and query $q$, training it to predict trace $\tau$. Crucially, $\tau$ was generated by an agent looking **only** at proxy $P$. As a result, $\tau$ contains no references to where the evidence was located in $C$. The model is never rewarded for navigating $C$, identifying key sentences, or filtering out surrounding noise. When deployed on unseen, noisy documents, the policy frequently hallucinates facts that it "expects" to be present or falters when distractors share lexical overlap with target entities.

#### 2.3. The Untapped Value of Structured Representations
A fascinating, underexplored finding in Table 9 of Li et al. is that feeding models structured metadata (authors, tables, abstracts, key terms) yielded an accuracy of **91.5%**, compared to just **24.6%** when providing identical information in raw, unstructured prose. This 66.9-point delta indicates that the syntactic format of the proxy dramatically impacts reasoning policy stabilization. ProxyCoT treated this as an incidental observation; we propose treating it as a core architectural lever.

#### 2.4. Methodological and Computational Ambiguities
The original work asserts "reduced computational overhead," yet only measures inference token counts (617 vs. 1,744 tokens). It omits wall-clock execution times, memory peak allocations, and GPU-hours consumed during the Stage 2 prefill over 128K tokens. Furthermore, the baseline "SFT on $C$" scored 19.5% on SciTrek, falling well below the zero-shot baseline of 30.8%—a clear indicator of under-tuning or optimization collapse that demands rigorous re-benchmarking.

---

### 3. Research Questions and Hypotheses

This research proposal investigates four primary research questions:

- **RQ1 (Autonomous Proxy Synthesis):** Can an automated, chunk-level LLM extraction pipeline bridge the performance gap between heuristic retrieval (35.1%) and human-annotated proxies (~92%) without human intervention?
  - *Hypothesis 1:* Chunked extraction using high-throughput language models with strict JSON schema constraints can achieve $\ge 88\%$ evidence recall on multi-hop benchmarks, matching downstream human-proxy RLVR accuracy within 3 percentage points.
- **RQ2 (Structured Distillation Impact):** Does transforming extracted text fragments into structured key-value and tabular schemas systematically outperform raw textual concatenation during RLVR?
  - *Hypothesis 2:* Structured schema distillation reduces token entropy and prevents distractor interference, boosting reasoning stability by at least 15% across varied base model sizes.
- **RQ3 (Sufficiency Verification and Leakage Prevention):** How can automated proxies be validated without leaking ground-truth answers into the extraction prompt?
  - *Hypothesis 3:* An answer-free extraction step coupled with an auxiliary zero-shot verification filter (retaining $(q, P)$ pairs only if a lightweight verifier can solve $q$ from $P$) produces training data that equals or exceeds human proxy quality while strictly preventing leakage.
- **RQ4 (Explicit Grounding and Curriculum Noise Robustness):** Does introducing citation tokens into the reasoning trace and training under a noise curriculum resolve the "lost-in-the-middle" effect in ultra-long contexts (128K–512K)?
  - *Hypothesis 4:* Grounded traces with span pointers will maintain within 5% of peak accuracy regardless of whether the critical evidence resides in the initial 10%, middle 50%, or final 10% of the context window.

---

### 4. Proposed Methodology: AutoProxy-CoT

The proposed system comprises two primary subsystems: (A) the **Automated Proxy Synthesis and Verification Pipeline (APSVP)**, and (B) the **Grounded Multi-Stage Policy Optimizer (GMPO)**.

```
+---------------------------------------------------------------------------------------------------+
|                            PHASE A: AUTOMATED PROXY SYNTHESIS & VERIFICATION                      |
+---------------------------------------------------------------------------------------------------+
  Full Context C (128k+) 
         |
         v
  [1. Overlapping Chunking] ---> Chunks c_1, c_2, ..., c_m
         |
         +----------------------------+ (Query q ONLY, NO Answer)
         v                            v
  [2. LLM Span Extractor] <------- Prompt Caching & Batch API
         |
         v
    Raw Verbatim Spans {s_k} with Source IDs
         |
         v
  [3. Schema Structuring] ------> Key-Value Tables & Entity-Attribute Maps
         |
         v
  Synthesized Proxy Candidate P
         |
         v
  [4. Sufficiency Filter] -------> Verifier(P, q) == a* ?
         |                                  |
         | [NO: Discard / Re-chunk]        | [YES: Verified Proxy P*]
         x                                  v
+---------------------------------------------------------------------------------------------------+
|                            PHASE B: GROUNDED MULTI-STAGE TRAINING                                 |
+---------------------------------------------------------------------------------------------------+
  Stage 1: Proxy Policy RLVR
     Input: (P*, q)  ---> Fast Exploration ---> Reward: Format + Verifiable Correctness
     Output: Intermediate Policy pi_proxy
         |
         v
  Stage 2: Grounded SFT & Noise Curriculum
     Trace Format: <cite>[Span_IDs]</cite> <reasoning>...</reasoning> <answer>...</answer>
     Curriculum:   P*  --->  P* + 25% Noise  --->  P* + 75% Noise  --->  Full Context C (128k+)
     Output: Final Long-Context Grounded Policy pi_final
```

#### 4.1. Phase A: Automated Proxy Synthesis Pipeline

##### 4.1.1. Chunking and Verbatim Span Extraction
Given a long document $C$, we partition $C$ into overlapping chunks $\{c_1, c_2, \dots, c_m\}$ where each chunk spans $L_c = 2048$ tokens with an overlap $\delta = 256$ tokens to preserve boundary context.

We instruct a lightweight, fast extractor LLM (e.g., `Claude 3.5 Haiku` or a quantized local `Qwen2.5-7B-Instruct`) to identify evidentiary spans. 

**Anti-Leakage Guarantee:** The extractor receives **only** $(q, c_i)$. Under no circumstances is the gold target answer $a^*$ presented to the extractor. This prevents the extractor from reverse-engineering spurious shortcuts.

The prompt requires structured JSON output:
```json
{
  "evidence_found": true,
  "spans": [
    {
      "span_id": "c1_s01",
      "verbatim_text": "...",
      "entity": "...",
      "relation": "..."
    }
  ]
}
```

##### 4.1.2. Schema-Driven Structuring
Rather than concatenating extracted spans into an unorganized block of text, we apply a schema compiler. The compiler organizes the extracted facts into markdown tables and key-value dictionaries:
```markdown
### Extracted Evidence Table
| Entity | Attribute / Relationship | Reference Context ID | Verbatim Anchor |
| :--- | :--- | :--- | :--- |
| Entity A | Discovered in 1984 | [c2_s04] | "discovered by team A in 1984" |
| Entity B | Precursor to Entity A | [c5_s12] | "acted as primary precursor" |
```
This structured format reduces lexical noise and provides direct indexable anchors for the reasoning engine.

##### 4.1.3. Sufficiency Verification Filter
Not all automated extractions contain complete multi-hop chains. To filter out deficient proxies, candidate proxy $P$ is passed through a lightweight verifier model $\mathcal{M}_{\text{ver}}$:
$$\hat{a} = \mathcal{M}_{\text{ver}}(P, q)$$
The candidate proxy is admitted to the gold proxy set $\mathcal{D}_{\text{train}}$ if and only if:
$$\text{Score}(\hat{a}, a^*) \ge \tau_{\text{threshold}}$$
where $\text{Score}$ evaluates exact match or numerical equivalence. If $\mathcal{M}_{\text{ver}}$ fails, the sample is flagged for multi-pass extraction with larger chunk windows or discarded. This step guarantees that every generated proxy contains a verifiable reasoning pathway.

#### 4.2. Phase B: Grounded Multi-Stage Policy Optimization

##### 4.2.1. Stage 1: Proxy-Level RLVR
We initialize policy $\pi_\theta$ from a strong open base model (e.g., `Qwen2.5-7B` or `Gemma-2-9B`). We train $\pi_\theta$ on $(P^*, q)$ using Group Relative Policy Optimization (GRPO) or PPO. The reward function $R(\tau, a^*)$ is defined as:
$$R(\tau, a^*) = R_{\text{correctness}}(\text{ans}(\tau), a^*) + \lambda_1 R_{\text{format}}(\tau) + \lambda_2 R_{\text{conciseness}}(\tau)$$
Because $P^*$ is compact ($\le 1.5\text{K}$ tokens), the generation of rollouts $\tau$ requires minimal KV cache, and rollouts converge rapidly without wandering into irrelevant text.

##### 4.2.2. Stage 2: Grounded SFT with Explicit Citation
To bridge the grounding gap identified in ProxyCoT, we mandate that reasoning traces include citation pointers matching the original document spans. The target trace format is:
```xml
<grounding>
  Based on [c2_s04] and [c5_s12], we establish the chronological link.
</grounding>
<thought>
  Step 1: Entity B was synthesized first in 1981...
  Step 2: Connecting Entity B with the finding in [c2_s04]...
</thought>
<answer>
  Target Answer
</answer>
```
During Stage 2 fine-tuning on full context $(C, q)$, the model is optimized via cross-entropy loss:
$$\mathcal{L}_{\text{Grounded-SFT}}(\theta) = -\sum_{t=1}^{|\tau|} \log \pi_\theta(\tau_t \mid C, q, \tau_{<t})$$
Because the model must predict `<grounding>` tokens citing valid context IDs before generating `<thought>`, it learns attention routing into the full context $C$.

##### 4.2.3. Curriculum Noise Schedule
To prevent policy collapse when transitioning from clean proxies $P^*$ to 128K-token noisy contexts $C$, we introduce a curriculum noise schedule:
$$\mathcal{C}_k = P^* \cup \mathcal{N}_k, \quad \text{where } |\mathcal{N}_k| = \alpha_k \cdot |C \setminus P^*|, \quad \alpha_k \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$$
In each curriculum phase $k$, the model adapts to progressively higher noise ratios, making reasoning invariant to distractor density.

---

### 5. Experimental Setup and Evaluation Plan

#### 5.1. Datasets and Benchmarks

We will conduct experiments across diverse multi-hop and long-context benchmarks to ensure our findings generalize beyond templated tasks:

| Benchmark | Context Length | Task Type | Key Purpose |
| :--- | :--- | :--- | :--- |
| **HotpotQA (Distractor / Full)** | 4K – 32K | Multi-hop bridge & comparison | Evaluates evidence precision/recall against gold supporting facts. |
| **SciTrek** | 32K – 128K (Ext to 512K) | Multi-step scientific literature QA | Directly tests claims from Li et al. (2026) at scale. |
| **MuSiQue** | 8K – 32K | 2-hop to 4-hop multi-hop QA | Rigorous disconnected multi-hop reasoning without lexical shortcuts. |
| **2WikiMQA** | 8K – 32K | Multi-hop with explicit entity graphs | Evaluates structured relation tracking. |
| **LongBench v2 / RULER** | 32K – 128K+ | Synthetic & real long-context suites | Tests needle retrieval, tracking, and aggregation. |
| **Loong** | 32K – 128K | Academic document understanding | Evaluates out-of-domain transfer on research papers. |

#### 5.2. Baseline Comparisons

We benchmark AutoProxy-CoT against a comprehensive suite of competitive methods:

1. **Zero-Shot / Few-Shot Direct:** Prompting base models (`Qwen2.5-7B/14B`, `Gemma-2-9B`, `Llama-3.1-8B`) directly on $C$ with standard and CoT prompting.
2. **Standard Long-Context SFT:** Fine-tuning directly on $(C, q \to \tau)$ using full context.
3. **Retrieval-Augmented Generation (RAG):**
   - BM25 retrieval top-$k$ sentences.
   - Dense retrieval with `BGE-M3` and `Contriever-MSMARCO`.
   - Iterative / Multi-hop RAG (e.g., Self-RAG or Tree-of-Thoughts retrieval).
4. **Original ProxyCoT (Human Upper Bound):** ProxyCoT trained with gold human-annotated proxies (Li et al., 2026).
5. **Original ProxyCoT (Heuristic Lower Bound):** ProxyCoT trained with BM25 / embedding-retrieved proxies (the 35.1% baseline).

#### 5.3. Ablation Matrix

To isolate the contribution of each proposed component, we construct a 7-tier ablation study:

```
[Ablation 1]: Random Sentence Selection (Baseline heuristic)
     |
[Ablation 2]: Standard BM25 Retrieval (Lexical proxy)
     |
[Ablation 3]: Dense Embedding Retrieval (BGE-M3 semantic proxy)
     |
[Ablation 4]: LLM Extraction Only (Raw concatenated text spans)
     |
[Ablation 5]: LLM Extraction + Schema Structuring (Tabular key-value format)
     |
[Ablation 6]: LLM Extraction + Structuring + Sufficiency Filter (AutoProxy-CoT full data)
     |
[Ablation 7]: AutoProxy-CoT + Grounding Citation Loss + Curriculum Noise (Full proposed pipeline)
     |
[Upper Bound]: Human-Annotated Proxies (Original ProxyCoT gold standard)
```

#### 5.4. Metrics and Evaluation Dimensions

We measure performance across four distinct axes:

1. **Reasoning Task Accuracy:**
   - Exact Match (EM) and Macro-averaged F1 score on QA pairs.
2. **Proxy Quality Assessment:**
   - **Evidence Recall:** Fraction of gold supporting sentences captured in the proxy:
     $$\text{Recall}_{\text{ev}} = \frac{|P \cap S^*|}{|S^*|}$$
   - **Evidence Precision:** Density of relevant facts within the proxy:
     $$\text{Precision}_{\text{ev}} = \frac{|P \cap S^*|}{|P|}$$
3. **Grounding and Hallucination Diagnostics:**
   - Citation Accuracy: Precision of `<grounding>` span IDs relative to true premises.
   - Hallucination Rate: Frequency of factual assertions in `<thought>` that cannot be resolved to any span in $C$.
4. **Positional Robustness ("Lost-in-the-Middle"):**
   - Accuracy curves plotted as a function of the normalized depth of gold evidence: $\text{Depth} \in [0.0, 1.0]$ in increments of $0.1$.
5. **Computational and Economic Efficiency:**
   - Total GPU-hours consumed across all training stages.
   - Prefill latency vs. generation latency across context sizes (8K, 32K, 64K, 128K, 512K).
   - API dollar cost per 1,000 synthetic training instances (utilizing prompt caching and batch pricing).

---

### 6. Technical Implementation and System Architecture

#### 6.1. Extraction Economics and Prompt Caching
Processing thousands of 128K-token documents using commercial APIs could be cost-prohibitive without careful design. We implement two specific cost-reduction mechanisms:
- **Prompt Prefix Caching:** Because multi-hop benchmarks frequently ask multiple questions about the same large document or collection of papers (as in SciTrek and Loong), caching the shared context prefix reduces input token costs by up to 80%.
- **High-Throughput Batch Processing:** Non-interactive extraction jobs are dispatched via provider Batch APIs (50% cost discount).
- **Self-Hosted Open SLM Fallback:** To ensure absolute reproducibility and independence from closed APIs, we benchmark an offline extractor using `Qwen2.5-7B-Instruct-1M` running on local dual-NVIDIA RTX 4090 / A100 nodes via vLLM with PagedAttention.

#### 6.2. Training Hyperparameters and Infrastructure Plan

```markdown
### Target Training Hyperparameters
* Base Models: Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B-Instruct
* Sequence Length Limits: 
    - Stage 1 (Proxy RLVR): 2,048 tokens
    - Stage 2 (Grounded SFT): 32,768 - 131,072 tokens (via RoPE scaling)
* RL Framework: GRPO / OpenRLHF with DeepSpeed ZeRO-3
* Learning Rate: 
    - Stage 1: 5e-7 (with cosine decay and warmup)
    - Stage 2: 2e-5
* Batch Sizes: Effective batch size of 64 (gradient accumulation = 8, micro-batch = 1)
* Hardware: 4 x NVIDIA A100-SXM4-80GB (or 8 x H100) cluster
```

---

### 7. Risk Analysis and Mitigation Strategies

| Potential Risk / Failure Mode | Likelihood | Impact | Concrete Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Circularity Critique:** Reviewers claim that using an LLM to read long context to make proxies invalidates the efficiency motivation. | Medium | High | Clearly distinguish between *one-time amortized training-data synthesis* and *repetitive test-time inference*. Training extraction occurs once per document, whereas the resulting model performs inference across millions of user queries with drastically shorter CoT traces. We will explicitly quantify the breakeven inference volume. |
| **Answer Leakage:** Extractor inadvertently favors or exposes answers. | High | Critical | Enforce strict input masking: the extractor prompt contains *only* $(q, c_i)$ without access to labels. The answer is used solely as a Boolean post-hoc gatekeeper in Phase 3. |
| **Extractor Hallucination:** Extractor invents facts not present in chunk $c_i$. | Low | Medium | Enforce verbatim substring matching in the JSON parsing script. Any extracted span failing Python `span in chunk_text` is automatically dropped. |
| **Overfitting to Templated SciTrek:** Short traces may overfit to synthetic template questions. | Medium | High | Evaluate models out-of-domain on natural, non-templated datasets (HotpotQA, MuSiQue, and Loong). If templates bias the policy, apply trace length regularization in the reward function. |
| **Hardware / VRAM Constraints at 128K+:** OOM errors during Stage 2 SFT. | Medium | High | Utilize FlashAttention-3, ZeRO-Offload, and 4-bit/8-bit LoRA (QLoRA) parameter-efficient tuning as an alternative ablation for resource-constrained environments. |

---

### 8. Work Plan, Milestones, and Timeline

The project is structured across a four-month execution timeline (16 weeks):

```
Weeks 1 - 3: Pipeline Engineering & Data Infrastructure
  [===] Chunking engine, API extraction with JSON schema, prompt caching
  [===] Baseline HotpotQA & SciTrek data ingestion, verification filter build

Weeks 4 - 6: Automated Proxy Synthesis & Ablation Experiments
  [===] Generate datasets for Ablations 1 - 6 (Random, BM25, Embedding, Raw LLM, Structured, Filtered)
  [===] Benchmark proxy evidence precision, recall, and token compression metrics

Weeks 7 - 10: Model Training (Stage 1 RLVR & Stage 2 Grounded SFT)
  [===] Conduct Stage 1 GRPO on verified proxies
  [===] Execute Stage 2 Grounded SFT with citation format and curriculum noise schedule

Weeks 11 - 13: Comprehensive Evaluation & Diagnostics
  [===] Full test runs on HotpotQA, SciTrek, MuSiQue, 2WikiMQA, LongBench v2
  [===] Diagnostic profiling: Lost-in-the-middle curves, hallucination audits, GPU-hour accounting

Weeks 14 - 16: Thesis / Paper Synthesis & Camera-Ready Revisions
  [===] Draft complete results, format ablation tables, finalize discussion and open-source release
```

---

### 9. Expected Contributions and Broader Impact

1. **A Fully Autonomous Proxy Generation Framework:** We resolve the central limitation of ProxyCoT by replacing human annotation with an automated, schema-driven extraction and verification pipeline that achieves human-level proxy efficacy.
2. **Empirical Validation of Structured Schemas in Reasoning:** We systematically quantify the performance advantage of structured key-value representations over unstructured prose, providing empirical guidelines for training context design.
3. **Grounded Long-Context Reasoning Mechanism:** Our *cite-before-reason* trace format and curriculum noise schedule bridge the transfer gap, preventing models from degenerating when deployed on noisy, ultra-long contexts.
4. **Transparent Cost and Latency Accounting:** We deliver reproducible wall-clock, GPU-hour, and token-efficiency benchmarks, providing the research community with an honest appraisal of the computational trade-offs inherent in proxy-based reasoning.

---

### 10. References and Bibliography

1. **Li, X., Saparina, A., Gurung, C., & Lapata, M.** (2026). *Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning*. arXiv preprint arXiv:2605.20201v2.
2. **Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D.** (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP 2018.
3. **Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A.** (2022). *MuSiQue: Multihop Questions via Single-hop Question Inversion*. TACL, 10, 539-554.
4. **Ho, X., Duong Nguyen, A. K., Sugawara, S., & Aizawa, A.** (2020). *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps*. COLING 2020.
5. **Bai, Y., Lv, X., Zhang, J., et al.** (2024). *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*. ACL 2024.
6. **Hsieh, C. Y., et al.** (2024). *RULER: What’s the Real Context Size of Your Long-Context Language Models?* arXiv preprint arXiv:2404.06654.
7. **Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P.** (2024). *Lost in the Middle: How Language Models Use Long Contexts*. TACL, 12, 157-173.
8. **Shao, Z., Wang, P., Zhu, Q., et al.** (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv preprint arXiv:2402.03300.
9. **Kwon, W., Li, Z., Zhuang, S., et al.** (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.
10. **Dao, T.** (2024). *FlashAttention-3: Fast and Memory-Efficient Exact Attention with Asynchrony and Low-Precision FP8*. arXiv preprint arXiv:2407.08608.
