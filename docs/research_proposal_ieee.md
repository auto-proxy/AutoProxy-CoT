AutoProxy-CoT: Automated Proxy Synthesis and Evidence-Grounded Chain-of-Thought Tuning for Long-Context Reasoning in Compact Language Models

ASHIK RAJAN                 ARIGHNA NANDI              DILKUSH M             Prof. Divyanshu Prakash
Department of AIML        Department of AIML        Department of AIML          Department of AIML
 Jain University            Jain University           Jain University             Jain University
 Bengaluru, India          Bengaluru, India          Bengaluru, India            Bengaluru, India
                                                                               (Faculty Supervisor)

I. INTRODUCTION AND BACKGROUND
Large Language Models (LLMs) have achieved remarkable success in multi-step deductive reasoning through Chain-of-Thought (CoT) prompting. However, when deployed on long contexts (32K to 128K+ tokens) in tasks such as multi-hop question answering and scientific literature synthesis, models suffer from severe attention dilution, quadratic computational scaling, and the lost-in-the-middle phenomenon [4]. Reinforcement Learning with Verifiable Rewards (RLVR) directly on long contexts is computationally prohibitive due to KV-cache memory saturation and excessive prefill latency during exploratory rollouts.

Recently, the Proxy-based Chain-of-Thought Tuning (ProxyCoT) framework was proposed by Li et al. [1], based on the Reasoning Invariance Hypothesis: multi-step deductive reasoning is structurally invariant to irrelevant context. By training the reasoning policy on compact "proxy contexts" (1K to 2K tokens) via RLVR (Stage 1) and transferring this capability to full documents via Supervised Fine-Tuning (Stage 2), ProxyCoT demonstrated significant training speedups. 

However, ProxyCoT harbors an acute dependency: its empirical success relies almost entirely on human-annotated proxies. When the authors attempted heuristic automation, performance collapsed drastically—dropping to 35.1% accuracy on HotpotQA using BM25 or dense retrieval (compared to ~92% with human proxies), and falling to 3.4%–15.8% with random sentence selection [1]. Furthermore, Stage 2 SFT trains models on traces generated strictly over isolated proxies, leaving the policy ungrounded and unable to locate or cite evidence within noisy full documents. This proposal presents AutoProxy-CoT, an automated framework that eliminates human annotations via chunk-level LLM extraction, schema structuring, and closed-loop sufficiency verification, paired with explicit citation grounding.

II. LITERATURE REVIEW
Prior research in long-context reasoning, multi-hop question answering, and evidence verification spans retrieval-augmented generation, proxy optimization, and context-position analysis:

| Research Paper | Approach / Methodology | Key Contribution | Relevance to Our Study |
| :--- | :--- | :--- | :--- |
| Li et al. (2026) [1] | Trains reasoning policies via RLVR on short proxy contexts; transfers to 128K tokens via SFT. | Formulated reasoning invariance across context scales; demonstrated token efficiency in CoT rollouts. | Serves as our primary foundational paper; automated retrieval proxies collapsed to 35.1%, which our study resolves. |
| Yang et al. (2018) [2] | Formulates multi-hop reasoning over Wikipedia bridge entities with sentence-level annotations. | Introduced the HotpotQA benchmark and established multi-hop evidence evaluation standards. | Provides the core multi-hop benchmark with gold supporting sentences to measure evidence precision and recall. |
| Trivedi et al. (2022) [3] | Constructs disconnected multi-hop questions via single-hop question inversion. | Eliminates superficial lexical shortcuts and multi-hop reasoning leaks. | Evaluates whether automated proxies generalize to complex, non-templated reasoning graphs. |
| Liu et al. (2024) [4] | Evaluates language model retrieval and reasoning accuracy as a function of evidence depth. | Identified systematic "lost-in-the-middle" accuracy degradation in long contexts. | Provides the core rationale for our explicit citation grounding mechanism to anchor attention across full contexts. |
| Wang et al. (2023) [5] | Samples diverse reasoning paths and aggregates final answers via majority voting. | Demonstrated that test-time path consistency improves reasoning accuracy without parameter updates. | Serves as our primary inference-time baseline for multi-path reasoning comparison against compact models. |

AutoProxy-CoT builds upon these foundations by automating proxy generation, exploiting schema structuring, and introducing grounded citation loss to ensure robust long-context transfer.

III. RESEARCH GAP AND PROBLEM STATEMENT
Although ProxyCoT demonstrates that reasoning policies can be trained on compact proxies to reduce long-context RLVR overhead [1], its practical deployment is obstructed by its dependency on manual human annotations, with automated retrieval baselines experiencing an acute 56.9-point accuracy collapse (falling to 35.1% on HotpotQA). Additionally, the model is never trained to locate or cite evidence within long context during Stage 2 SFT, leaving it susceptible to positional bias and hallucinated premises.

Research Gap Statement: Although ProxyCoT demonstrates the theoretical validity of reasoning invariance between compact proxies and long contexts, its reliance on manual human proxy construction prevents scalable deployment. Automated sparse and dense retrieval mechanisms fail to preserve multi-hop reasoning pathways, causing severe accuracy collapse. Furthermore, the absence of explicit evidence grounding during Stage 2 SFT leaves the model vulnerable to distractors and position-dependent attention dilution. This study addresses this gap by developing AutoProxy-CoT, an automated chunk-level extraction and schema-structuring pipeline that validates proxy sufficiency without answer leakage, and introduces an explicit citation grounding loss to bridge the transfer gap on compact open models (Qwen2.5-7B-Instruct and Gemma-2-9B-It).

IV. RESEARCH AIM, OBJECTIVES, AND RESEARCH QUESTIONS
Research Aim: To evaluate the effectiveness of AutoProxy-CoT for automated proxy synthesis and evidence-grounded long-context reasoning using compact language models (Qwen2.5-7B-Instruct and Gemma-2-9B-It) on HotpotQA and SciTrek, by comparing its accuracy against heuristic retrieval and human-proxy baselines, analysing the contributions of schema structuring and sufficiency filtering, and quantifying computational cost.

Objectives:
1) Implement an automated chunk-level LLM extraction and schema-structuring pipeline with strict zero-answer-leakage guarantees.
2) Design and validate a closed-loop sufficiency verification filter to prune defective proxies prior to policy optimization.
3) Implement explicit span citation grounding in Stage 2 SFT paired with a curriculum noise schedule ($P \to P \cup \mathcal{N}_k \to C$).
4) Evaluate downstream multi-hop accuracy, evidence recall/precision, positional lost-in-the-middle robustness, and computational overhead in terms of prefill latency, generation time, and GPU-hours.

Research Questions:
RQ1: Does AutoProxy-CoT eliminate the performance collapse of automated proxies (35.1%) compared to human-annotated proxies (~92%) when evaluated on HotpotQA and SciTrek?
RQ2: How do schema-driven structuring (tables/key-values) and sufficiency filtering individually contribute to proxy quality and downstream accuracy?
RQ3: Does explicit citation grounding in Stage 2 SFT alleviate lost-in-the-middle degradation across context windows from 32K to 128K tokens?
RQ4: What is the net computational and financial trade-off of offline API-based proxy synthesis compared to direct long-context RL exploration?

V. DATASET: HOTPOTQA AND SCITREK RESEARCH AND DATA PLAN
This study utilizes two primary benchmarks representing diverse long-context and multi-hop reasoning challenges:
1) HotpotQA: Obtained from Hugging Face (`hotpot_qa`, distractor and full-wiki configurations) [2]. The dataset contains 90,447 training pairs and 7,405 validation pairs under the standard split. Each example consists of a natural language `question`, ten context paragraphs containing distractors, and gold `supporting_facts` with sentence-level indices, terminated by a concise `answer`.
2) SciTrek: A multi-step scientific literature reasoning benchmark spanning sequence lengths from 32K to 128K tokens, terminating in verifiable factual answers grounded in academic papers [1].

Evaluation Protocol: To ensure rigorous evaluation under academic computational resources, an initial stratified evaluation subset of 200 multi-hop questions will be sampled without replacement from the held-out validation split of HotpotQA and SciTrek using a fixed random seed (42). All compared methods, ablations, and baselines will be benchmarked on this identical fixed subset.

VI. PROPOSED MODEL AND METHODOLOGY
We propose an automated evaluation setup using Qwen2.5-7B-Instruct and Gemma-2-9B-It, chosen for their competitive reasoning capabilities relative to parameter count. For full context document $C$ partitioned into overlapping chunks $c_1, \dots, c_m$ of size $L_c = 2048$ with stride $S = 1792$, the AutoProxy-CoT pipeline operates via four formal stages:

Chunk Extraction:
S_i = Extract(c_i, q)    where c_i in C, with Answer Masking a* not in Prompt

Schema Structuring:
P* = Structure(Union_{i=1}^m S_i)

Sufficiency Verification:
Gate(P*, q) = 1( M_ver(P*, q) == a* )

Grounded SFT Loss:
L_Grounded(theta) = - Sum_{t=1}^{|tau|} log pi_theta( tau_t | C, q, tau_{<t} )

where:
tau = <grounding> [c_k_s_j] </grounding> <thought> ... </thought> <answer> a </answer>

Here, $q$ denotes the query, $a^*$ represents the gold target answer, and $\mathcal{M}_{\text{ver}}$ is a lightweight zero-shot verifier. The extractor receives strictly $(c_i, q)$ without answer visibility to prevent leakage. The gold answer is utilized solely as a post-hoc gatekeeper. Verified proxies feed into Stage 1 RLVR, while Stage 2 SFT enforces explicit evidence citation tokens prior to reasoning chain emission.

VII. EXPERIMENTAL AND EVALUATION PLAN
| Method | Description / Purpose |
| :--- | :--- |
| Zero-Shot Direct | Single-output baseline evaluating base model performance directly on full context C without fine-tuning. |
| Standard Long-Context SFT | Directly fine-tunes on (C, q -> tau) across full 32K-128K context; serves as primary direct baseline. |
| Heuristic ProxyCoT (BM25 / Dense) | Replicates the original paper's automated retrieval failure point (scoring 35.1% on HotpotQA) [1]. |
| AutoProxy-CoT (Proposed) | Evaluates the complete automated extraction, structuring, verification, and grounded SFT framework. |
| Ablation 1: Raw Text Proxy | Concatenates extracted spans as flat prose to evaluate the benefit of schema structuring (Table 9 in [1]). |
| Ablation 2: No Sufficiency Gate | Trains on candidate proxies without verification to assess the impact of noisy extraction filtering. |
| Ablation 3: No Citation Grounding | Trains Stage 2 SFT without <grounding> citation tokens to isolate position-dependent attention degradation. |
| Human ProxyCoT (Oracle Upper Bound) | Evaluates ProxyCoT trained with gold human-annotated proxies (~92.0% upper bound from Li et al.) [1]. |

All selection methods will evaluate identical test instances under standardized sampling parameters (temperature 0.7, top-p 0.8). Greedy decoding serves as a deterministic reference, while Human ProxyCoT establishes the empirical ceiling.

Metrics: Exact-Match (EM) accuracy, Macro F1, Evidence Recall ($|P \cap S^*| / |S^*|$), Evidence Precision ($|P \cap S^*| / |P|$), Citation Faithfulness, Lost-in-the-Middle accuracy across normalized depths $[0.0, 1.0]$, prefill latency, generation time, and total GPU-hours.

VIII. REFERENCES
[1] X. Li, A. Saparina, C. Gurung, and M. Lapata, "Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning," arXiv preprint arXiv:2605.20201v2, 2026.
[2] Z. Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering," in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2369–2380, 2018.
[3] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, "MuSiQue: Multihop Questions via Single-hop Question Inversion," Transactions of the Association for Computational Linguistics (TACL), vol. 10, pp. 539–554, 2022.
[4] N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," Transactions of the Association for Computational Linguistics (TACL), vol. 12, pp. 157–173, 2024.
[5] X. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models," in International Conference on Learning Representations (ICLR), 2023. arXiv:2203.11171.
