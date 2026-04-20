# Awesome-LLM4SVD 🌟-🧠👩‍💻🔍

This repository contains the artifacts from the systematic literature review (SLR) on LLM-based software vulnerability detection ("A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models"). 
The SLR analyzes 263 studies published between January 2020 and November 2025 and provides a structured taxonomy of detection approaches, input representations, system architectures, techniques, and dataset usage.


## Table of Contents

To support open science and reproducibility, we publicly release:
- 📝 [Surveyed Papers](#papers): A curated list of surveyed papers. This list will be continuously updated to track the latest papers.
- 🗂️ [Taxonomy](https://github.com/hs-esslingen-it-security/Awesome-LLM4SVD/tree/main/taxonomy): Taxonomy of LLM-based vulnerability detection studies along with the categorization of each surveyed paper.
- 📝 [Selected Datasets](#datasets): A list of the most commonly used datasets in the surveyed studies with their download sources.



<br>

For details, see our [preprint here](https://arxiv.org/abs/2507.22659): 

📚 S. Kaniewski, F. Schmidt, M. Enzweiler, M. Menth, und T. Heer. 2025. *A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models*. arXiv:2507.22659.
```bibtex
@preprint{kaniewskiLLM4SVD2025,
    title={{A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models}}, 
    author={Kaniewski, Sabrina and Schmidt, Fabian and Enzweiler, Markus and Menth, Michael and Heer, Tobias},
    year={2025},
    eprint={2507.22659},
    archivePrefix={arXiv},
    primaryClass={cs.SE},
    url={https://arxiv.org/abs/2507.22659}, 
}
```
Please cite our paper if you use this resource.

<br>

- 🤝 [Contribute to this repository](#contribution)
- ⚖️ [License](#license)


<br>

----------------
----------------

## Papers

> **Note:** Entries marked with ✨ indicate the latest papers that are not discussed in the preprint of the SLR. The latest preprint version covers all studies up to November 2025.

### 2026
- ✨ (03/2026) Evaluating Retrieval-Augmented Generation for LLM-Based Vulnerability Detection: An Empirical Study on Real-World Java Vulnerabilities.  **`IEEE Access 2026`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11450344)]
- ✨ (03/2026) CTX-Coder: Cross-Attention Architectures Empower LLMs for Long-Context Vulnerability Detection.  **`AAAI 2026`** [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37087)] [[Code](https://github.com/wangjvjie/CTX-Coder)]
- ✨ (02/2026) Leveraging Transformers to Discover Software Vulnerabilities based on Source Code Slices.  **`AISC 2026`** [[Paper](https://dl.acm.org/doi/10.1145/3793638.3793639)]
- ✨ (02/2026) Enhancing Continual Learning for Software Vulnerability Prediction: Addressing Catastrophic Forgetting via Hybrid-Confidence-Aware Selective Replay for Temporal LLM Fine-Tuning.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.23834v1)]
- ✨ (02/2026) From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.14012v1)] [[Code](https://github.com/youpengl/OpenVul)]
- ✨ (02/2026) SecCodePRM: A Process Reward Model for Code Security.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.10418v1)] [[Code](https://github.com/viviable/seccodeprm)]
- ✨ (02/2026) VulReaD: Knowledge-Graph-guided Software Vulnerability Reasoning and Detection.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.10787v1)] [[Code](https://anonymous.4open.science/r/Vul-ReaD)]
- ✨ (02/2026) Beyond Function-Level Analysis: Context-Aware Reasoning for Inter-Procedural Vulnerability Detection.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06751v1)] [[Code](https://github.com/yikun-li/CPRVul)]
- ✨ (02/2026) Evaluating and Enhancing the Vulnerability Reasoning Capabilities of Large Language Models.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06687v1)]
- ✨ (02/2026) One Model, Many Skills: Parameter-Efficient Fine-Tuning for Multitask Code Analysis.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2603.09978)] [[Code](https://github.com/Amal-AK/multitask_PEFT)]
- ✨ (01/2026) RAG-Enhanced Multi-Model Ensemble for Automated Vulnerability Detection Using SLMs.  **`ICECTE 2026`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11429262)] [[Code](https://github.com/rafi79/RAG-Enhanced-Multi-Model-Ensemble-for-Automated-Vulnerability-Detection-Using-SLMs)]
- ✨ (01/2026) LLMs in Code Vulnerability Analysis: A Proof of Concept.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.08691)] [[Code](https://figshare.com/s/a06ec09cd1bd98e6dd45)]
- ✨ (01/2026) MulVul: Retrieval-augmented Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.18847)]
- ✨ (01/2026) LLM-based Vulnerability Detection at Project Scale: An Empirical Study.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.19239)] [[Code](https://github.com/Feng-Jay/LLM4Security)]
- ✨ (01/2026) The Semantic Trap: Do Fine-tuned LLMs Learn Vulnerability Root Cause or Just Functional Pattern?.  **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.22655)] [[Code](https://anonymous.4open.science/r/TrapEval)]


### 2025
- ✨ (12/2025) ResVul-LLM: A Neurosymbolic Framework Combining Large Language Models and Symbolic Reasoning for C/C++ Vulnerability Analysis.  **`BigData 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11402220)]
- ✨ (12/2025) CodeVul+: A Structure-Aware Framework for Cross-Repository Vulnerability Detection.  **`BigData 2025`** [[Paper](https://ieeexplore.ieee.org/document/11401065)]
- ✨ (12/2025) Trust-Calibrated Multi-Stage Large Language Model Pipeline for Vulnerability Assessment in DevSecOps Workflows.  **`ACSAC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11417999)]
- ✨ (12/2025) Large Language Models Cannot Reliably Detect Vulnerabilities in JavaScript: The First Systematic Benchmark and Evaluation.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.01255)] [[Code](https://github.com/SecJS-Vuln-Benchmark/SecJS-Benchmark)] [[Code](https://secjs-vuln-benchmark.github.io/SecJS-Benchmark/)]
- ✨ (12/2025) The Impact of Prompt Language and Representation on LLM Reasoning: A Multilingual Empirical Study.  **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/document/11318327)]
- ✨ (12/2025) A Systematic Study of Code Obfuscation Against LLM-based Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.16538)] [[Code](https://github.com/oxygen-hunter/SoK-Code-Obfuscation-in-LLM-VD-arxiv)]
- ✨ (12/2025) From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.10485)] [[Code](https://github.com/Chaomeng-Lu/A-Practical-Evaluation-of-Deep-Learning-Models-and-LLMs-for-Vulnerability-Detection)]
- ✨ (12/2025) VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.07533)] [[Code](https://github.com/ucsb-mlsec/VulnLLM-R)]
- ✨ (12/2025) On the Effectiveness of Instruction-Tuning Local LLMs for Identifying Software Vulnerabilities.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.20062)]
- ✨ (12/2025) Diverse LLMs vs. Vulnerabilities: Who Detects and Fixes Them Better?.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.12536)] [[Code](https://github.com/Erroristotle/DVDR_LLM)]
- ✨ (11/2025) Retrieval-Augmented Few-Shot Prompting Versus Fine-Tuning for Code Vulnerability Detection.  **`FLLM 2025`** [[Paper](https://ieeexplore.ieee.org/document/11391248)]
- ✨ (11/2025) Should We Evaluate LLM Based Security Analysis Approaches on Open Source Systems?.  **`ASE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11334389/)]
- ✨ (11/2025) LOSVER: Line-Level Modifiability Signal-Guided Vulnerability Detection and Classification.  **`ASE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11334430)] [[Code](https://github.com/waroad/losver)] [[Code](https://figshare.com/articles/conference_contribution/Backup_code_and_checkpoints_for_Localizer_and_Detector_from_paper_b_LOSVER_Line-Level_Modifiability_Signal-Guided_Vulnerability_Detection_and_Classification_b_/29192708)]
- ✨ (11/2025) An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems.  **`CASCON 2025`** [[Paper](https://ieeexplore.ieee.org/document/11344502)]
- (11/2025) Leveraging Self-Paced Learning for Software Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.09212)] [[Code](https://figshare.com/s/bef3211194fc18fe375e)]
- (11/2025) Specification-Guided Vulnerability Detection with Large Language Models.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.04014)] [[Code](https://github.com/zhuhaopku/VulInstruct-temp)]
- (11/2025) Compressing Large Language Models for SQL Injection Detection: A Case Study on Deep Seek-Coder and Meta-llama-3-70b-instruct.  **`FRUCT 2025`** [[Paper](https://ieeexplore.ieee.org/document/11239157)]
- (11/2025) VulTrLM: LLM-assisted Vulnerability Detection via AST Decomposition and Comment Enhancement.  **`EMSE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10664-025-10738-7)]
- (11/2025) Cross-Domain Evaluation of Transformer-Based Vulnerability Detection on Open and Industry Data.  **`PROFES 2025`** [[Paper](https://arxiv.org/abs/2509.09313)] [[Code](https://github.com/CybersecurityLab-unibz/cross_domain_evaluation)]
- (11/2025) Learning-based Models for Vulnerability Detection: An Extensive Study.  **`EMSE 2025`** [[Paper](https://arxiv.org/abs/2408.07526)] [[Code](https://figshare.com/s/bde8e41890e8179fbe5f?file=41894784)]
- (11/2025) A Sequential Multi-Stage Approach for Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making.  **`EMNLP 2025`** [[Paper](https://aclanthology.org/2025.emnlp-main.1071/)]
- ✨ (10/2025) Transfer-Guided Konwledge Distillation for Enhancing Cross-Project Vulnerability Detection.  **`CCNS 2025`** [[Paper](https://ieeexplore.ieee.org/document/11337967)]
- ✨ (10/2025) Code Vulnerability Detection Method Based On PreTrained Language Model and Gating Graph Neural Network.  **`CBASE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11335562)]
- ✨ (10/2025) VLD-LP: Vulnerability Detection and Root Cause Localization with Large Language Model and Parameter-efficient Language Model Tuning.  **`SMC 2025`** [[Paper](https://ieeexplore.ieee.org/document/11343151)]
- ✨ (10/2025) The Richer Representation Fallacy: Are We Just Adding Noise to LLM-based Software Vulnerability Detectors?.  **`ICOCO 2025`** [[Paper](https://ieeexplore.ieee.org/document/11334069)]
- ✨ (10/2025) CTVD: Collaborative Training of Deep Learning and Large Model for C/C++ Source Code Vulnerability Detection.  **`SMC 2025`** [[Paper](https://ieeexplore.ieee.org/document/11343541)]
- (10/2025) Leveraging Intra-and Inter-References in Vulnerability Detection using Multi-Agent Collaboration Based on LLMs.  **`Cluster Computing 2025`** [[Paper](https://link.springer.com/article/10.1007/s10586-025-05721-2)]
- (10/2025) iCodeReviewer: Improving Secure Code Review with Mixture of Prompts.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.12186)]
- (10/2025) Bridging Semantics \& Structure for Software Vulnerability Detection using Hybrid Network Models.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.10321)] [[Code](https://zenodo.org/records/17259519)]
- (10/2025) FuncVul: An Effective Function Level Vulnerability Detection Model using LLM and Code Chunk.  **`ESORICS 2025`** [[Paper](https://arxiv.org/abs/2506.19453)] [[Code](https://github.com/sajalhalder/FuncVul)]
- (10/2025) On Selecting Few-Shot Examples for LLM-based Code Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.27675)]
- (10/2025) A Zero-Shot Framework for Cross-Project Vulnerability Detection in Source Code.  **`EMSE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10664-025-10749-4)] [[Code](https://github.com/Radowan98/ZSVulD)]
- (10/2025) Towards Explainable Vulnerability Detection With Large Language Models.  **`TSE 2025`** [[Paper](https://arxiv.org/abs/2406.09701)]
- (10/2025) MulVuln: Enhancing Pre-trained LMs with Shared and Language-Specific Knowledge for Multilingual Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.04397)]
- (10/2025) Llama-Based Source Code Vulnerability Detection: Prompt Engineering vs Fine Tuning.  **`ESORICS 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-032-07884-1_15)] [[Code](https://github.com/DynaSoumhaneOuchebara/Llama-based-vulnerability-detection)]
- (10/2025) Real-VulLLM: An LLM Based Assessment Framework in the Wild.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.04056)]
- ✨ (10/2025) A Comprehensive Comparison of LLaMA 3.1 and Traditional ML Approaches in Automated Vulnerability Detection.  **`AICCSA 2025`** [[Paper](https://ieeexplore.ieee.org/document/11315404)]
- (10/2025) Distilling Lightweight Language Models for C/C++ Vulnerabilities.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.06645)] [[Code](https://github.com/yangxiaoxuan123/ FineSec_detect)]
- (10/2025) Sparse-MoE: Syntax-Aware Multi-view Mixture of Experts for Long-Sequence Software Vulnerability Detection.  **`ADMA 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-981-95-3456-2_24)]
- ✨ (09/2025) Transformer-Based Semantic Embeddings and Hybrid Neural Networks for Robust Software Vulnerability Detection.  **`i-PACT 2025`** [[Paper](https://ieeexplore.ieee.org/document/11307989)]
- (09/2025) DeepVulHunter: Enhancing the Code Vulnerability Detection Capability of LLMs through Multi-Round Analysis.  **`JIIS 2025`** [[Paper](https://link.springer.com/article/10.1007/s10844-025-00982-0)]
- (09/2025) Can LLM Prompting Serve as a Proxy for Static Analysis in Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2412.12039)]
- (09/2025) GPTVD: vulnerability detection and analysis method based on LLM’s chain of thoughts.  **`ASE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10515-025-00550-4)] [[Code](https://github.com/chenyn273/GPTVD)]
- (09/2025) An Advanced Detection Framework for Embedded System Vulnerabilities.  **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11153853)]
- (09/2025) Utilizing Large Programming Language Models on Software Vulnerability Detection.  **`ASYU 2025`** [[Paper](https://ieeexplore.ieee.org/document/11208282)]
- (09/2025) MAVUL: Multi-Agent Vulnerability Detection via Contextual Reasoning and Interactive Refinement.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.00317)] [[Code](https://github.com/youpengl/MAVUL)]
- (09/2025) PIONEER: Improving the Robustness of Student Models when Compressing Pre-Trained Models of Code.  **`ASE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10515-025-00560-2)] [[Code](https://github.com/illsui1on/PIONEER)]
- (09/2025) Ensembling Large Language Models for Code Vulnerability Detection: An Empirical Evaluation.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2509.12629)] [[Code](https://github.com/sssszh/ELVul4LLM)]
- (09/2025) VulAgent: Hypothesis-Validation based Multi-Agent Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2509.11523)]
- ✨ (08/2025) Optimizing Code Vulnerability Detection via GRPO and SFT Fine-Tuning of Compact LLMs.  **`DSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11360339)]
- (08/2025) VulPr: A Prompt Learning-based Method for Vulnerability Detection.  **`EIT 2025`** [[Paper](https://ieeexplore.ieee.org/document/11231886)]
- (08/2025) Improving Software Security Through a LLM-Based Vulnerability Detection Model.  **`DEXA 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-032-02049-9_9)]
- (08/2025) MalCodeAI: Autonomous Vulnerability Detection and Remediation via Language Agnostic Code Reasoning.  **`IRI 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11153184)]
- (08/2025) Large Language Models Versus Static Code Analysis Tools: A Systematic Benchmark for Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/pdf/2508.04448)] [[Code](https://github.com/Damian0401/ProjectAnalyzer)]
- (08/2025) Enhancing Fine-Grained Vulnerability Detection With Reinforcement Learning.  **`TSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11145224)] [[Code](https://github.com/YuanJiangGit/RLFD)]
- (08/2025) CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.11599)]
- (08/2025) Out of Distribution, Out of Luck: How Well Can LLMs Trained on Vulnerability Datasets Detect Top 25 CWE Weaknesses?.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.21817)] [[Code](https://github.com/yikun-li/TitanVul-BenchVul)]
- (08/2025) LLM-GUARD: Large Language Model-Based Detection and Repair of Bugs and Security Vulnerabilities in C++ and Python.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.16419)] [[Code](https://github.com/NoujoudNader/LLM-Bugs-Detection)]
- (08/2025) Multimodal Fusion for Vulnerability Detection: Integrating Sequence and Graph-Based Analysis with LLM Augmentation.  **`MAPR 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11133833)]
- (08/2025) SAFE: A Novel Approach For Software Vulnerability Detection from Enhancing The Capability of Large Language Models.  **`ASIACCS 2025`** [[Paper](https://arxiv.org/abs/2409.00882)]
- (08/2025) Software Vulnerability Detection using Large Language Models.  **`SecureComm 2025`** [[Paper](https://arxiv.org/abs/2410.00249)]
- (08/2025) Data and Context Matter: Towards Generalizing AI-based Software Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.16625)]
- (08/2025) Think Broad, Act Narrow: CWE Identification with Multi-Agent Large Language Models.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.01451)] [[Code](https://zenodo.org/records/15871507)]
- ✨ (07/2025) Enhancing Vulnerability Detection by Fusing Code Semantic Features with LLM-generated Explanations.  **`Information Fusion 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253525005238)] [[Code](https://github.com/XUPT-SSS/FuSEVul)]
- ✨ (07/2025) Structural Semantic Enhancement: Better Integrating Code Semantics for Vulnerability Detection.  **`InfSof 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0950584925001636?via%3Dihub)]
- (07/2025) An Automatic Classification Model for Long Code Vulnerabilities Based on the Teacher-Student Framework.  **`QRS 2025`** [[Paper](https://ieeexplore.ieee.org/document/11216609)]
- (07/2025) LLMxCPG: Context-Aware Vulnerability Detection Through Code Property Graph-Guided Large Language Models.  **`USENIX Security 2025`** [[Paper](https://arxiv.org/abs/2507.16585)] [[Code](https://github.com/qcri/llmxcpg)] [[Code](https://zenodo.org/records/15614095)]
- (07/2025) CLeVeR: Multi-modal Contrastive Learning for Vulnerability Code Representation.  **`ACL 2025`** [[Paper](https://aclanthology.org/2025.findings-acl.414/)] [[Code](https://github.com/yoimiya-nlp/CLeVeR)]
- (07/2025) Revisiting Pre-trained Language Models for Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.16887)]
- (07/2025) Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.03051)]
- (07/2025) HgtJIT: Just-in-Time Vulnerability Detection Based on Heterogeneous Graph Transformer.  **`TDSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11072308)]
- (07/2025) AI-Powered Vulnerability Detection in Code Using BERT-Based LLM with Transparency Measures.  **`ITC-Egypt 2025`** [[Paper](https://ieeexplore.ieee.org/document/11186618)]
- (07/2025) Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories.  **`Unknown 2025`** [[Paper](https://arxiv.org/abs/2503.03586)]
- (06/2025) VulnTeam: A Team Collaboration Framework for LLM-based Vulnerability Detection.  **`IJCNN 2025`** [[Paper](https://ieeexplore.ieee.org/document/11229292)]
- (06/2025) One-for-All Does Not Work! Enhancing Vulnerability Detection by Mixture-of-Experts (MoE).  **`PACMSE 2025`** [[Paper](https://arxiv.org/abs/2501.16454)]
- (06/2025) Improving Vulnerability Type Prediction and Line-Level Detection via Adversarial Training-based Data Augmentation and Multi-Task Learning.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.23534)] [[Code](https://github.com/Karelye/EDAT-MLT)]
- (06/2025) Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2406.11147)] [[Code](https://github.com/knowledgerag4llmvuld/knowledgerag4llmvuld)]
- (06/2025) Expert-in-the-Loop Systems with Cross-Domain and In-Domain Few-Shot Learning for Software Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10104)]
- (06/2025) Evaluating LLaMA 3.2 for Software Vulnerability Detection.  **`EICC 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-94855-8_3)]
- (06/2025) How Well Do Large Language Models Serve as End-to-End Secure Code Agents for Python?.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2408.10495)] [[Code](https://github.com/jianian0318/LLMSecureCode)]
- (06/2025) Detecting Code Vulnerabilities using LLMs.  **`DSN 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11068842)] [[Code](https://github.com/a24167566/LLMs-Code-Vulnerability-Detection)]
- (06/2025) LPASS: Linear Probes as Stepping Stones for Vulnerability Detection using Compressed LLMs.  **`JISA 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212625001620)]
- (06/2025) Smart Cuts: Enhance Active Learning for Vulnerability Detection by Pruning Bad Seeds.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.20444)]
- (06/2025) CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2411.17274)] [[Code](https://github.com/yikun-li/CleanVul)]
- (06/2025) Large Language Models for Multilingual Vulnerability Detection: How Far Are We?.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07503)] [[Code](https://github.com/SpanShu96/Large-Language-Model-for-Multilingual-Vulnerability-Detection/tree/main)]
- (06/2025) Large Language Models for In-File Vulnerability Localization Can Be ""Lost in the End"".  **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3715758)] [[Code](https://zenodo.org/records/14840519)]
- (06/2025) LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- (06/2025) ANVIL: Anomaly-based Vulnerability Identification without Labelled Training Data.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2408.16028)] [[Code](https://anonymous.4open.science/r/anvil)]
- (06/2025) Line-level Semantic Structure Learning for Code Vulnerability Detection.  **`Internetware 2025`** [[Paper](https://arxiv.org/abs/2407.18877)] [[Code](https://figshare.com/articles/dataset/CSLS_model_code_and_data/26391658)]
- (06/2025) SecureMind: A Framework for Benchmarking Large Language Models in Memory Bug Detection and Repair.  **`ISMM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3735950.3735954)] [[Code](https://github.com/HuantWang/SecureMind)]
- (06/2025) VuL-MCBERT: A Vulnerability Detection Method Based on Self-Supervised Contrastive Learning.  **`CAIBDA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11183103)]
- (06/2025) Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07390)] [[Code](https://github.com/Xin-Cheng-Wen/PO4Vul)]
- (06/2025) Beyond Static Pattern Matching? Rethinking Automatic Cryptographic API Misuse Detection in the Era of LLMs.  **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3728875)]
- (06/2025) An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2401.16310)] [[Code](https://zenodo.org/records/15572151)]
- (05/2025) SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.19828)] [[Code](https://github.com/basimbd/SecVulEval)]
- (05/2025) AutoAdapt: On the Application of AutoML for Parameter-Efficient Fine-Tuning of Pre-Trained Code Models.  **`TOSEM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3734867)] [[Code](https://github.com/serval-uni-lu/AutoAdapt)]
- (05/2025) Automating the Detection of Code Vulnerabilities by Analyzing GitHub Issues.  **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028308)]
- (05/2025) LLaVul: A Multimodal LLM for Interpretable Vulnerability Reasoning about Source Code.  **`ICSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11140501)]
- (05/2025) A Comparative Study of Machine Learning and Large Language Models for SQL and NoSQL Injection Vulnerability Detection.  **`SIST 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11139190)]
- (05/2025) Are Sparse Autoencoders Useful for Java Function Bug Detection?.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10375)]
- (05/2025) ♪ With a Little Help from My (LLM) Friends: Enhancing Static Analysis with LLMs to Detect Software Vulnerabilities.  **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028575)]
- (05/2025) GraphCodeBERT-Augmented Graph Attention Networks for Code Vulnerability Detection.  **`CAI 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11050748)]
- (05/2025) Leveraging Large Language Models for Command Injection Vulnerability Analysis in Python: An Empirical Study on Popular Open-Source Projects.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.15088)]
- (05/2025) Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10961)] [[Code](https://figshare.com/s/1514bc9a7aa64b46d94e)]
- (05/2025) Adversarial Training for Robustness Enhancement in LLM-Based Code Vulnerability Detection.  **`CISCE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11065803)]
- (05/2025) Learning to Focus: Context Extraction for Efficient Code Vulnerability Detection with Language Models.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.17460)]
- (05/2025) An Automated Code Review Framework Based on BERT and Qianwen Large Model.  **`CCAI 2025`** [[Paper](https://ieeexplore.ieee.org/document/11189422)]
- (04/2025) Human-Understandable Explanation for Software Vulnerability Prediction.  **`JSS 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0164121225001232)] [[Code](https://github.com/quy-ng/human-xai-software-vulnerability-prediction)]
- (04/2025) A Software Vulnerability Detection Model Combined with Graph Simplification.  **`AIBDF 2025`** [[Paper](https://dl.acm.org/doi/full/10.1145/3718491.3718525)]
- (04/2025) Case Study: Fine-tuning Small Language Models for Accurate and Private CWE Detection in Python Code.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.16584)] [[Code](https://huggingface.co/floxihunter/codegen-mono-CWEdetect)] [[Code](https://huggingface.co/datasets/floxihunter/synthetic_python_cwe)]
- (04/2025) Vulnerability Detection with Code Language Models: How Far are We?.  **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029911)] [[Code](https://github.com/DLVulDet/PrimeVul)]
- (04/2025) Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.13474)] [[Code](https://anonymous.4open.science/r/CORRECT/README.md)]
- (04/2025) IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities.  **`ICLR 2025`** [[Paper](https://arxiv.org/abs/2405.17238)] [[Code](https://github.com/iris-sast/iris)]
- (04/2025) Trace Gadgets: Minimizing Code Context for Machine Learning-Based Vulnerability Prediction.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.13676)]
- (04/2025) An Ensemble Transformer Approach with Cross-Attention for Automated Code Security Vulnerability Detection and Documentation.  **`ISDFS 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11012039)]
- (04/2025) Metamorphic-Based Many-Objective Distillation of LLMs for Code-Related Tasks.  **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11029766)] [[Code](https://zenodo.org/records/14857610)]
- (04/2025) XGV-BERT: Leveraging Contextualized Language Model and Graph Neural Network for Efficient Software Vulnerability Detection.  **`The Journal of Supercomputing 2025`** [[Paper](https://link.springer.com/article/10.1007/s11227-025-07198-7)]
- (04/2025) Leveraging Multi-Task Learning to Improve the Detection of SATD and Vulnerability.  **`ICPC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11025930)] [[Code](https://github.com/moritzmock/multitask-vulberability-detection)]
- (04/2025) Closing the Gap: A User Study on the Real-world Usefulness of AI-powered Vulnerability Detection \& Repair in the IDE.  **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029760)] [[Code](https://figshare.com/articles/dataset/Closing_the_Gap_A_User_Study_on_the_Real-world_Usefulness_of_AI-powered_Vulnerability_Detection_Repair_in_the_IDE/26367139?file=52478936)]
- (04/2025) R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.04699)] [[Code](https://github.com/martin-wey/R2Vul)]
- (04/2025) Context-Enhanced Vulnerability Detection Based on Large Language Models.  **`TOSEM 2025`** [[Paper](https://arxiv.org/abs/2504.16877)] [[Code](https://github.com/DoeSEResearch/PacVD)]
- (04/2025) SSRFSeek: An LLM-based Static Analysis Framework for Detecting SSRF Vulnerabilities in PHP Applications.  **`AINIT 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11035424)]
- (03/2025) CASTLE: Benchmarking Dataset for Static Code Analyzers and LLMs towards CWE Detection.  **`TASE 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-98208-8_15)] [[Code](https://github.com/CASTLE-Benchmark)]
- (03/2025) SecureFalcon: Are We There Yet in Automated Software Vulnerability Detection With LLMs?.  **`TSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10910240)]
- ✨ (03/2025) Impact of Identifier Normalization on Vulnerability  Detection Techniques.  **`SANER 2025`** [[Paper](https://ieeexplore.ieee.org/document/11272061)] [[Code](https://github.com/tuhh-softsec/Impact-of-Identifier-Normalization-on-Vulnerability-Detection-Techniques)]
- (03/2025) Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities.  **`ICST 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10988968)] [[Code](https://github.com/seal-research/secvul-llm-study/)]
- (03/2025) Assessing the Effectiveness of LLMs in Android Application Vulnerability Analysis.  **`ADIoT 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-85593-1_9)]
- (03/2025) Steering Large Language Models for Vulnerability Detection.  **`ICASSP 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10887736)]
- (03/2025) HALURust: Exploiting Hallucinations of Large Language Models to Detect Vulnerabilities in Rust.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.10793)]
- (03/2025) You Only Train Once: A Flexible Training Framework for Code Vulnerability Detection Driven by Vul-Vector.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10988)]
- (03/2025) Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.01449)] [[Code](https://github.com/soarsmu/SVD-Bench)]
- (03/2025) Reasoning with LLMs for Zero-Shot Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.17885)] [[Code](https://github.com/Erroristotle/VulnSage)]
- (02/2025) EFVD: A Framework of Source Code Vulnerability Detection via Fusion of Enhanced Graph Representation Learning and Pre-trained Transformer-Based Model.  **`CNSSE 2025`** [[Paper](https://dl.acm.org/doi/full/10.1145/3732365.3732421)]
- (02/2025) Fine-Tuning Transformer LLMs for Detecting SQL Injection and XSS Vulnerabilities.  **`ICAIIC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10920868)]
- (02/2025) Finetuning Large Language Models for Vulnerability Detection.  **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908394)] [[Code](https://github.com/rmusab/vul-llm-finetune)]
- (02/2025) Harnessing Large Language Models for Software Vulnerability Detection: A Comprehensive Benchmarking Study.  **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10879492)]
- (02/2025) Manual Prompt Engineering is Not Dead: A Case Study on Large Language Models for Code Vulnerability Detection with DSPy.  **`CDMA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908746)]
- (02/2025) AIDetectVul: Software Vulnerability Detection Method Based on Feature Fusion of Pre-trained Models.  **`ICCECE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10985370)]
- (01/2025) DMVL4AVD: A Deep Multi-View Learning Model for Automated Vulnerability Detection.  **`Neural Comput. Appl. 2025`** [[Paper](https://link.springer.com/article/10.1007/s00521-024-10892-x)] [[Code](https://drive.google.com/file/d/1-qWqmRuBi8kRAAE2yiG6JNiY8vLYxXlz/view)]
- (01/2025) Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2412.14841)]
- (01/2025) CGP-Tuning: Structure-Aware Soft Prompt Tuning for Code Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.04510)]
- (01/2025) Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2412.18260)] [[Code](https://github.com/SakiRinn/LLM4CVD)] [[Code](https://huggingface.co/datasets/xuefen/VulResource)]
- (01/2025) Sink Vulnerability Type Prediction Using Small Language Model (SLM).  **`IC3ECSBHI 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10991300)]
- (01/2025) To Err is Machine: Vulnerability Detection Challenges LLM Reasoning.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- (01/2025) Streamlining Security Vulnerability Triage with Large Language Models.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.18908)] [[Code](https://zenodo.org/records/14776104)]
- (01/2025) A Vulnerability Detection Framework Based on Graph Decomposition Fusion and Augmented Abstract Syntax Tree.  **`BDICN 2025`** [[Paper](https://dl.acm.org/doi/full/10.1145/3727353.3727471)]

### 2024
- (12/2024) Vulnerability Detection in Popular Programming Languages with Language Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.15905)] [[Code](https://github.com/syafiq/llm_vd)]
- (12/2024) On the Compression of Language Models for Code: An Empirical Study on CodeBERT.  **`SANER 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10992473)] [[Code](https://zenodo.org/records/14357478)]
- (12/2024) LLM-Based Approach for Buffer Overflow Detection in Source Code.  **`CIT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11021816)]
- (12/2024) A Source Code Vulnerability Detection Method Based on Positive-Unlabeled Learning.  **`RICAI 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10911761)]
- (12/2024) Evaluating Large Language Models in Vulnerability Detection Under Variable Context Windows.  **`ICMLA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10903489)]
- (12/2024) EnStack: An Ensemble Stacking Framework of Large Language Models for Enhanced Vulnerability Detection in Source Code.  **`BigData 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10825609)]
- (12/2024) Software Vulnerability Detection Using LLM: Does Additional Information Help?.  **`ACSAC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10917361)] [[Code](https://github.com/research7485/vulnerability_detection)]
- (12/2024) Enhancing Source Code Vulnerability Detection Using Flattened Code Graph Structures.  **`ICFTIC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10913325)]
- (12/2024) SQL Injection Vulnerability Detection Based on Pissa-Tuned Llama 3 Large Language Model.  **`ICFTIC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10912886)]
- (12/2024) A Method of SQL Injection Attack Detection Based on Large Language Models.  **`CNTEIE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10987904)]
- (12/2024) MVD: A Multi-Lingual Software Vulnerability Detection Framework.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.06166)] [[Code](https://figshare.com/s/10ec70108294a225f391)]
- (12/2024) Python Source Code Vulnerability Detection Based on CodeBERT Language Model.  **`ACAI 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10899694)]
- (11/2024) RealVul: Can We Detect Vulnerabilities in Web Applications with LLM?.  **`EMNLP 2024`** [[Paper](https://arxiv.org/abs/2410.07573)]
- (11/2024) StagedVulBERT: Multigranular Vulnerability Detection With a Novel Pretrained Code Model.  **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10746847)] [[Code](https://github.com/YuanJiangGit/StagedVulBERT)]
- (11/2024) Applying Contrastive Learning to Code Vulnerability Type Classification.  **`EMNLP 2024`** [[Paper](https://aclanthology.org/2024.emnlp-main.666/)]
- (11/2024) Boosting Cybersecurity Vulnerability Scanning based on LLM-supported Static Application Security Testing.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.15735)]
- (11/2024) Enhancing Vulnerability Detection Efficiency: An Exploration of Light-weight LLMs with Hybrid Code Features.  **`JISA 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212624002278)] [[Code](https://github.com/JNL-28/Enhancing-Vulnerability-Detection-Efficiency)]
- (11/2024) Research on the LLM-Driven Vulnerability Detection System Using LProtector.  **`ICDSCA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10859408)]
- (11/2024) Enhanced LLM-Based Framework for Predicting Null Pointer Dereference in Source Code.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.00216)]
- (10/2024) Vulnerability Prediction using Pre-trained Models: An Empirical Evaluation.  **`MASCOTS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10786510)] [[Code](https://sites.google.com/view/vpllm/)]
- (10/2024) Fine-Tuning Pre-trained Model with Optimizable Prompt Learning for Code Vulnerability Detection.  **`ISSRE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10771498)] [[Code](https://github.com/Exclusisve-V/PromptVulnerabilityDetection)]
- (10/2024) Improving Long-Tail Vulnerability Detection Through Data Augmentation Based on Large Language Models.  **`ICSME 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10795073)] [[Code](https://github.com/LuckyDengXiao/LERT)]
- (10/2024) Exploring AI for Vulnerability Detection and Repair.  **`CARS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10778769)]
- (10/2024) DetectBERT: Code Vulnerability Detection.  **`GCCIT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10862235)]
- (10/2024) VULREM: Fine-Tuned BERT-Based Source-Code Potential Vulnerability Scanning System to Mitigate Attacks in Web Applications.  **`Applied Sciences 2024`** [[Paper](https://www.mdpi.com/2076-3417/14/21/9697)]
- (10/2024) A Qualitative Study on Using ChatGPT for Software Security: Perception vs. Practicality.  **`TPS-ISA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10835695)] [[Code](https://figshare.com/articles/dataset/Reproduction_package_for_paper_A_Qualitative_Study_on_Using_ChatGPT_for_Software_Security_Perception_vs_Practicality_/24452365?file=48008890)]
- (10/2024) A Source Code Vulnerability Detection Method Based on Adaptive Graph Neural Networks.  **`ASE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10765114)]
- (10/2024) Vul-LMGNNs: Fusing Language Models and Online-distilled Graph Neural Networks for Code Vulnerability Detection.  **`Information Fusion 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253524005268)] [[Code](https://github.com/Vul-LMGNN/vul-LMGGNN)]
- (10/2024) SecureQwen: Leveraging LLMs for Vulnerability Detection in Python Codebases.  **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824004565)]
- (10/2024) VulnerAI: GPT Based Web Application Vulnerability Detection.  **`ICAMAC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10828788)]
- (10/2024) DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection.  **`JSS 2024`** [[Paper](nan)] [[Code](https://github.com/Yang-Yanjing/DLAP)]
- (10/2024) Multitask-Based Evaluation of Open-Source LLM on Software Vulnerability.  **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10706805)] [[Code](https://github.com/vinci-grape/VulEmpirical)]
- (10/2024) Detecting Source Code Vulnerabilities Using Fine-Tuned Pre-Trained LLMs.  **`ICSP 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10846595)]
- (09/2024) Outside the Comfort Zone: Analysing LLM Capabilities in Software Vulnerability Detection.  **`ESORICS 2024`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70879-4_14)]
- (09/2024) Navigating (In)Security of AI-Generated Code.  **`CSR 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10679468)]
- (09/2024) Bridge and Hint: Extending Pre-trained Language Models for Long-Range Code.  **`ISSTA 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3650212.3652127)] [[Code](https://anonymous.4open.science/r/EXPO/README.md)]
- (09/2024) Can a Llama Be a Watchdog? Exploring Llama 3 and Code Llama for Static Application Security Testing.  **`CSR 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10679444)]
- (09/2024) May the Source Be with You: On ChatGPT, Cybersecurity, and Secure Coding.  **`Information 2024`** [[Paper](https://www.mdpi.com/2078-2489/15/9/572)]
- (09/2024) Enhancing Source Code Security with LLMs: Demystifying The Challenges and Generating Reliable Repairs.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.00571)]
- (09/2024) Code Vulnerability Detection: A Comparative Analysis of Emerging Large Language Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.10490)]
- (09/2024) SCALE: Constructing Structured Natural Language Comment Trees for Software Vulnerability Detection.  **`ISSTA 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3650212.3652124)] [[Code](https://github.com/Xin-Cheng-Wen/Comment4Vul)]
- (09/2024) Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.01001)] [[Code](https://figshare.com/s/5da14b0776750c6fa787)]
- (09/2024) VulnLLMEval: A Framework for Evaluating Large Language Models in Software Vulnerability Detection and Patching.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.10756)]
- (08/2024) VulDetectBench: Evaluating the Deep Capability of Vulnerability Detection with Large Language Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.07595)] [[Code](https://github.com/Sweetaroo/VulDetectBench)]
- (08/2024) Defect-Scanner: A Comparative Empirical Study on Language Model and Deep Learning Approach for Software Vulnerability Detection.  **`IJIS 2024`** [[Paper](https://link.springer.com/article/10.1007/s10207-024-00901-4)]
- (08/2024) From Generalist to Specialist: Exploring CWE-Specific Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.02329)]
- (08/2024) Large Language Models for Secure Code Assessment: A Multi-Language Empirical Study.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.06428)]
- (08/2024) Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning.  **`ACL 2024`** [[Paper](https://arxiv.org/abs/2406.03718)] [[Code](https://github.com/CGCL-codes/VulLLM)]
- (08/2024) Unintentional Security Flaws in Code: Automated Defense via Root Cause Analysis.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.00199)] [[Code](https://anonymous.4open.science/r/Threat_Detection_Modeling-BB7B/README.md)]
- (08/2024) Uncovering the Limits of Machine Learning for Automatic Vulnerability Detection.  **`USENIX Security 2024`** [[Paper](https://www.usenix.org/conference/usenixsecurity24/presentation/risse)] [[Code](https://github.com/niklasrisse/USENIX_2024)] [[Code](https://github.com/niklasrisse/VPP)]
- (08/2024) : VulSim: Leveraging Similarity of Multi-Dimensional Neighbor Embeddings for Vulnerability Detection.  **`USENIX Security 2024`** [[Paper](https://www.usenix.org/conference/usenixsecurity24/presentation/shimmi)] [[Code](https://github.com/SamihaShimmi/VulSim)]
- (07/2024) Enhancing Software Code Vulnerability Detection Using GPT-4o and Claude-3.5 Sonnet: A Study on Prompt Engineering Techniques.  **`Electronics 2024`** [[Paper](https://www.mdpi.com/2079-9292/13/13/2657)]
- (07/2024) MultiVD: A Transformer-based Multitask Approach for Software Vulnerability Detection.  **`SECRYPT 2024`** [[Paper](https://www.scitepress.org/Papers/2024/127194/127194.pdf)]
- (07/2024) DFEPT: Data Flow Embedding for Enhancing Pre-Trained Model Based Vulnerability Detection.  **`Internetware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3671016.3671388)] [[Code](https://github.com/GCVulnerability/DFEPT)]
- (07/2024) Vulnerability Classification on Source Code Using Text Mining and Deep Learning Techniques.  **`QRS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10727022)] [[Code](https://sites.google.com/view/vulnerabilityclassification/)]
- (07/2024) Exploration On Prompting LLM With Code-Specific Information For Vulnerability Detection.  **`SSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10664399)]
- (07/2024) Effectiveness of ChatGPT for Static Analysis: How Far Are We?.  **`AIware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3664646.3664777)] [[Code](https://zenodo.org/records/10828316)]
- (07/2024) Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.00197)]
- (07/2024) M2CVD: Enhancing Vulnerability Understanding through Multi-Model Collaboration for Code Vulnerability Detection.  **`TOSEM 2024`** [[Paper](https://arxiv.org/abs/2406.05940)] [[Code](https://github.com/HotFrom/M2CVD)]
- (07/2024) SCL-CVD: Supervised Contrastive Learning for Code Vulnerability Detection via GraphCodeBERT.  **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824002992)]
- (07/2024) Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.16235)]
- (06/2024) Software Vulnerability Prediction in Low-Resource Languages: An Empirical Study of CodeBERT and ChatGPT.  **`EASE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3661167.3661281)] [[Code](https://github.com/lhmtriet/LLM4Vul)]
- (06/2024) Greening Large Language Models of Code.  **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3639475.3640097)] [[Code](https://github.com/soarsmu/Avatar)]
- (06/2024) Security Vulnerability Detection with Multitask Self-Instructed Fine-Tuning of Large Language Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.05892)] [[Code](https://zenodo.org/records/11403208)]
- (06/2024) Evaluating the Impact of Conventional Code Analysis Against Large Language Models in API Vulnerability Detection.  **`EICC 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3655693.3655701)]
- (06/2024) SVulDetector: Vulnerability Detection based on Similarity using Tree-based Attention and Weighted Graph Embedding Mechanisms.  **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824002335)] [[Code](https://figshare.com/s/426156a96a83da1d38d0)]
- (05/2024) DB-CBIL: A DistilBert-Based Transformer Hybrid Model Using CNN and BiLSTM for Software Vulnerability Detection.  **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10517582)]
- (05/2024) LLM-CloudSec: Large Language Model Empowered Automatic and Deep Vulnerability Analysis for Intelligent Clouds.  **`INFOCOM 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10620804)] [[Code](https://github.com/DPCa0/LLM-CloudSec)]
- (05/2024) LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks.  **`SP 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10646663/)] [[Code](https://github.com/ai4cloudops/SecLLMHolmes)]
- (05/2024) VulD-CodeBERT: CodeBERT-Based Vulnerability Detection Model for C/C++ Code.  **`CISCE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10653337)]
- (05/2024) Large Language Model for Vulnerability Detection: Emerging Results and Future Directions.  **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3639476.3639762)] [[Code](https://github.com/soarsmu/ChatGPT-VulDetection)]
- (04/2024) VulnGPT: Enhancing Source Code Vulnerability Detection Using AutoGPT and Adaptive Supervision Strategies.  **`DCOSS-IoT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10621527)]
- (04/2024) BiT5: A Bidirectional NLP Approach for Advanced Vulnerability Detection in Codebase.  **`Procedia Computer Science 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1877050924006306)]
- (04/2024) Software Vulnerability and Functionality Assessment using Large Language Models.  **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3643787.3648036)]
- (04/2024) Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks.  **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10548173)] [[Code](https://zenodo.org/records/10140638)]
- (04/2024) Towards Causal Deep Learning for Vulnerability Detection.  **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3597503.3639170)] [[Code](https://figshare.com/s/0ffda320dcb96c249ef2?file=41801019)]
- (04/2024) ProRLearn: Boosting Prompt Tuning-based Vulnerability Detection by Reinforcement Learning.  **`ASE 2024`** [[Paper](https://link.springer.com/article/10.1007/s10515-024-00438-9)] [[Code](https://github.com/ProRLearn/ProRLearn001)]
- (04/2024) VulEval: Towards Repository-Level Evaluation of Software Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2404.15596)]
- (03/2024) Learning Defect Prediction from Unrealistic Data.  **`SANER 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10589866)] [[Code](https://zenodo.org/records/10514652)]
- (03/2024) Python Source Code Vulnerability Detection with Named Entity Recognition.  **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824001032)] [[Code](https://github.com/mmeberg/PyVulDet-NER)]
- (03/2024) Making Vulnerability Prediction more Practical: Prediction, Categorization, and Localization.  **`IST 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0950584924000636)] [[Code](https://github.com/liucyy/VulPCL)]
- (03/2024) GRACE: Empowering LLM-based Software Vulnerability Detection with Graph Structure and In-Context Learning.  **`JSS 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0164121224000748)] [[Code](https://github.com/P-E-Vul/GRACE)]
- (02/2024) A Preliminary Study on Using Large Language Models in Software Pentesting.  **`NDSS 2024`** [[Paper](https://arxiv.org/abs/2401.17459)]
- (02/2024) TRACED: Execution-aware Pre-training for Source Code.  **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3597503.3608140)] [[Code](https://github.com/ARiSE-Lab/TRACED_ICSE_24)]
- (02/2024) LLbezpeky: Leveraging Large Language Models for Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.01269)]
- (02/2024) Chain-of-Thought Prompting of Large Language Models for Discovering and Fixing Software Vulnerabilities.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2402.17230)]
- (01/2024) Your Instructions Are Not Always Helpful: Assessing the Efficacy of Instruction Fine-tuning for Software Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.07466)]

### 2023
- (12/2023) Joint Geometrical and Statistical Domain Adaptation for Cross-domain  Code Vulnerability Detection.  **`EMNLP 2023`** [[Paper](https://aclanthology.org/2023.emnlp-main.788/)]
- (12/2023) ChatGPT for Vulnerability Detection, Classification, and Repair: How Far Are We?.  **`APSEC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10479409)] [[Code](https://github.com/awsm-research/ChatGPT4Vul)]
- (12/2023) Code Defect Detection Method Based on BERT and Ensemble.  **`ICCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10507306)]
- (12/2023) Assessing the Effectiveness of Vulnerability Detection via Prompt Tuning: An Empirical Study.  **`APSEC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10479384)] [[Code](https://github.com/P-E-Vul/prompt-empircial-vulnerability)]
- (12/2023) Enhancing Code Security Through Open-source Large Language Models: A Comparative Study.  **`FPS 2023`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-57537-2_15)]
- (12/2023) Optimizing Pre-trained Language Models for Efficient Vulnerability Detection in Code Snippets.  **`ICCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10507456)]
- (12/2023) Exploring the Limits of ChatGPT in Software Security Applications.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2312.05275)]
- (11/2023) How To Get Better Embeddings with Code Pre-trained Models? An Empirical Study.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2311.08066)]
- (11/2023) AIBugHunter: A Practical Tool for Predicting, Classifying and Repairing Software Vulnerabilities.  **`EMSE 2023`** [[Paper](https://link.springer.com/article/10.1007/s10664-023-10346-3)] [[Code](https://github.com/awsm-research/AIBugHunter)]
- (11/2023) The EarlyBIRD Catches the Bug: On Exploiting Early Layers of Encoder Models for More Efficient Code Classification.  **`ESEC/FSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3611643.3616304)] [[Code](https://zenodo.org/records/10499843)]
- (11/2023) Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation.  **`ESEC/FSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3611643.3616358)] [[Code](https://github.com/jacknichao/SVulD)]
- (11/2023) Do Language Models Learn Semantics of Code? A Case Study in Vulnerability Detection.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2311.04109)] [[Code](https://figshare.com/s/4a16a528d6874aad51a0)]
- (11/2023) Software Vulnerabilities Detection Based on a Pre-trained Language Model.  **`TrustCom 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10538979)]
- (10/2023) DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection.  **`RAID 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3607199.3607242)] [[Code](https://github.com/wagner-group/diversevul)]
- (10/2023) PTLVD:Program Slicing and Transformer-based Line-level Vulnerability Detection System.  **`SCAM 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10356694)] [[Code](https://github.com/chenshixu/PTLVD)]
- (10/2023) Software Vulnerability Detection using Large Language Models.  **`ISSRE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10301302)]
- (10/2023) Enhancing Large Language Models for Secure Code Generation: A Dataset-driven Study on Vulnerability Mitigation.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2310.16263)]
- (09/2023) Function-Level Vulnerability Detection Through Fusing Multi-Modal Knowledge.  **`ASE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10298584)] [[Code](https://github.com/jacknichao/MVulD)]
- (09/2023) DefectHunter: A Novel LLM-Driven Boosted-Conformer-based Code Vulnerability Detection Mechanism.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2309.15324)] [[Code](https://github.com/WJ-8/DefectHunter)]
- (09/2023) When Less is Enough: Positive and Unlabeled Learning Model for Vulnerability Detection.  **`ASE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10298363)] [[Code](https://github.com/PILOT-VD-2023/PILOT)]
- (08/2023) Using ChatGPT as a Static Application Security Testing Tool.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.14434)] [[Code](https://github.com/abakhshandeh/ChatGPTasSAST)]
- (08/2023) VulExplainer: A Transformer-Based Hierarchical  Distillation for Explaining Vulnerability Types.  **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10220166)] [[Code](https://github.com/awsm-research/VulExplainer)]
- (08/2023) Software Vulnerability Detection with GPT and In-Context Learning.  **`DSC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10381286)]
- (08/2023) Can Large Language Models Find And Fix Vulnerable Software?.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.10345)]
- (07/2023) Leveraging Deep Learning Models for Cross-function Null Pointer Risks Detection.  **`AITest 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10229470)]
- (07/2023) An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph.  **`EuroS&P 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10190505)] [[Code](https://github.com/pial08/SemVulDet)]
- (07/2023) VulDetect: A novel technique for detecting software vulnerabilities using Language Models.  **`CSR 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10224924)]
- (07/2023) An Enhanced Vulnerability Detection in Software Using a Heterogeneous Encoding Ensemble.  **`ISCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10217978)]
- (06/2023) New Tricks to Old Codes: Can AI Chatbots Replace Static Code Analysis Tools?.  **`EICC 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3590777.3590780)] [[Code](https://github.com/New-Tricks-to-Old-Codes/Replace-Static-Analysis-Tools)]
- (06/2023) Vulnerability Detection by Learning From Syntax-Based Execution Paths of Code.  **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10153647)] [[Code](https://zenodo.org/records/7123322)]
- (05/2023) An Empirical Study of Deep Learning Models for Vulnerability Detection.  **`ICSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10172583)] [[Code](https://figshare.com/articles/dataset/An_Empirical_Study_of_Deep_Learning_Models_for_Vulnerability_Detection/20791240?file=39183863)]
- (05/2023) Transformer-based Vulnerability Detection in Code at EditTime: Zero-shot, Few-shot, or Fine-tuning?.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2306.01754)]
- (05/2023) Keeping Pace with Ever-Increasing Data: Towards Continual Learning of Code Intelligence Models.  **`ICSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10172346)] [[Code](https://github.com/ReliableCoding/REPEAT)]
- (05/2023) Detecting Vulnerabilities in IoT Software: New Hybrid Model and Comprehensive Data Analysis.  **`JISA 2023`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212623000510)]
- (05/2023) VulDefend: A Novel Technique based on Pattern-exploiting Training for Detecting Software Vulnerabilities Using Language Models.  **`JEEIT 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10185860)]
- (04/2023) Evaluation of ChatGPT Model for Vulnerability Detection.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2304.07232)]

### 2022
- (12/2022) BBVD: A BERT-based Method for Vulnerability Detection.  **`IJACSA 2022`** [[Paper](https://www.proquest.com/docview/2770373789?pq-origsite=gscholar&fromopenview=true&sourcetype=Scholarly%20Journals)]
- (12/2022) Exploring Transformers for Multi-Label Classification of Java Vulnerabilities.  **`QRS 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10062434)] [[Code](https://github.com/TQRG/VDET-for-Java)]
- (12/2022) Transformer-Based Language Models for Software Vulnerability Detection.  **`ACSAC 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3564625.3567985)] [[Code](https://bitbucket.csiro.au/users/jan087/repos/acsac-2022-submission/browse)]
- (12/2022) PATVD: Vulnerability Detection Based on Pre-training Techniques and Adversarial Training.  **`SmartWorld/UIC/ScalCom/DigitalTwin/PriComp/Meta 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10189687/)]
- (11/2022) Multi-view Pre-trained Model for Code Vulnerability Identification.  **`WASA 2022`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19211-1_11)]
- (11/2022) Distilled and Contextualized Neural Models Benchmarked for Vulnerable Function Detection.  **`Mathematics 2022`** [[Paper](https://www.mdpi.com/2227-7390/10/23/4482)]
- (11/2022) BERT-Based Vulnerability Type Identification with Effective Program Representation.  **`WASA 2022`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19208-1_23#citeas)]
- (10/2022) VulDeBERT: A Vulnerability Detection System Using BERT.  **`ISSRE 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9985089)] [[Code](https://github.com/SKKU-SecLab/VulDeBERT)]
- (07/2022) VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection.  **`IJCNN 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9892280)] [[Code](https://github.com/ICL-ml4csec/VulBERTa)]
- (06/2022) Cyber Security Vulnerability Detection Using Natural Language Processing.  **`AIIoT 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9817336)]
- (05/2022) LineVul: A Transformer-based Line-level Vulnerability Prediction.  **`MSR 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3524842.3528452)] [[Code](https://github.com/awsm-research/LineVul)]
- (05/2022) LineVD: Statement-level Vulnerability Detection using Graph Neural Networks.  **`MSR 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3524842.3527949)] [[Code](https://github.com/davidhin/linevd)]
- (03/2022) Intelligent Detection of Vulnerable Functions in Software through Neural Embedding-based Code Analysis.  **`IJNM 2022`** [[Paper](https://onlinelibrary.wiley.com/doi/full/10.1002/nem.2198)] [[Code](https://cybercodeintelligence.github.io/CyberCI/)]
- (01/2022) Deep Neural Embedding for Software Vulnerability Discovery: Comparison and Optimization.  **`Security and Communication Networks 2022`** [[Paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/5203217)] [[Code](https://cybercodeintelligence.github.io/CyberCI/)]

### 2021
- (12/2021) Automated Software Vulnerability Detection via Pre-trained Context Encoder and Self Attention.  **`ICDF2C 2021`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-06365-7_15)]
- (11/2021) Detecting Integer Overflow Errors in Java Source Code via Machine Learning.  **`ICTAI 2021`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9643278)]
- (06/2021) Unified Pre-training for Program Understanding and Generation.  **`NAACL 2021`** [[Paper](https://par.nsf.gov/servlets/purl/10336701)] [[Code](https://github.com/wasiahmad/PLBART)]
- (05/2021) Security Vulnerability Detection Using Deep Learning Natural Language Processing.  **`INFOCOM 2021`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9484500)]

### 2020
- (06/2020) Exploring Software Naturalness through Neural Language Models.  **`arXiv 2020`** [[Paper](https://arxiv.org/abs/2006.12641)]




## Datasets

- SARD. [[Repo](https://samate.nist.gov/SARD)]
- Juliet C/C++. [[Repo](https://samate.nist.gov/SARD/test-suites/112)]
- Juliet Java. [[Repo](https://samate.nist.gov/SARD/test-suites/111)]
- VulDeePecker.  **`NDSS`** [[Paper](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-2_Li_paper.pdf)] [[Repo](https://github.com/CGCL-codes/VulDeePecker)]
- Draper.  **`ICMLA`** [[Paper](https://ieeexplore.ieee.org/document/8614145)] [[Repo](https://osf.io/d45bw/)]
- Devign.  **`NeurIPS`** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html)] [[Repo](https://github.com/epicosy/devign)]
- Big-Vul.  **`MSR`** [[Paper](https://dl.acm.org/doi/10.1145/3379597.3387501)] [[Repo](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset)]
- D2A.  **`ICSE-SEIP`** [[Paper](https://ieeexplore.ieee.org/document/9402126)] [[Repo](https://github.com/IBM/D2A)]
- Reveal.  **`TSE`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9448435)] [[Repo](https://github.com/VulDetProject/ReVeal)]
- CVEfixes.  **`PROMISE`** [[Paper](https://dl.acm.org/doi/10.1145/3475960.3475985)] [[Repo](https://zenodo.org/records/13118970)]
- CrossVul.  **`ESEC/FSE`** [[Paper](https://dl.acm.org/doi/10.1145/3468264.3473122)] [[Repo](https://zenodo.org/records/4734050)]
- SecurityEval.  **`MSR4P&S`** [[Paper](https://dl.acm.org/doi/10.1145/3549035.3561184)] [[Repo](https://github.com/s2e-lab/SecurityEval)]
- DiverseVul.  **`RAID`** [[Paper](https://dl.acm.org/doi/10.1145/3607199.3607242)] [[Repo](https://github.com/wagner-group/diversevul)]
- SVEN.  **`CCS`** [[Paper](https://dl.acm.org/doi/10.1145/3576915.3623175)] [[Repo](https://github.com/eth-sri/sven)]
- FormAI.  **`PROMISE`** [[Paper](https://dl.acm.org/doi/10.1145/3617555.3617874)] [[Repo](https://github.com/FormAI-Dataset/FormAI-dataset)]
- ReposVul.  **`ICSE-Companion`** [[Paper](https://dl.acm.org/doi/10.1145/3639478.3647634)] [[Repo](https://github.com/Eshe0922/ReposVul)]
- PrimeVul.  **`arXiv`** [[Paper](https://arxiv.org/abs/2403.18624)] [[Repo](https://github.com/DLVulDet/PrimeVul)]
- PairVul.  **`arXiv`** [[Paper](https://arxiv.org/abs/2406.11147)] [[Repo](https://github.com/KnowledgeRAG4LLMVulD/KnowledgeRAG4LLMVulD/tree/main/dataset)]
- MegaVul.  **`MSR`** [[Paper](https://dl.acm.org/doi/10.1145/3643991.3644886)] [[Repo](https://github.com/Icyrockton/MegaVul)]
- CleanVul.  **`arXiv`** [[Paper](https://arxiv.org/abs/2411.17274)] [[Repo](https://github.com/yikun-li/CleanVul)]



## Contribution

If you want to suggest additions to the list of studies or datasets, please open a pull request or submit an issue. 


## License

- 🧠 Code & scripts (`*.py`, `*.ipynb`, etc.): Licensed under the [MIT License](LICENSE).
- 📚 Taxonomy, markdown outputs and lists: Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).


