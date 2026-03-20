# LLM4SVD TAXONOMY 🗂️

We categorize existing LLM4SVD approaches according to detection task, input representation, system architecture, and technique. The presented [taxonomy](https://github.com/hs-esslingen-it-security/Awesome-LLM4SVD/tree/main/taxonomy/taxonomy.xlsx) allows for meaningful comparison and benchmarking of studies. <br><br>


![Taxonomy of LLM-based vulnerability detection studies.](taxonomy.png)

## Papers by Taxonomy - Navigation

We list papers for selected categories below.

- [Generation (F2)](#generation-f2)
  - [F2.1 Description](#f21-description)
  - [F2.2 Reasoning](#f22-reasoning)
  - [F2.3 Report](#f23-report)
- [Auxiliary Information (I2)](#auxiliary-information-i2)
  - [I2.1 Vulnerability Information](#i21-vulnerability-information)
  - [I2.2 Semantic Artifacts](#i22-semantic-artifacts)
  - [I2.3 Execution Artifacts](#i23-execution-artifacts)
  - [I2.4 Tool Output](#i24-tool-output)
- [Hybrid (S2)](#hybrid-s2)
  - [S2.1 RNN](#s21-rnn)
  - [S2.2 CNN](#s22-cnn)
  - [S2.3 GNN](#s23-gnn)
  - [S2.4 Other](#s24-other)
- [Technique (T)](#technique-t)
  - [T1 Feature Extraction](#t1-feature-extraction)
    - [Feature Extraction](#feature-extraction)
  - [T2 Adaptation](#t2-adaptation)
    - [T2.1 Prompt Engineering](#t21-prompt-engineering)
      - [Zero-Shot](#zero-shot)
      - [In-Context](#in-context)
      - [Few-Shot](#few-shot)
      - [RAG](#rag)
      - [CoT](#cot)
    - [T2.2 Training](#t22-training)
      - [T2.2.1 Pre-Training](#t221-pre-training)
        - [Pre-Training](#pre-training)
      - [T2.2.2 Fine-Tuning](#t222-fine-tuning)
        - [T2.2.2.1 Full-Parameter Fine-Tuning](#t2221-full-parameter-fine-tuning)
          - [Full-Parameter Fine-Tuning](#full-parameter-fine-tuning)
          - [Instruction-Tuning](#instruction-tuning)
        - [T2.2.2.2 Parameter-Efficient Fine-Tuning (PEFT)](#t2222-parameter-efficient-fine-tuning-peft)
          - [T2.2.2.2.1 Selective](#t22221-selective)
            - [Selective](#selective)
          - [T2.2.2.2.2 Additive](#t22222-additive)
            - [Adapter-Tuning](#adapter-tuning)
            - [Prompt-Tuning](#prompt-tuning)
            - [Additive-Other](#additive-other)
          - [T2.2.2.2.3 Re-parameterized](#t22223-re-parameterized)
            - [Low-Rank Decomposition](#low-rank-decomposition)
            - [LoRA Derivates](#lora-derivates)
    - [T2.3 Learning Paradigms](#t23-learning-paradigms)
      - [Contrastive Learning](#contrastive-learning)
      - [Causal Learning](#causal-learning)
      - [Multi-Task Learning](#multi-task-learning)
      - [Knowledge Distillation](#knowledge-distillation)
      - [Continual Learning](#continual-learning)
      - [Reinforcement Learning](#reinforcement-learning)
      - [Other Data-Centric](#other-data-centric)
  - [T3 Orchestration](#t3-orchestration)
    - [Multi-Step](#multi-step)
    - [Verification](#verification)
    - [Agentic](#agentic)
    - [Ensemble](#ensemble)
    - [Controller](#controller)

<hr>

## Papers by Taxonomy

<a name="generation-f2"></a>
## Generation (F2)
<a name="f21-description"></a>
### F2.1 Description
- An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph. **`EuroS&P 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10190505)] [[Code](https://github.com/pial08/SemVulDet)]
- VulExplainer: A Transformer-Based Hierarchical  Distillation for Explaining Vulnerability Types. **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10220166)] [[Code](https://github.com/awsm-research/VulExplainer)]
- Software Vulnerability and Functionality Assessment using Large Language Models. **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3643787.3648036)]
- Evaluating the Impact of Conventional Code Analysis Against Large Language Models in API Vulnerability Detection. **`EICC 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3655693.3655701)]
- Effectiveness of ChatGPT for Static Analysis: How Far Are We?. **`AIware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3664646.3664777)] [[Code](https://zenodo.org/records/10828316)]
- M2CVD: Enhancing Vulnerability Understanding through Multi-Model Collaboration for Code Vulnerability Detection. **`TOSEM 2024`** [[Paper](https://arxiv.org/abs/2406.05940)] [[Code](https://github.com/HotFrom/M2CVD)]
- Harnessing Large Language Models for Software Vulnerability Detection: A Comprehensive Benchmarking Study. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10879492)]
- Closing the Gap: A User Study on the Real-world Usefulness of AI-powered Vulnerability Detection \& Repair in the IDE. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029760)] [[Code](https://figshare.com/articles/dataset/Closing_the_Gap_A_User_Study_on_the_Real-world_Usefulness_of_AI-powered_Vulnerability_Detection_Repair_in_the_IDE/26367139?file=52478936)]
- Large Language Models for Multilingual Vulnerability Detection: How Far Are We?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07503)] [[Code](https://github.com/SpanShu96/Large-Language-Model-for-Multilingual-Vulnerability-Detection/tree/main)]
- Large Language Models for In-File Vulnerability Localization Can Be ""Lost in the End"". **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3715758)] [[Code](https://zenodo.org/records/14840519)]
- ANVIL: Anomaly-based Vulnerability Identification without Labelled Training Data. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.16028)] [[Code](https://anonymous.4open.science/r/anvil)]
- VulnTeam: A Team Collaboration Framework for LLM-based Vulnerability Detection. **`IJCNN 2025`** [[Paper](https://ieeexplore.ieee.org/document/11229292)]
- Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories. **`Unknown 2025`** [[Paper](https://arxiv.org/abs/2503.03586)]
- Evaluating Large Language Models in Vulnerability Detection Under Variable Context Windows. **`ICMLA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10903489)]
- On the Effectiveness of Instruction-Tuning Local LLMs for Identifying Software Vulnerabilities. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.20062)]
- Assessing the Effectiveness of LLMs in Android Application Vulnerability Analysis. **`ADIoT 2024`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-85593-1_9)]
- Steering Large Language Models for Vulnerability Detection. **`ICASSP 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10887736)]
- Transformer-based Vulnerability Detection in Code at EditTime: Zero-shot, Few-shot, or Fine-tuning?. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2306.01754)]
- Automating the Detection of Code Vulnerabilities by Analyzing GitHub Issues. **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028308)]
- Exploring AI for Vulnerability Detection and Repair. **`CARS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10778769)]
- Bridging Semantics \& Structure for Software Vulnerability Detection using Hybrid Network Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.10321)] [[Code](https://zenodo.org/records/17259519)]
- Real-VulLLM: An LLM Based Assessment Framework in the Wild. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.04056)]

<a name="f22-reasoning"></a>
### F2.2 Reasoning
- Software Vulnerability Detection with GPT and In-Context Learning. **`DSC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10381286)]
- Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation. **`ESEC/FSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3611643.3616358)] [[Code](https://github.com/jacknichao/SVulD)]
- LLbezpeky: Leveraging Large Language Models for Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.01269)]
- VulDetectBench: Evaluating the Deep Capability of Vulnerability Detection with Large Language Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.07595)] [[Code](https://github.com/Sweetaroo/VulDetectBench)]
- May the Source Be with You: On ChatGPT, Cybersecurity, and Secure Coding. **`Information 2024`** [[Paper](https://www.mdpi.com/2078-2489/15/9/572)]
- Enhancing Source Code Security with LLMs: Demystifying The Challenges and Generating Reliable Repairs. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.00571)]
- Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.01001)] [[Code](https://figshare.com/s/5da14b0776750c6fa787)]
- Boosting Cybersecurity Vulnerability Scanning based on LLM-supported Static Application Security Testing. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.15735)]
- Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.14841)]
- To Err is Machine: Vulnerability Detection Challenges LLM Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.13474)] [[Code](https://anonymous.4open.science/r/CORRECT/README.md)]
- IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities. **`ICLR 2024`** [[Paper](https://arxiv.org/abs/2405.17238)] [[Code](https://github.com/iris-sast/iris)]
- R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.04699)] [[Code](https://github.com/martin-wey/R2Vul)]
- Context-Enhanced Vulnerability Detection Based on Large Language Models. **`TOSEM 2025`** [[Paper](https://arxiv.org/abs/2504.16877)] [[Code](https://github.com/DoeSEResearch/PacVD)]
- Human-Understandable Explanation for Software Vulnerability Prediction. **`JSS 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0164121225001232)] [[Code](https://github.com/quy-ng/human-xai-software-vulnerability-prediction)]
- Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.11147)] [[Code](https://github.com/knowledgerag4llmvuld/knowledgerag4llmvuld)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- SecureMind: A Framework for Benchmarking Large Language Models in Memory Bug Detection and Repair. **`ISMM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3735950.3735954)] [[Code](https://github.com/HuantWang/SecureMind)]
- Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07390)] [[Code](https://github.com/Xin-Cheng-Wen/PO4Vul)]
- VulnTeam: A Team Collaboration Framework for LLM-based Vulnerability Detection. **`IJCNN 2025`** [[Paper](https://ieeexplore.ieee.org/document/11229292)]
- Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.03051)]
- CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.11599)]
- Optimizing Code Vulnerability Detection via GRPO and SFT Fine-Tuning of Compact LLMs. **`DSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11360339)]
- LLM-GUARD: Large Language Model-Based Detection and Repair of Bugs and Security Vulnerabilities in C++ and Python. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.16419)] [[Code](https://github.com/NoujoudNader/LLM-Bugs-Detection)]
- Think Broad, Act Narrow: CWE Identification with Multi-Agent Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.01451)] [[Code](https://zenodo.org/records/15871507)]
- MAVUL: Multi-Agent Vulnerability Detection via Contextual Reasoning and Interactive Refinement. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.00317)] [[Code](https://github.com/youpengl/MAVUL)]
- VulAgent: Hypothesis-Validation based Multi-Agent Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2509.11523)]
- GPTVD: vulnerability detection and analysis method based on LLM’s chain of thoughts. **`ASE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10515-025-00550-4)] [[Code](https://github.com/chenyn273/GPTVD)]
- Towards Explainable Vulnerability Detection With Large Language Models. **`TSE 2024`** [[Paper](https://arxiv.org/abs/2406.09701)]
- Compressing Large Language Models for SQL Injection Detection: A Case Study on Deep Seek-Coder and Meta-llama-3-70b-instruct. **`FRUCT 2025`** [[Paper](https://ieeexplore.ieee.org/document/11239157)]
- Specification-Guided Vulnerability Detection with Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.04014)] [[Code](https://github.com/zhuhaopku/VulInstruct-temp)]
- LLM-based Vulnerability Detection at Project Scale: An Empirical Study. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.19239)] [[Code](https://github.com/Feng-Jay/LLM4Security)]
- Beyond Function-Level Analysis: Context-Aware Reasoning for Inter-Procedural Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06751v1)] [[Code](https://github.com/yikun-li/CPRVul)]
- Evaluating and Enhancing the Vulnerability Reasoning Capabilities of Large Language Models. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06687v1)]
- VulReaD: Knowledge-Graph-guided Software Vulnerability Reasoning and Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.10787v1)] [[Code](https://anonymous.4open.science/r/Vul-ReaD)]
- From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.14012v1)] [[Code](https://github.com/youpengl/OpenVul)]
- A Systematic Study of Code Obfuscation Against LLM-based Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.16538)] [[Code](https://github.com/oxygen-hunter/SoK-Code-Obfuscation-in-LLM-VD-arxiv)]
- VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.07533)] [[Code](https://github.com/ucsb-mlsec/VulnLLM-R)]
- Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities. **`ICST 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10988968)] [[Code](https://github.com/seal-research/secvul-llm-study/)]
- Reasoning with LLMs for Zero-Shot Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.17885)] [[Code](https://github.com/Erroristotle/VulnSage)]
- LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks. **`SP 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10646663/)] [[Code](https://github.com/ai4cloudops/SecLLMHolmes)]
- SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.19828)] [[Code](https://github.com/basimbd/SecVulEval)]
- LLaVul: A Multimodal LLM for Interpretable Vulnerability Reasoning about Source Code. **`ICSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11140501)]
- Leveraging Large Language Models for Command Injection Vulnerability Analysis in Python: An Empirical Study on Popular Open-Source Projects. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.15088)]
- A Qualitative Study on Using ChatGPT for Software Security: Perception vs. Practicality. **`TPS-ISA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10835695)] [[Code](https://figshare.com/articles/dataset/Reproduction_package_for_paper_A_Qualitative_Study_on_Using_ChatGPT_for_Software_Security_Perception_vs_Practicality_/24452365?file=48008890)]
- DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection. **`JSS 2024`** [[Paper]()] [[Code](https://github.com/Yang-Yanjing/DLAP)]
- Real-VulLLM: An LLM Based Assessment Framework in the Wild. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.04056)]
- Distilling Lightweight Language Models for C/C++ Vulnerabilities. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.06645)] [[Code](https://github.com/yangxiaoxuan123/FineSec_detect)]

<a name="f23-report"></a>
### F2.3 Report
- Can Large Language Models Find And Fix Vulnerable Software?. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.10345)]
- A Preliminary Study on Using Large Language Models in Software Pentesting. **`NDSS 2024`** [[Paper](https://arxiv.org/abs/2401.17459)]
- Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.00197)]
- Boosting Cybersecurity Vulnerability Scanning based on LLM-supported Static Application Security Testing. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.15735)]
- An Ensemble Transformer Approach with Cross-Attention for Automated Code Security Vulnerability Detection and Documentation. **`ISDFS 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11012039)]
- R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.04699)] [[Code](https://github.com/martin-wey/R2Vul)]
- SSRFSeek: An LLM-based Static Analysis Framework for Detecting SSRF Vulnerabilities in PHP Applications. **`AINIT 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11035424)]
- Human-Understandable Explanation for Software Vulnerability Prediction. **`JSS 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0164121225001232)] [[Code](https://github.com/quy-ng/human-xai-software-vulnerability-prediction)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- Beyond Static Pattern Matching? Rethinking Automatic Cryptographic API Misuse Detection in the Era of LLMs. **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3728875)]
- An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16310)] [[Code](https://zenodo.org/records/15572151)]
- MalCodeAI: Autonomous Vulnerability Detection and Remediation via Language Agnostic Code Reasoning. **`IRI 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11153184)]
- Large Language Models Versus Static Code Analysis Tools: A Systematic Benchmark for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/pdf/2508.04448)] [[Code](https://github.com/Damian0401/ProjectAnalyzer)]
- Optimizing Code Vulnerability Detection via GRPO and SFT Fine-Tuning of Compact LLMs. **`DSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11360339)]
- DeepVulHunter: Enhancing the Code Vulnerability Detection Capability of LLMs through Multi-Round Analysis. **`JIIS 2025`** [[Paper](https://link.springer.com/article/10.1007/s10844-025-00982-0)]
- iCodeReviewer: Improving Secure Code Review with Mixture of Prompts. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.12186)]
- Leveraging Intra-and Inter-References in Vulnerability Detection using Multi-Agent Collaboration Based on LLMs. **`Cluster Computing 2025`** [[Paper](https://link.springer.com/article/10.1007/s10586-025-05721-2)]
- Should We Evaluate LLM Based Security Analysis Approaches on Open Source Systems?. **`ASE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11334389/)]
- Specification-Guided Vulnerability Detection with Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.04014)] [[Code](https://github.com/zhuhaopku/VulInstruct-temp)]
- Large Language Models Cannot Reliably Detect Vulnerabilities in JavaScript: The First Systematic Benchmark and Evaluation. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.01255)] [[Code](https://github.com/SecJS-Vuln-Benchmark/SecJS-Benchmark)] [[Code](https://secjs-vuln-benchmark.github.io/SecJS-Benchmark/)]
- CASTLE: Benchmarking Dataset for Static Code Analyzers and LLMs towards CWE Detection. **`TASE 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-98208-8_15)] [[Code](https://github.com/CASTLE-Benchmark)]
- Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities. **`ICST 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10988968)] [[Code](https://github.com/seal-research/secvul-llm-study/)]
- Reasoning with LLMs for Zero-Shot Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.17885)] [[Code](https://github.com/Erroristotle/VulnSage)]
- Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10961)] [[Code](https://figshare.com/s/1514bc9a7aa64b46d94e)]
- DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection. **`JSS 2024`** [[Paper]()] [[Code](https://github.com/Yang-Yanjing/DLAP)]
- Distilling Lightweight Language Models for C/C++ Vulnerabilities. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.06645)] [[Code](https://github.com/yangxiaoxuan123/FineSec_detect)]

<a name="auxiliary-information-i2"></a>
## Auxiliary Information (I2)
<a name="i21-vulnerability-information"></a>
### I2.1 Vulnerability Information
- Using ChatGPT as a Static Application Security Testing Tool. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.14434)] [[Code](https://github.com/abakhshandeh/ChatGPTasSAST)]
- Software Vulnerability Detection with GPT and In-Context Learning. **`DSC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10381286)]
- LLbezpeky: Leveraging Large Language Models for Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.01269)]
- Chain-of-Thought Prompting of Large Language Models for Discovering and Fixing Software Vulnerabilities. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2402.17230)]
- A Preliminary Study on Using Large Language Models in Software Pentesting. **`NDSS 2024`** [[Paper](https://arxiv.org/abs/2401.17459)]
- Software Vulnerability Prediction in Low-Resource Languages: An Empirical Study of CodeBERT and ChatGPT. **`EASE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3661167.3661281)] [[Code](https://github.com/lhmtriet/LLM4Vul)]
- Exploration On Prompting LLM With Code-Specific Information For Vulnerability Detection. **`SSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10664399)]
- Effectiveness of ChatGPT for Static Analysis: How Far Are We?. **`AIware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3664646.3664777)] [[Code](https://zenodo.org/records/10828316)]
- Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.00197)]
- Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.16235)]
- Large Language Models for Secure Code Assessment: A Multi-Language Empirical Study. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.06428)]
- VulDetectBench: Evaluating the Deep Capability of Vulnerability Detection with Large Language Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.07595)] [[Code](https://github.com/Sweetaroo/VulDetectBench)]
- Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.01001)] [[Code](https://figshare.com/s/5da14b0776750c6fa787)]
- Boosting Cybersecurity Vulnerability Scanning based on LLM-supported Static Application Security Testing. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.15735)]
- Research on the LLM-Driven Vulnerability Detection System Using LProtector. **`ICDSCA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10859408)]
- Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.14841)]
- To Err is Machine: Vulnerability Detection Challenges LLM Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- Manual Prompt Engineering is Not Dead: A Case Study on Large Language Models for Code Vulnerability Detection with DSPy. **`CDMA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908746)]
- Vulnerability Detection with Code Language Models: How Far are We?. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029911)] [[Code](https://github.com/DLVulDet/PrimeVul)]
- Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.13474)] [[Code](https://anonymous.4open.science/r/CORRECT/README.md)]
- Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.11147)] [[Code](https://github.com/knowledgerag4llmvuld/knowledgerag4llmvuld)]
- Expert-in-the-Loop Systems with Cross-Domain and In-Domain Few-Shot Learning for Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10104)]
- Large Language Models for Multilingual Vulnerability Detection: How Far Are We?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07503)] [[Code](https://github.com/SpanShu96/Large-Language-Model-for-Multilingual-Vulnerability-Detection/tree/main)]
- Large Language Models for In-File Vulnerability Localization Can Be ""Lost in the End"". **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3715758)] [[Code](https://zenodo.org/records/14840519)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- SecureMind: A Framework for Benchmarking Large Language Models in Memory Bug Detection and Repair. **`ISMM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3735950.3735954)] [[Code](https://github.com/HuantWang/SecureMind)]
- Beyond Static Pattern Matching? Rethinking Automatic Cryptographic API Misuse Detection in the Era of LLMs. **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3728875)]
- An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16310)] [[Code](https://zenodo.org/records/15572151)]
- VulnTeam: A Team Collaboration Framework for LLM-based Vulnerability Detection. **`IJCNN 2025`** [[Paper](https://ieeexplore.ieee.org/document/11229292)]
- Revisiting Pre-trained Language Models for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.16887)]
- Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories. **`Unknown 2025`** [[Paper](https://arxiv.org/abs/2503.03586)]
- Large Language Models Versus Static Code Analysis Tools: A Systematic Benchmark for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/pdf/2508.04448)] [[Code](https://github.com/Damian0401/ProjectAnalyzer)]
- CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.11599)]
- Think Broad, Act Narrow: CWE Identification with Multi-Agent Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.01451)] [[Code](https://zenodo.org/records/15871507)]
- GPTVD: vulnerability detection and analysis method based on LLM’s chain of thoughts. **`ASE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10515-025-00550-4)] [[Code](https://github.com/chenyn273/GPTVD)]
- Can LLM Prompting Serve as a Proxy for Static Analysis in Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.12039)]
- DeepVulHunter: Enhancing the Code Vulnerability Detection Capability of LLMs through Multi-Round Analysis. **`JIIS 2025`** [[Paper](https://link.springer.com/article/10.1007/s10844-025-00982-0)]
- Towards Explainable Vulnerability Detection With Large Language Models. **`TSE 2024`** [[Paper](https://arxiv.org/abs/2406.09701)]
- Leveraging Intra-and Inter-References in Vulnerability Detection using Multi-Agent Collaboration Based on LLMs. **`Cluster Computing 2025`** [[Paper](https://link.springer.com/article/10.1007/s10586-025-05721-2)]
- Learning-based Models for Vulnerability Detection: An Extensive Study. **`EMSE 2024`** [[Paper](https://arxiv.org/abs/2408.07526)] [[Code](https://figshare.com/s/bde8e41890e8179fbe5f?file=41894784)]
- An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems. **`CASCON 2025`** [[Paper](https://ieeexplore.ieee.org/document/11344502)]
- A Sequential Multi-Stage Approach for Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making. **`EMNLP 2025`** [[Paper](https://aclanthology.org/2025.emnlp-main.1071/)]
- Specification-Guided Vulnerability Detection with Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.04014)] [[Code](https://github.com/zhuhaopku/VulInstruct-temp)]
- Retrieval-Augmented Few-Shot Prompting Versus Fine-Tuning for Code Vulnerability Detection. **`FLLM 2025`** [[Paper](https://ieeexplore.ieee.org/document/11391248)]
- ChatGPT for Vulnerability Detection, Classification, and Repair: How Far Are We?. **`APSEC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10479409)] [[Code](https://github.com/awsm-research/ChatGPT4Vul)]
- Software Vulnerability Detection Using LLM: Does Additional Information Help?. **`ACSAC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10917361)] [[Code](https://github.com/research7485/vulnerability_detection)]
- GRACE: Empowering LLM-based Software Vulnerability Detection with Graph Structure and In-Context Learning. **`JSS 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0164121224000748)] [[Code](https://github.com/P-E-Vul/GRACE)]
- Assessing the Effectiveness of LLMs in Android Application Vulnerability Analysis. **`ADIoT 2024`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-85593-1_9)]
- HALURust: Exploiting Hallucinations of Large Language Models to Detect Vulnerabilities in Rust. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.10793)]
- Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.01449)] [[Code](https://github.com/soarsmu/SVD-Bench)]
- Transformer-based Vulnerability Detection in Code at EditTime: Zero-shot, Few-shot, or Fine-tuning?. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2306.01754)]
- LLM-CloudSec: Large Language Model Empowered Automatic and Deep Vulnerability Analysis for Intelligent Clouds. **`INFOCOM 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10620804)] [[Code](https://github.com/DPCa0/LLM-CloudSec)]
- LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks. **`SP 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10646663/)] [[Code](https://github.com/ai4cloudops/SecLLMHolmes)]
- Large Language Model for Vulnerability Detection: Emerging Results and Future Directions. **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3639476.3639762)] [[Code](https://github.com/soarsmu/ChatGPT-VulDetection)]
- Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10961)] [[Code](https://figshare.com/s/1514bc9a7aa64b46d94e)]
- Enhancing Large Language Models for Secure Code Generation: A Dataset-driven Study on Vulnerability Mitigation. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2310.16263)]
- DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection. **`JSS 2024`** [[Paper]()] [[Code](https://github.com/Yang-Yanjing/DLAP)]
- Multitask-Based Evaluation of Open-Source LLM on Software Vulnerability. **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10706805)] [[Code](https://github.com/vinci-grape/VulEmpirical)]
- On Selecting Few-Shot Examples for LLM-based Code Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.27675)]
- Llama-Based Source Code Vulnerability Detection: Prompt Engineering vs Fine Tuning. **`ESORICS 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-032-07884-1_15)] [[Code](https://github.com/DynaSoumhaneOuchebara/Llama-based-vulnerability-detection)]
- Real-VulLLM: An LLM Based Assessment Framework in the Wild. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.04056)]

<a name="i22-semantic-artifacts"></a>
### I2.2 Semantic Artifacts
- Streamlining Security Vulnerability Triage with Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.18908)] [[Code](https://zenodo.org/records/14776104)]
- Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.11147)] [[Code](https://github.com/knowledgerag4llmvuld/knowledgerag4llmvuld)]
- Detecting Code Vulnerabilities using LLMs. **`DSN 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11068842)] [[Code](https://github.com/a24167566/LLMs-Code-Vulnerability-Detection)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- CLeVeR: Multi-modal Contrastive Learning for Vulnerability Code Representation. **`ACL 2025`** [[Paper](https://aclanthology.org/2025.findings-acl.414/)] [[Code](https://github.com/yoimiya-nlp/CLeVeR)]
- Enhancing Vulnerability Detection by Fusing Code Semantic Features with LLM-generated Explanations. **`Information Fusion 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253525005238)] [[Code](https://github.com/XUPT-SSS/FuSEVul)]
- CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.11599)]
- LLaVul: A Multimodal LLM for Interpretable Vulnerability Reasoning about Source Code. **`ICSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11140501)]

<a name="i23-execution-artifacts"></a>
### I2.3 Execution Artifacts
- VulEval: Towards Repository-Level Evaluation of Software Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2404.15596)]
- Harnessing Large Language Models for Software Vulnerability Detection: A Comprehensive Benchmarking Study. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10879492)]
- Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.13474)] [[Code](https://anonymous.4open.science/r/CORRECT/README.md)]
- IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities. **`ICLR 2024`** [[Paper](https://arxiv.org/abs/2405.17238)] [[Code](https://github.com/iris-sast/iris)]
- Context-Enhanced Vulnerability Detection Based on Large Language Models. **`TOSEM 2025`** [[Paper](https://arxiv.org/abs/2504.16877)] [[Code](https://github.com/DoeSEResearch/PacVD)]
- SSRFSeek: An LLM-based Static Analysis Framework for Detecting SSRF Vulnerabilities in PHP Applications. **`AINIT 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11035424)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- Revisiting Pre-trained Language Models for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.16887)]
- LLM-GUARD: Large Language Model-Based Detection and Repair of Bugs and Security Vulnerabilities in C++ and Python. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.16419)] [[Code](https://github.com/NoujoudNader/LLM-Bugs-Detection)]
- Think Broad, Act Narrow: CWE Identification with Multi-Agent Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.01451)] [[Code](https://zenodo.org/records/15871507)]
- Beyond Function-Level Analysis: Context-Aware Reasoning for Inter-Procedural Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06751v1)] [[Code](https://github.com/yikun-li/CPRVul)]
- From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.14012v1)] [[Code](https://github.com/youpengl/OpenVul)]
- SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.19828)] [[Code](https://github.com/basimbd/SecVulEval)]

<a name="i24-tool-output"></a>
### I2.4 Tool Output
- Using ChatGPT as a Static Application Security Testing Tool. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.14434)] [[Code](https://github.com/abakhshandeh/ChatGPTasSAST)]
- Boosting Cybersecurity Vulnerability Scanning based on LLM-supported Static Application Security Testing. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.15735)]
- VulAgent: Hypothesis-Validation based Multi-Agent Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2509.11523)]
- VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.07533)] [[Code](https://github.com/ucsb-mlsec/VulnLLM-R)]
- ♪ With a Little Help from My (LLM) Friends: Enhancing Static Analysis with LLMs to Detect Software Vulnerabilities. **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028575)]
- DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection. **`JSS 2024`** [[Paper]()] [[Code](https://github.com/Yang-Yanjing/DLAP)]
- The Richer Representation Fallacy: Are We Just Adding Noise to LLM-based Software Vulnerability Detectors?. **`ICOCO 2025`** [[Paper](https://ieeexplore.ieee.org/document/11334069)]

<a name="hybrid-s2"></a>
## Hybrid (S2)
<a name="s21-rnn"></a>
### S2.1 RNN
- Defect-Scanner: A Comparative Empirical Study on Language Model and Deep Learning Approach for Software Vulnerability Detection. **`IJIS 2024`** [[Paper](https://link.springer.com/article/10.1007/s10207-024-00901-4)]
- Leveraging Self-Paced Learning for Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.09212)] [[Code](https://figshare.com/s/bef3211194fc18fe375e)]
- Automated Software Vulnerability Detection via Pre-trained Context Encoder and Self Attention. **`ICDF2C 2021`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-06365-7_15)]
- Code Defect Detection Method Based on BERT and Ensemble. **`ICCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10507306)]
- Making Vulnerability Prediction more Practical: Prediction, Categorization, and Localization. **`IST 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0950584924000636)] [[Code](https://github.com/liucyy/VulPCL)]
- Security Vulnerability Detection Using Deep Learning Natural Language Processing. **`INFOCOM 2021`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9484500)]
- Detecting Vulnerabilities in IoT Software: New Hybrid Model and Comprehensive Data Analysis. **`JISA 2023`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212623000510)]
- DB-CBIL: A DistilBert-Based Transformer Hybrid Model Using CNN and BiLSTM for Software Vulnerability Detection. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10517582)]
- VulD-CodeBERT: CodeBERT-Based Vulnerability Detection Model for C/C++ Code. **`CISCE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10653337)]
- Fine-Tuning Pre-trained Model with Optimizable Prompt Learning for Code Vulnerability Detection. **`ISSRE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10771498)] [[Code](https://github.com/Exclusisve-V/PromptVulnerabilityDetection)]

<a name="s22-cnn"></a>
### S2.2 CNN
- VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection. **`IJCNN 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9892280)] [[Code](https://github.com/ICL-ml4csec/VulBERTa)]
- Vulnerability Detection by Learning From Syntax-Based Execution Paths of Code. **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10153647)] [[Code](https://zenodo.org/records/7123322)]
- DefectHunter: A Novel LLM-Driven Boosted-Conformer-based Code Vulnerability Detection Mechanism. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2309.15324)] [[Code](https://github.com/WJ-8/DefectHunter)]
- Defect-Scanner: A Comparative Empirical Study on Language Model and Deep Learning Approach for Software Vulnerability Detection. **`IJIS 2024`** [[Paper](https://link.springer.com/article/10.1007/s10207-024-00901-4)]
- A Vulnerability Detection Framework Based on Graph Decomposition Fusion and Augmented Abstract Syntax Tree. **`BDICN 2025`** [[Paper](https://dl.acm.org/doi/full/10.1145/3727353.3727471)]
- DMVL4AVD: A Deep Multi-View Learning Model for Automated Vulnerability Detection. **`Neural Comput. Appl. 2025`** [[Paper](https://link.springer.com/article/10.1007/s00521-024-10892-x)] [[Code](https://drive.google.com/file/d/1-qWqmRuBi8kRAAE2yiG6JNiY8vLYxXlz/view)]
- A Software Vulnerability Detection Model Combined with Graph Simplification. **`AIBDF 2025`** [[Paper](https://dl.acm.org/doi/full/10.1145/3718491.3718525)]
- Code Defect Detection Method Based on BERT and Ensemble. **`ICCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10507306)]
- Detecting Vulnerabilities in IoT Software: New Hybrid Model and Comprehensive Data Analysis. **`JISA 2023`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212623000510)]
- DB-CBIL: A DistilBert-Based Transformer Hybrid Model Using CNN and BiLSTM for Software Vulnerability Detection. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10517582)]

<a name="s23-gnn"></a>
### S2.3 GNN
- An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph. **`EuroS&P 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10190505)] [[Code](https://github.com/pial08/SemVulDet)]
- An Enhanced Vulnerability Detection in Software Using a Heterogeneous Encoding Ensemble. **`ISCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10217978)]
- Function-Level Vulnerability Detection Through Fusing Multi-Modal Knowledge. **`ASE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10298584)] [[Code](https://github.com/jacknichao/MVulD)]
- Security Vulnerability Detection with Multitask Self-Instructed Fine-Tuning of Large Language Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.05892)] [[Code](https://zenodo.org/records/11403208)]
- SVulDetector: Vulnerability Detection based on Similarity using Tree-based Attention and Weighted Graph Embedding Mechanisms. **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824002335)] [[Code](https://figshare.com/s/426156a96a83da1d38d0)]
- DFEPT: Data Flow Embedding for Enhancing Pre-Trained Model Based Vulnerability Detection. **`Internetware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3671016.3671388)] [[Code](https://github.com/GCVulnerability/DFEPT)]
- Unintentional Security Flaws in Code: Automated Defense via Root Cause Analysis. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.00199)] [[Code](https://anonymous.4open.science/r/Threat_Detection_Modeling-BB7B/README.md)]
- DMVL4AVD: A Deep Multi-View Learning Model for Automated Vulnerability Detection. **`Neural Comput. Appl. 2025`** [[Paper](https://link.springer.com/article/10.1007/s00521-024-10892-x)] [[Code](https://drive.google.com/file/d/1-qWqmRuBi8kRAAE2yiG6JNiY8vLYxXlz/view)]
- EFVD: A Framework of Source Code Vulnerability Detection via Fusion of Enhanced Graph Representation Learning and Pre-trained Transformer-Based Model. **`CNSSE 2025`** [[Paper](https://dl.acm.org/doi/full/10.1145/3732365.3732421)]
- XGV-BERT: Leveraging Contextualized Language Model and Graph Neural Network for Efficient Software Vulnerability Detection. **`The Journal of Supercomputing 2023`** [[Paper](https://link.springer.com/article/10.1007/s11227-025-07198-7)]
- HgtJIT: Just-in-Time Vulnerability Detection Based on Heterogeneous Graph Transformer. **`TDSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11072308)]
- Multimodal Fusion for Vulnerability Detection: Integrating Sequence and Graph-Based Analysis with LLM Augmentation. **`MAPR 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11133833)]
- Code Defect Detection Method Based on BERT and Ensemble. **`ICCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10507306)]
- A Source Code Vulnerability Detection Method Based on Positive-Unlabeled Learning. **`RICAI 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10911761)]
- LineVD: Statement-level Vulnerability Detection using Graph Neural Networks. **`MSR 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3524842.3527949)] [[Code](https://github.com/davidhin/linevd)]
- GraphCodeBERT-Augmented Graph Attention Networks for Code Vulnerability Detection. **`CAI 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11050748)]
- Fine-Tuning Pre-trained Model with Optimizable Prompt Learning for Code Vulnerability Detection. **`ISSRE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10771498)] [[Code](https://github.com/Exclusisve-V/PromptVulnerabilityDetection)]
- A Source Code Vulnerability Detection Method Based on Adaptive Graph Neural Networks. **`ASE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10765114)]
- Vul-LMGNNs: Fusing Language Models and Online-distilled Graph Neural Networks for Code Vulnerability Detection. **`Information Fusion 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253524005268)] [[Code](https://github.com/Vul-LMGNN/vul-LMGGNN)]
- Bridging Semantics \& Structure for Software Vulnerability Detection using Hybrid Network Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.10321)] [[Code](https://zenodo.org/records/17259519)]
- The Richer Representation Fallacy: Are We Just Adding Noise to LLM-based Software Vulnerability Detectors?. **`ICOCO 2025`** [[Paper](https://ieeexplore.ieee.org/document/11334069)]
- Code Vulnerability Detection Method Based On PreTrained Language Model and Gating Graph Neural Network. **`CBASE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11335562)]
- CTVD: Collaborative Training of Deep Learning and Large Model for C/C++ Source Code Vulnerability Detection. **`SMC 2025`** [[Paper](https://ieeexplore.ieee.org/document/11343541)]

<a name="s24-other"></a>
### S2.4 Other
- Defect-Scanner: A Comparative Empirical Study on Language Model and Deep Learning Approach for Software Vulnerability Detection. **`IJIS 2024`** [[Paper](https://link.springer.com/article/10.1007/s10207-024-00901-4)]
- AIDetectVul: Software Vulnerability Detection Method Based on Feature Fusion of Pre-trained Models. **`ICCECE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10985370)]
- Enhancing Fine-Grained Vulnerability Detection With Reinforcement Learning. **`TSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11145224)] [[Code](https://github.com/YuanJiangGit/RLFD)]
- Are Sparse Autoencoders Useful for Java Function Bug Detection?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10375)]
- PTLVD:Program Slicing and Transformer-based Line-level Vulnerability Detection System. **`SCAM 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10356694)] [[Code](https://github.com/chenshixu/PTLVD)]

<a name="technique-t"></a>
## Technique (T)
<a name="t1-feature-extraction"></a>
### T1 Feature Extraction
<a name="feature-extraction"></a>
#### Feature Extraction
- Multi-view Pre-trained Model for Code Vulnerability Identification. **`WASA 2022`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19211-1_11)]
- Vulnerability Detection by Learning From Syntax-Based Execution Paths of Code. **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10153647)] [[Code](https://zenodo.org/records/7123322)]
- An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph. **`EuroS&P 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10190505)] [[Code](https://github.com/pial08/SemVulDet)]
- An Enhanced Vulnerability Detection in Software Using a Heterogeneous Encoding Ensemble. **`ISCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10217978)]
- Function-Level Vulnerability Detection Through Fusing Multi-Modal Knowledge. **`ASE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10298584)] [[Code](https://github.com/jacknichao/MVulD)]
- DefectHunter: A Novel LLM-Driven Boosted-Conformer-based Code Vulnerability Detection Mechanism. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2309.15324)] [[Code](https://github.com/WJ-8/DefectHunter)]
- How To Get Better Embeddings with Code Pre-trained Models? An Empirical Study. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2311.08066)]
- DFEPT: Data Flow Embedding for Enhancing Pre-Trained Model Based Vulnerability Detection. **`Internetware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3671016.3671388)] [[Code](https://github.com/GCVulnerability/DFEPT)]
- Unintentional Security Flaws in Code: Automated Defense via Root Cause Analysis. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.00199)] [[Code](https://anonymous.4open.science/r/Threat_Detection_Modeling-BB7B/README.md)]
- Defect-Scanner: A Comparative Empirical Study on Language Model and Deep Learning Approach for Software Vulnerability Detection. **`IJIS 2024`** [[Paper](https://link.springer.com/article/10.1007/s10207-024-00901-4)]
- Enhanced LLM-Based Framework for Predicting Null Pointer Dereference in Source Code. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.00216)]
- A Vulnerability Detection Framework Based on Graph Decomposition Fusion and Augmented Abstract Syntax Tree. **`BDICN 2025`** [[Paper](https://dl.acm.org/doi/full/10.1145/3727353.3727471)]
- Fine-Tuning Transformer LLMs for Detecting SQL Injection and XSS Vulnerabilities. **`ICAIIC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10920868)]
- AIDetectVul: Software Vulnerability Detection Method Based on Feature Fusion of Pre-trained Models. **`ICCECE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10985370)]
- EFVD: A Framework of Source Code Vulnerability Detection via Fusion of Enhanced Graph Representation Learning and Pre-trained Transformer-Based Model. **`CNSSE 2025`** [[Paper](https://dl.acm.org/doi/full/10.1145/3732365.3732421)]
- An Ensemble Transformer Approach with Cross-Attention for Automated Code Security Vulnerability Detection and Documentation. **`ISDFS 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11012039)]
- A Software Vulnerability Detection Model Combined with Graph Simplification. **`AIBDF 2025`** [[Paper](https://dl.acm.org/doi/full/10.1145/3718491.3718525)]
- ANVIL: Anomaly-based Vulnerability Identification without Labelled Training Data. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.16028)] [[Code](https://anonymous.4open.science/r/anvil)]
- HgtJIT: Just-in-Time Vulnerability Detection Based on Heterogeneous Graph Transformer. **`TDSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11072308)]
- Enhancing Vulnerability Detection by Fusing Code Semantic Features with LLM-generated Explanations. **`Information Fusion 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253525005238)] [[Code](https://github.com/XUPT-SSS/FuSEVul)]
- Enhancing Fine-Grained Vulnerability Detection With Reinforcement Learning. **`TSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11145224)] [[Code](https://github.com/YuanJiangGit/RLFD)]
- Multimodal Fusion for Vulnerability Detection: Integrating Sequence and Graph-Based Analysis with LLM Augmentation. **`MAPR 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11133833)]
- Transformer-Based Semantic Embeddings and Hybrid Neural Networks for Robust Software Vulnerability Detection. **`i-PACT 2025`** [[Paper](https://ieeexplore.ieee.org/document/11307989)]
- Joint Geometrical and Statistical Domain Adaptation for Cross-domain  Code Vulnerability Detection. **`EMNLP 2023`** [[Paper](https://aclanthology.org/2023.emnlp-main.788/)]
- LineVD: Statement-level Vulnerability Detection using Graph Neural Networks. **`MSR 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3524842.3527949)] [[Code](https://github.com/davidhin/linevd)]
- LLaVul: A Multimodal LLM for Interpretable Vulnerability Reasoning about Source Code. **`ICSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11140501)]
- Are Sparse Autoencoders Useful for Java Function Bug Detection?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10375)]
- GraphCodeBERT-Augmented Graph Attention Networks for Code Vulnerability Detection. **`CAI 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11050748)]
- An Automated Code Review Framework Based on BERT and Qianwen Large Model. **`CCAI 2025`** [[Paper](https://ieeexplore.ieee.org/document/11189422)]
- PTLVD:Program Slicing and Transformer-based Line-level Vulnerability Detection System. **`SCAM 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10356694)] [[Code](https://github.com/chenshixu/PTLVD)]
- DetectBERT: Code Vulnerability Detection. **`GCCIT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10862235)]
- A Source Code Vulnerability Detection Method Based on Adaptive Graph Neural Networks. **`ASE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10765114)]
- A Zero-Shot Framework for Cross-Project Vulnerability Detection in Source Code. **`EMSE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10664-025-10749-4)] [[Code](https://github.com/Radowan98/ZSVulD)]
- MulVuln: Enhancing Pre-trained LMs with Shared and Language-Specific Knowledge for Multilingual Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.04397)]
- Code Vulnerability Detection Method Based On PreTrained Language Model and Gating Graph Neural Network. **`CBASE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11335562)]

<a name="t2-adaptation"></a>
### T2 Adaptation
<a name="t21-prompt-engineering"></a>
#### T2.1 Prompt Engineering
<a name="zero-shot"></a>
##### Zero-Shot
- Evaluation of ChatGPT Model for Vulnerability Detection. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2304.07232)]
- New Tricks to Old Codes: Can AI Chatbots Replace Static Code Analysis Tools?. **`EICC 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3590777.3590780)] [[Code](https://github.com/New-Tricks-to-Old-Codes/Replace-Static-Analysis-Tools)]
- Using ChatGPT as a Static Application Security Testing Tool. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.14434)] [[Code](https://github.com/abakhshandeh/ChatGPTasSAST)]
- Can Large Language Models Find And Fix Vulnerable Software?. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.10345)]
- VulnGPT: Enhancing Source Code Vulnerability Detection Using AutoGPT and Adaptive Supervision Strategies. **`DCOSS-IoT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10621527)]
- Software Vulnerability and Functionality Assessment using Large Language Models. **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3643787.3648036)]
- Evaluating the Impact of Conventional Code Analysis Against Large Language Models in API Vulnerability Detection. **`EICC 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3655693.3655701)]
- Enhancing Software Code Vulnerability Detection Using GPT-4o and Claude-3.5 Sonnet: A Study on Prompt Engineering Techniques. **`Electronics 2024`** [[Paper](https://www.mdpi.com/2079-9292/13/13/2657)]
- Exploration On Prompting LLM With Code-Specific Information For Vulnerability Detection. **`SSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10664399)]
- M2CVD: Enhancing Vulnerability Understanding through Multi-Model Collaboration for Code Vulnerability Detection. **`TOSEM 2024`** [[Paper](https://arxiv.org/abs/2406.05940)] [[Code](https://github.com/HotFrom/M2CVD)]
- Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.16235)]
- Large Language Models for Secure Code Assessment: A Multi-Language Empirical Study. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.06428)]
- Navigating (In)Security of AI-Generated Code. **`CSR 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10679468)]
- May the Source Be with You: On ChatGPT, Cybersecurity, and Secure Coding. **`Information 2024`** [[Paper](https://www.mdpi.com/2078-2489/15/9/572)]
- VulnLLMEval: A Framework for Evaluating Large Language Models in Software Vulnerability Detection and Patching. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.10756)]
- Outside the Comfort Zone: Analysing LLM Capabilities in Software Vulnerability Detection. **`ESORICS 2024`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70879-4_14)]
- To Err is Machine: Vulnerability Detection Challenges LLM Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- Manual Prompt Engineering is Not Dead: A Case Study on Large Language Models for Code Vulnerability Detection with DSPy. **`CDMA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908746)]
- Closing the Gap: A User Study on the Real-world Usefulness of AI-powered Vulnerability Detection \& Repair in the IDE. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029760)] [[Code](https://figshare.com/articles/dataset/Closing_the_Gap_A_User_Study_on_the_Real-world_Usefulness_of_AI-powered_Vulnerability_Detection_Repair_in_the_IDE/26367139?file=52478936)]
- Expert-in-the-Loop Systems with Cross-Domain and In-Domain Few-Shot Learning for Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10104)]
- How Well Do Large Language Models Serve as End-to-End Secure Code Agents for Python?. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.10495)] [[Code](https://github.com/jianian0318/LLMSecureCode)]
- Detecting Code Vulnerabilities using LLMs. **`DSN 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11068842)] [[Code](https://github.com/a24167566/LLMs-Code-Vulnerability-Detection)]
- Large Language Models for Multilingual Vulnerability Detection: How Far Are We?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07503)] [[Code](https://github.com/SpanShu96/Large-Language-Model-for-Multilingual-Vulnerability-Detection/tree/main)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- SecureMind: A Framework for Benchmarking Large Language Models in Memory Bug Detection and Repair. **`ISMM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3735950.3735954)] [[Code](https://github.com/HuantWang/SecureMind)]
- An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16310)] [[Code](https://zenodo.org/records/15572151)]
- VulnTeam: A Team Collaboration Framework for LLM-based Vulnerability Detection. **`IJCNN 2025`** [[Paper](https://ieeexplore.ieee.org/document/11229292)]
- Revisiting Pre-trained Language Models for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.16887)]
- Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.03051)]
- Enhancing Vulnerability Detection by Fusing Code Semantic Features with LLM-generated Explanations. **`Information Fusion 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253525005238)] [[Code](https://github.com/XUPT-SSS/FuSEVul)]
- Should We Evaluate LLM Based Security Analysis Approaches on Open Source Systems?. **`ASE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11334389/)]
- LOSVER: Line-Level Modifiability Signal-Guided Vulnerability Detection and Classification. **`ASE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11334430)] [[Code](https://github.com/waroad/losver)] [[Code](https://figshare.com/articles/conference_contribution/Backup_code_and_checkpoints_for_Localizer_and_Detector_from_paper_b_LOSVER_Line-Level_Modifiability_Signal-Guided_Vulnerability_Detection_and_Classification_b_/29192708)]
- Learning-based Models for Vulnerability Detection: An Extensive Study. **`EMSE 2024`** [[Paper](https://arxiv.org/abs/2408.07526)] [[Code](https://figshare.com/s/bde8e41890e8179fbe5f?file=41894784)]
- A Sequential Multi-Stage Approach for Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making. **`EMNLP 2025`** [[Paper](https://aclanthology.org/2025.emnlp-main.1071/)]
- Compressing Large Language Models for SQL Injection Detection: A Case Study on Deep Seek-Coder and Meta-llama-3-70b-instruct. **`FRUCT 2025`** [[Paper](https://ieeexplore.ieee.org/document/11239157)]
- Retrieval-Augmented Few-Shot Prompting Versus Fine-Tuning for Code Vulnerability Detection. **`FLLM 2025`** [[Paper](https://ieeexplore.ieee.org/document/11391248)]
- LLM-based Vulnerability Detection at Project Scale: An Empirical Study. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.19239)] [[Code](https://github.com/Feng-Jay/LLM4Security)]
- LLMs in Code Vulnerability Analysis: A Proof of Concept. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.08691)] [[Code](https://figshare.com/s/a06ec09cd1bd98e6dd45)]
- Evaluating and Enhancing the Vulnerability Reasoning Capabilities of Large Language Models. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06687v1)]
- ChatGPT for Vulnerability Detection, Classification, and Repair: How Far Are We?. **`APSEC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10479409)] [[Code](https://github.com/awsm-research/ChatGPT4Vul)]
- Enhancing Code Security Through Open-source Large Language Models: A Comparative Study. **`FPS 2023`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-57537-2_15)]
- Exploring the Limits of ChatGPT in Software Security Applications. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2312.05275)]
- Evaluating Large Language Models in Vulnerability Detection Under Variable Context Windows. **`ICMLA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10903489)]
- The Impact of Prompt Language and Representation on LLM Reasoning: A Multilingual Empirical Study. **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/document/11318327)]
- A Systematic Study of Code Obfuscation Against LLM-based Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.16538)] [[Code](https://github.com/oxygen-hunter/SoK-Code-Obfuscation-in-LLM-VD-arxiv)]
- From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.10485)] [[Code](https://github.com/Chaomeng-Lu/A-Practical-Evaluation-of-Deep-Learning-Models-and-LLMs-for-Vulnerability-Detection)]
- Diverse LLMs vs. Vulnerabilities: Who Detects and Fixes Them Better?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.12536)] [[Code](https://github.com/Erroristotle/DVDR_LLM)]
- GRACE: Empowering LLM-based Software Vulnerability Detection with Graph Structure and In-Context Learning. **`JSS 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0164121224000748)] [[Code](https://github.com/P-E-Vul/GRACE)]
- CASTLE: Benchmarking Dataset for Static Code Analyzers and LLMs towards CWE Detection. **`TASE 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-98208-8_15)] [[Code](https://github.com/CASTLE-Benchmark)]
- Impact of Identifier Normalization on Vulnerability  Detection Techniques. **`SANER 2025`** [[Paper](https://ieeexplore.ieee.org/document/11272061)] [[Code](https://github.com/tuhh-softsec/Impact-of-Identifier-Normalization-on-Vulnerability-Detection-Techniques)]
- Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities. **`ICST 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10988968)] [[Code](https://github.com/seal-research/secvul-llm-study/)]
- Assessing the Effectiveness of LLMs in Android Application Vulnerability Analysis. **`ADIoT 2024`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-85593-1_9)]
- HALURust: Exploiting Hallucinations of Large Language Models to Detect Vulnerabilities in Rust. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.10793)]
- Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.01449)] [[Code](https://github.com/soarsmu/SVD-Bench)]
- Reasoning with LLMs for Zero-Shot Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.17885)] [[Code](https://github.com/Erroristotle/VulnSage)]
- Transformer-based Vulnerability Detection in Code at EditTime: Zero-shot, Few-shot, or Fine-tuning?. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2306.01754)]
- LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks. **`SP 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10646663/)] [[Code](https://github.com/ai4cloudops/SecLLMHolmes)]
- Large Language Model for Vulnerability Detection: Emerging Results and Future Directions. **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3639476.3639762)] [[Code](https://github.com/soarsmu/ChatGPT-VulDetection)]
- Automating the Detection of Code Vulnerabilities by Analyzing GitHub Issues. **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028308)]
- A Comparative Study of Machine Learning and Large Language Models for SQL and NoSQL Injection Vulnerability Detection. **`SIST 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11139190)]
- Leveraging Large Language Models for Command Injection Vulnerability Analysis in Python: An Empirical Study on Popular Open-Source Projects. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.15088)]
- Software Vulnerability Detection using Large Language Models. **`ISSRE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10301302)]
- Exploring AI for Vulnerability Detection and Repair. **`CARS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10778769)]
- A Qualitative Study on Using ChatGPT for Software Security: Perception vs. Practicality. **`TPS-ISA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10835695)] [[Code](https://figshare.com/articles/dataset/Reproduction_package_for_paper_A_Qualitative_Study_on_Using_ChatGPT_for_Software_Security_Perception_vs_Practicality_/24452365?file=48008890)]
- VulnerAI: GPT Based Web Application Vulnerability Detection. **`ICAMAC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10828788)]
- Llama-Based Source Code Vulnerability Detection: Prompt Engineering vs Fine Tuning. **`ESORICS 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-032-07884-1_15)] [[Code](https://github.com/DynaSoumhaneOuchebara/Llama-based-vulnerability-detection)]
- Real-VulLLM: An LLM Based Assessment Framework in the Wild. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.04056)]
- Distilling Lightweight Language Models for C/C++ Vulnerabilities. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.06645)] [[Code](https://github.com/yangxiaoxuan123/FineSec_detect)]

<a name="in-context"></a>
##### In-Context
- Using ChatGPT as a Static Application Security Testing Tool. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.14434)] [[Code](https://github.com/abakhshandeh/ChatGPTasSAST)]
- Software Vulnerability Detection with GPT and In-Context Learning. **`DSC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10381286)]
- LLbezpeky: Leveraging Large Language Models for Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.01269)]
- A Preliminary Study on Using Large Language Models in Software Pentesting. **`NDSS 2024`** [[Paper](https://arxiv.org/abs/2401.17459)]
- Exploration On Prompting LLM With Code-Specific Information For Vulnerability Detection. **`SSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10664399)]
- Effectiveness of ChatGPT for Static Analysis: How Far Are We?. **`AIware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3664646.3664777)] [[Code](https://zenodo.org/records/10828316)]
- Large Language Models for Secure Code Assessment: A Multi-Language Empirical Study. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.06428)]
- VulDetectBench: Evaluating the Deep Capability of Vulnerability Detection with Large Language Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.07595)] [[Code](https://github.com/Sweetaroo/VulDetectBench)]
- Boosting Cybersecurity Vulnerability Scanning based on LLM-supported Static Application Security Testing. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.15735)]
- To Err is Machine: Vulnerability Detection Challenges LLM Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- Streamlining Security Vulnerability Triage with Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.18908)] [[Code](https://zenodo.org/records/14776104)]
- Harnessing Large Language Models for Software Vulnerability Detection: A Comprehensive Benchmarking Study. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10879492)]
- IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities. **`ICLR 2024`** [[Paper](https://arxiv.org/abs/2405.17238)] [[Code](https://github.com/iris-sast/iris)]
- Context-Enhanced Vulnerability Detection Based on Large Language Models. **`TOSEM 2025`** [[Paper](https://arxiv.org/abs/2504.16877)] [[Code](https://github.com/DoeSEResearch/PacVD)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- Beyond Static Pattern Matching? Rethinking Automatic Cryptographic API Misuse Detection in the Era of LLMs. **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3728875)]
- An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16310)] [[Code](https://zenodo.org/records/15572151)]
- Large Language Models Versus Static Code Analysis Tools: A Systematic Benchmark for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/pdf/2508.04448)] [[Code](https://github.com/Damian0401/ProjectAnalyzer)]
- LLM-GUARD: Large Language Model-Based Detection and Repair of Bugs and Security Vulnerabilities in C++ and Python. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.16419)] [[Code](https://github.com/NoujoudNader/LLM-Bugs-Detection)]
- Think Broad, Act Narrow: CWE Identification with Multi-Agent Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.01451)] [[Code](https://zenodo.org/records/15871507)]
- VulAgent: Hypothesis-Validation based Multi-Agent Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2509.11523)]
- From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.14012v1)] [[Code](https://github.com/youpengl/OpenVul)]
- ChatGPT for Vulnerability Detection, Classification, and Repair: How Far Are We?. **`APSEC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10479409)] [[Code](https://github.com/awsm-research/ChatGPT4Vul)]
- Software Vulnerability Detection Using LLM: Does Additional Information Help?. **`ACSAC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10917361)] [[Code](https://github.com/research7485/vulnerability_detection)]
- Large Language Models Cannot Reliably Detect Vulnerabilities in JavaScript: The First Systematic Benchmark and Evaluation. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.01255)] [[Code](https://github.com/SecJS-Vuln-Benchmark/SecJS-Benchmark)] [[Code](https://secjs-vuln-benchmark.github.io/SecJS-Benchmark/)]
- VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.07533)] [[Code](https://github.com/ucsb-mlsec/VulnLLM-R)]
- LLM-CloudSec: Large Language Model Empowered Automatic and Deep Vulnerability Analysis for Intelligent Clouds. **`INFOCOM 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10620804)] [[Code](https://github.com/DPCa0/LLM-CloudSec)]
- LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks. **`SP 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10646663/)] [[Code](https://github.com/ai4cloudops/SecLLMHolmes)]
- SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.19828)] [[Code](https://github.com/basimbd/SecVulEval)]
- ♪ With a Little Help from My (LLM) Friends: Enhancing Static Analysis with LLMs to Detect Software Vulnerabilities. **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028575)]
- Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10961)] [[Code](https://figshare.com/s/1514bc9a7aa64b46d94e)]
- Enhancing Large Language Models for Secure Code Generation: A Dataset-driven Study on Vulnerability Mitigation. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2310.16263)]
- DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection. **`JSS 2024`** [[Paper]()] [[Code](https://github.com/Yang-Yanjing/DLAP)]
- The Richer Representation Fallacy: Are We Just Adding Noise to LLM-based Software Vulnerability Detectors?. **`ICOCO 2025`** [[Paper](https://ieeexplore.ieee.org/document/11334069)]

<a name="few-shot"></a>
##### Few-Shot
- Chain-of-Thought Prompting of Large Language Models for Discovering and Fixing Software Vulnerabilities. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2402.17230)]
- Software Vulnerability Prediction in Low-Resource Languages: An Empirical Study of CodeBERT and ChatGPT. **`EASE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3661167.3661281)] [[Code](https://github.com/lhmtriet/LLM4Vul)]
- Effectiveness of ChatGPT for Static Analysis: How Far Are We?. **`AIware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3664646.3664777)] [[Code](https://zenodo.org/records/10828316)]
- Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.16235)]
- Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.01001)] [[Code](https://figshare.com/s/5da14b0776750c6fa787)]
- Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.14841)]
- To Err is Machine: Vulnerability Detection Challenges LLM Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- Manual Prompt Engineering is Not Dead: A Case Study on Large Language Models for Code Vulnerability Detection with DSPy. **`CDMA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908746)]
- Vulnerability Detection with Code Language Models: How Far are We?. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029911)] [[Code](https://github.com/DLVulDet/PrimeVul)]
- Context-Enhanced Vulnerability Detection Based on Large Language Models. **`TOSEM 2025`** [[Paper](https://arxiv.org/abs/2504.16877)] [[Code](https://github.com/DoeSEResearch/PacVD)]
- Expert-in-the-Loop Systems with Cross-Domain and In-Domain Few-Shot Learning for Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10104)]
- Large Language Models for Multilingual Vulnerability Detection: How Far Are We?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07503)] [[Code](https://github.com/SpanShu96/Large-Language-Model-for-Multilingual-Vulnerability-Detection/tree/main)]
- Large Language Models for In-File Vulnerability Localization Can Be ""Lost in the End"". **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3715758)] [[Code](https://zenodo.org/records/14840519)]
- SecureMind: A Framework for Benchmarking Large Language Models in Memory Bug Detection and Repair. **`ISMM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3735950.3735954)] [[Code](https://github.com/HuantWang/SecureMind)]
- Revisiting Pre-trained Language Models for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.16887)]
- Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories. **`Unknown 2025`** [[Paper](https://arxiv.org/abs/2503.03586)]
- CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.11599)]
- GPTVD: vulnerability detection and analysis method based on LLM’s chain of thoughts. **`ASE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10515-025-00550-4)] [[Code](https://github.com/chenyn273/GPTVD)]
- Can LLM Prompting Serve as a Proxy for Static Analysis in Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.12039)]
- Towards Explainable Vulnerability Detection With Large Language Models. **`TSE 2024`** [[Paper](https://arxiv.org/abs/2406.09701)]
- Leveraging Intra-and Inter-References in Vulnerability Detection using Multi-Agent Collaboration Based on LLMs. **`Cluster Computing 2025`** [[Paper](https://link.springer.com/article/10.1007/s10586-025-05721-2)]
- Learning-based Models for Vulnerability Detection: An Extensive Study. **`EMSE 2024`** [[Paper](https://arxiv.org/abs/2408.07526)] [[Code](https://figshare.com/s/bde8e41890e8179fbe5f?file=41894784)]
- Retrieval-Augmented Few-Shot Prompting Versus Fine-Tuning for Code Vulnerability Detection. **`FLLM 2025`** [[Paper](https://ieeexplore.ieee.org/document/11391248)]
- LLMs in Code Vulnerability Analysis: A Proof of Concept. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.08691)] [[Code](https://figshare.com/s/a06ec09cd1bd98e6dd45)]
- MulVul: Retrieval-augmented Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.18847)]
- Transformer-based Vulnerability Detection in Code at EditTime: Zero-shot, Few-shot, or Fine-tuning?. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2306.01754)]
- LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks. **`SP 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10646663/)] [[Code](https://github.com/ai4cloudops/SecLLMHolmes)]
- Large Language Model for Vulnerability Detection: Emerging Results and Future Directions. **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3639476.3639762)] [[Code](https://github.com/soarsmu/ChatGPT-VulDetection)]
- Multitask-Based Evaluation of Open-Source LLM on Software Vulnerability. **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10706805)] [[Code](https://github.com/vinci-grape/VulEmpirical)]
- On Selecting Few-Shot Examples for LLM-based Code Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.27675)]
- Llama-Based Source Code Vulnerability Detection: Prompt Engineering vs Fine Tuning. **`ESORICS 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-032-07884-1_15)] [[Code](https://github.com/DynaSoumhaneOuchebara/Llama-based-vulnerability-detection)]

<a name="rag"></a>
##### RAG
- Software Vulnerability Detection with GPT and In-Context Learning. **`DSC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10381286)]
- LLbezpeky: Leveraging Large Language Models for Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.01269)]
- VulEval: Towards Repository-Level Evaluation of Software Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2404.15596)]
- Exploration On Prompting LLM With Code-Specific Information For Vulnerability Detection. **`SSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10664399)]
- Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.00197)]
- Boosting Cybersecurity Vulnerability Scanning based on LLM-supported Static Application Security Testing. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.15735)]
- Research on the LLM-Driven Vulnerability Detection System Using LProtector. **`ICDSCA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10859408)]
- To Err is Machine: Vulnerability Detection Challenges LLM Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- SSRFSeek: An LLM-based Static Analysis Framework for Detecting SSRF Vulnerabilities in PHP Applications. **`AINIT 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11035424)]
- Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.11147)] [[Code](https://github.com/knowledgerag4llmvuld/knowledgerag4llmvuld)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.11599)]
- Think Broad, Act Narrow: CWE Identification with Multi-Agent Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.01451)] [[Code](https://zenodo.org/records/15871507)]
- DeepVulHunter: Enhancing the Code Vulnerability Detection Capability of LLMs through Multi-Round Analysis. **`JIIS 2025`** [[Paper](https://link.springer.com/article/10.1007/s10844-025-00982-0)]
- Leveraging Intra-and Inter-References in Vulnerability Detection using Multi-Agent Collaboration Based on LLMs. **`Cluster Computing 2025`** [[Paper](https://link.springer.com/article/10.1007/s10586-025-05721-2)]
- An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems. **`CASCON 2025`** [[Paper](https://ieeexplore.ieee.org/document/11344502)]
- A Sequential Multi-Stage Approach for Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making. **`EMNLP 2025`** [[Paper](https://aclanthology.org/2025.emnlp-main.1071/)]
- Specification-Guided Vulnerability Detection with Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.04014)] [[Code](https://github.com/zhuhaopku/VulInstruct-temp)]
- Retrieval-Augmented Few-Shot Prompting Versus Fine-Tuning for Code Vulnerability Detection. **`FLLM 2025`** [[Paper](https://ieeexplore.ieee.org/document/11391248)]
- MulVul: Retrieval-augmented Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.18847)]
- Software Vulnerability Detection Using LLM: Does Additional Information Help?. **`ACSAC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10917361)] [[Code](https://github.com/research7485/vulnerability_detection)]
- GRACE: Empowering LLM-based Software Vulnerability Detection with Graph Structure and In-Context Learning. **`JSS 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0164121224000748)] [[Code](https://github.com/P-E-Vul/GRACE)]
- Assessing the Effectiveness of LLMs in Android Application Vulnerability Analysis. **`ADIoT 2024`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-85593-1_9)]
- Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.01449)] [[Code](https://github.com/soarsmu/SVD-Bench)]
- LLM-CloudSec: Large Language Model Empowered Automatic and Deep Vulnerability Analysis for Intelligent Clouds. **`INFOCOM 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10620804)] [[Code](https://github.com/DPCa0/LLM-CloudSec)]
- DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection. **`JSS 2024`** [[Paper]()] [[Code](https://github.com/Yang-Yanjing/DLAP)]
- Llama-Based Source Code Vulnerability Detection: Prompt Engineering vs Fine Tuning. **`ESORICS 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-032-07884-1_15)] [[Code](https://github.com/DynaSoumhaneOuchebara/Llama-based-vulnerability-detection)]
- Real-VulLLM: An LLM Based Assessment Framework in the Wild. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.04056)]

<a name="cot"></a>
##### CoT
- Chain-of-Thought Prompting of Large Language Models for Discovering and Fixing Software Vulnerabilities. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2402.17230)]
- Enhancing Software Code Vulnerability Detection Using GPT-4o and Claude-3.5 Sonnet: A Study on Prompt Engineering Techniques. **`Electronics 2024`** [[Paper](https://www.mdpi.com/2079-9292/13/13/2657)]
- Exploration On Prompting LLM With Code-Specific Information For Vulnerability Detection. **`SSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10664399)]
- Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.00197)]
- Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.16235)]
- Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning. **`ACL 2024`** [[Paper](https://arxiv.org/abs/2406.03718)] [[Code](https://github.com/CGCL-codes/VulLLM)]
- Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.01001)] [[Code](https://figshare.com/s/5da14b0776750c6fa787)]
- Research on the LLM-Driven Vulnerability Detection System Using LProtector. **`ICDSCA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10859408)]
- Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.14841)]
- To Err is Machine: Vulnerability Detection Challenges LLM Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- Harnessing Large Language Models for Software Vulnerability Detection: A Comprehensive Benchmarking Study. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10879492)]
- Manual Prompt Engineering is Not Dead: A Case Study on Large Language Models for Code Vulnerability Detection with DSPy. **`CDMA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908746)]
- Vulnerability Detection with Code Language Models: How Far are We?. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029911)] [[Code](https://github.com/DLVulDet/PrimeVul)]
- Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.13474)] [[Code](https://anonymous.4open.science/r/CORRECT/README.md)]
- Context-Enhanced Vulnerability Detection Based on Large Language Models. **`TOSEM 2025`** [[Paper](https://arxiv.org/abs/2504.16877)] [[Code](https://github.com/DoeSEResearch/PacVD)]
- SSRFSeek: An LLM-based Static Analysis Framework for Detecting SSRF Vulnerabilities in PHP Applications. **`AINIT 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11035424)]
- Detecting Code Vulnerabilities using LLMs. **`DSN 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11068842)] [[Code](https://github.com/a24167566/LLMs-Code-Vulnerability-Detection)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- SecureMind: A Framework for Benchmarking Large Language Models in Memory Bug Detection and Repair. **`ISMM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3735950.3735954)] [[Code](https://github.com/HuantWang/SecureMind)]
- Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07390)] [[Code](https://github.com/Xin-Cheng-Wen/PO4Vul)]
- An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16310)] [[Code](https://zenodo.org/records/15572151)]
- VulnTeam: A Team Collaboration Framework for LLM-based Vulnerability Detection. **`IJCNN 2025`** [[Paper](https://ieeexplore.ieee.org/document/11229292)]
- Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.03051)]
- Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories. **`Unknown 2025`** [[Paper](https://arxiv.org/abs/2503.03586)]
- CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.11599)]
- Optimizing Code Vulnerability Detection via GRPO and SFT Fine-Tuning of Compact LLMs. **`DSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11360339)]
- MAVUL: Multi-Agent Vulnerability Detection via Contextual Reasoning and Interactive Refinement. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.00317)] [[Code](https://github.com/youpengl/MAVUL)]
- GPTVD: vulnerability detection and analysis method based on LLM’s chain of thoughts. **`ASE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10515-025-00550-4)] [[Code](https://github.com/chenyn273/GPTVD)]
- Can LLM Prompting Serve as a Proxy for Static Analysis in Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.12039)]
- Towards Explainable Vulnerability Detection With Large Language Models. **`TSE 2024`** [[Paper](https://arxiv.org/abs/2406.09701)]
- iCodeReviewer: Improving Secure Code Review with Mixture of Prompts. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.12186)]
- Learning-based Models for Vulnerability Detection: An Extensive Study. **`EMSE 2024`** [[Paper](https://arxiv.org/abs/2408.07526)] [[Code](https://figshare.com/s/bde8e41890e8179fbe5f?file=41894784)]
- An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems. **`CASCON 2025`** [[Paper](https://ieeexplore.ieee.org/document/11344502)]
- Specification-Guided Vulnerability Detection with Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.04014)] [[Code](https://github.com/zhuhaopku/VulInstruct-temp)]
- Beyond Function-Level Analysis: Context-Aware Reasoning for Inter-Procedural Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06751v1)] [[Code](https://github.com/yikun-li/CPRVul)]
- Evaluating and Enhancing the Vulnerability Reasoning Capabilities of Large Language Models. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06687v1)]
- From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.14012v1)] [[Code](https://github.com/youpengl/OpenVul)]
- Large Language Models Cannot Reliably Detect Vulnerabilities in JavaScript: The First Systematic Benchmark and Evaluation. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.01255)] [[Code](https://github.com/SecJS-Vuln-Benchmark/SecJS-Benchmark)] [[Code](https://secjs-vuln-benchmark.github.io/SecJS-Benchmark/)]
- The Impact of Prompt Language and Representation on LLM Reasoning: A Multilingual Empirical Study. **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/document/11318327)]
- Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities. **`ICST 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10988968)] [[Code](https://github.com/seal-research/secvul-llm-study/)]
- Reasoning with LLMs for Zero-Shot Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.17885)] [[Code](https://github.com/Erroristotle/VulnSage)]
- LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks. **`SP 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10646663/)] [[Code](https://github.com/ai4cloudops/SecLLMHolmes)]
- ♪ With a Little Help from My (LLM) Friends: Enhancing Static Analysis with LLMs to Detect Software Vulnerabilities. **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028575)]
- DLAP: A Deep Learning Augmented Large Language Model Prompting Framework for Software Vulnerability Detection. **`JSS 2024`** [[Paper]()] [[Code](https://github.com/Yang-Yanjing/DLAP)]
- Real-VulLLM: An LLM Based Assessment Framework in the Wild. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.04056)]

<a name="t22-training"></a>
#### T2.2 Training
<a name="t221-pre-training"></a>
##### T2.2.1 Pre-Training
<a name="pre-training"></a>
###### Pre-Training
- Exploring Software Naturalness through Neural Language Models. **`arXiv 2020`** [[Paper](https://arxiv.org/abs/2006.12641)]
- Unified Pre-training for Program Understanding and Generation. **`NAACL 2021`** [[Paper](https://par.nsf.gov/servlets/purl/10336701)] [[Code](https://github.com/wasiahmad/PLBART)]
- VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection. **`IJCNN 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9892280)] [[Code](https://github.com/ICL-ml4csec/VulBERTa)]
- Leveraging Deep Learning Models for Cross-function Null Pointer Risks Detection. **`AITest 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10229470)]
- Software Vulnerabilities Detection Based on a Pre-trained Language Model. **`TrustCom 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10538979)]
- TRACED: Execution-aware Pre-training for Source Code. **`ICSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3597503.3608140)] [[Code](https://github.com/ARiSE-Lab/TRACED_ICSE_24)]
- Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10548173)] [[Code](https://zenodo.org/records/10140638)]
- StagedVulBERT: Multigranular Vulnerability Detection With a Novel Pretrained Code Model. **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10746847)] [[Code](https://github.com/YuanJiangGit/StagedVulBERT)]
- VuL-MCBERT: A Vulnerability Detection Method Based on Self-Supervised Contrastive Learning. **`CAIBDA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11183103)]
- CLeVeR: Multi-modal Contrastive Learning for Vulnerability Code Representation. **`ACL 2025`** [[Paper](https://aclanthology.org/2025.findings-acl.414/)] [[Code](https://github.com/yoimiya-nlp/CLeVeR)]
- BBVD: A BERT-based Method for Vulnerability Detection. **`IJACSA 2022`** [[Paper](https://www.proquest.com/docview/2770373789?pq-origsite=gscholar&fromopenview=true&sourcetype=Scholarly%20Journals)]
- PATVD: Vulnerability Detection Based on Pre-training Techniques and Adversarial Training. **`SmartWorld/UIC/ScalCom/DigitalTwin/PriComp/Meta 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10189687/)]
- Learning Defect Prediction from Unrealistic Data. **`SANER 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10589866)] [[Code](https://zenodo.org/records/10514652)]
- LLaVul: A Multimodal LLM for Interpretable Vulnerability Reasoning about Source Code. **`ICSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11140501)]

<a name="t222-fine-tuning"></a>
##### T2.2.2 Fine-Tuning
<a name="t2221-full-parameter-fine-tuning"></a>
###### T2.2.2.1 Full-Parameter Fine-Tuning
<a name="full-parameter-fine-tuning"></a>
###### Full-Parameter Fine-Tuning
- Exploring Software Naturalness through Neural Language Models. **`arXiv 2020`** [[Paper](https://arxiv.org/abs/2006.12641)]
- Unified Pre-training for Program Understanding and Generation. **`NAACL 2021`** [[Paper](https://par.nsf.gov/servlets/purl/10336701)] [[Code](https://github.com/wasiahmad/PLBART)]
- Detecting Integer Overflow Errors in Java Source Code via Machine Learning. **`ICTAI 2021`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9643278)]
- Deep Neural Embedding for Software Vulnerability Discovery: Comparison and Optimization. **`Security and Communication Networks 2022`** [[Paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/5203217)] [[Code](https://cybercodeintelligence.github.io/CyberCI/)]
- Cyber Security Vulnerability Detection Using Natural Language Processing. **`AIIoT 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9817336)]
- VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection. **`IJCNN 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9892280)] [[Code](https://github.com/ICL-ml4csec/VulBERTa)]
- Distilled and Contextualized Neural Models Benchmarked for Vulnerable Function Detection. **`Mathematics 2022`** [[Paper](https://www.mdpi.com/2227-7390/10/23/4482)]
- BERT-Based Vulnerability Type Identification with Effective Program Representation. **`WASA 2022`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19208-1_23#citeas)]
- An Empirical Study of Deep Learning Models for Vulnerability Detection. **`ICSE 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10172583)] [[Code](https://figshare.com/articles/dataset/An_Empirical_Study_of_Deep_Learning_Models_for_Vulnerability_Detection/20791240?file=39183863)]
- Leveraging Deep Learning Models for Cross-function Null Pointer Risks Detection. **`AITest 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10229470)]
- VulDetect: A novel technique for detecting software vulnerabilities using Language Models. **`CSR 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10224924)]
- VulExplainer: A Transformer-Based Hierarchical  Distillation for Explaining Vulnerability Types. **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10220166)] [[Code](https://github.com/awsm-research/VulExplainer)]
- When Less is Enough: Positive and Unlabeled Learning Model for Vulnerability Detection. **`ASE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10298363)] [[Code](https://github.com/PILOT-VD-2023/PILOT)]
- AIBugHunter: A Practical Tool for Predicting, Classifying and Repairing Software Vulnerabilities. **`EMSE 2023`** [[Paper](https://link.springer.com/article/10.1007/s10664-023-10346-3)] [[Code](https://github.com/awsm-research/AIBugHunter)]
- The EarlyBIRD Catches the Bug: On Exploiting Early Layers of Encoder Models for More Efficient Code Classification. **`ESEC/FSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3611643.3616304)] [[Code](https://zenodo.org/records/10499843)]
- Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation. **`ESEC/FSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3611643.3616358)] [[Code](https://github.com/jacknichao/SVulD)]
- Do Language Models Learn Semantics of Code? A Case Study in Vulnerability Detection. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2311.04109)] [[Code](https://figshare.com/s/4a16a528d6874aad51a0)]
- Software Vulnerabilities Detection Based on a Pre-trained Language Model. **`TrustCom 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10538979)]
- TRACED: Execution-aware Pre-training for Source Code. **`ICSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3597503.3608140)] [[Code](https://github.com/ARiSE-Lab/TRACED_ICSE_24)]
- BiT5: A Bidirectional NLP Approach for Advanced Vulnerability Detection in Codebase. **`Procedia Computer Science 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1877050924006306)]
- Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10548173)] [[Code](https://zenodo.org/records/10140638)]
- Towards Causal Deep Learning for Vulnerability Detection. **`ICSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3597503.3639170)] [[Code](https://figshare.com/s/0ffda320dcb96c249ef2?file=41801019)]
- VulEval: Towards Repository-Level Evaluation of Software Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2404.15596)]
- Software Vulnerability Prediction in Low-Resource Languages: An Empirical Study of CodeBERT and ChatGPT. **`EASE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3661167.3661281)] [[Code](https://github.com/lhmtriet/LLM4Vul)]
- Greening Large Language Models of Code. **`ICSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3639475.3640097)] [[Code](https://github.com/soarsmu/Avatar)]
- SVulDetector: Vulnerability Detection based on Similarity using Tree-based Attention and Weighted Graph Embedding Mechanisms. **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824002335)] [[Code](https://figshare.com/s/426156a96a83da1d38d0)]
- MultiVD: A Transformer-based Multitask Approach for Software Vulnerability Detection. **`SECRYPT 2024`** [[Paper](https://www.scitepress.org/Papers/2024/127194/127194.pdf)]
- Vulnerability Classification on Source Code Using Text Mining and Deep Learning Techniques. **`QRS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10727022)] [[Code](https://sites.google.com/view/vulnerabilityclassification/)]
- M2CVD: Enhancing Vulnerability Understanding through Multi-Model Collaboration for Code Vulnerability Detection. **`TOSEM 2024`** [[Paper](https://arxiv.org/abs/2406.05940)] [[Code](https://github.com/HotFrom/M2CVD)]
- Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.16235)]
- From Generalist to Specialist: Exploring CWE-Specific Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.02329)]
- Uncovering the Limits of Machine Learning for Automatic Vulnerability Detection. **`USENIX Security 2023`** [[Paper](https://www.usenix.org/conference/usenixsecurity24/presentation/risse)] [[Code](https://github.com/niklasrisse/USENIX_2024)] [[Code](https://github.com/niklasrisse/VPP)]
- VulSim: Leveraging Similarity of Multi-Dimensional Neighbor Embeddings for Vulnerability Detection. **`USENIX Security 2024`** [[Paper](https://www.usenix.org/conference/usenixsecurity24/presentation/shimmi)] [[Code](https://github.com/SamihaShimmi/VulSim)]
- Bridge and Hint: Extending Pre-trained Language Models for Long-Range Code. **`ISSTA 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3650212.3652127)] [[Code](https://anonymous.4open.science/r/EXPO/README.md)]
- Code Vulnerability Detection: A Comparative Analysis of Emerging Large Language Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.10490)]
- SCALE: Constructing Structured Natural Language Comment Trees for Software Vulnerability Detection. **`ISSTA 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3650212.3652124)] [[Code](https://github.com/Xin-Cheng-Wen/Comment4Vul)]
- Outside the Comfort Zone: Analysing LLM Capabilities in Software Vulnerability Detection. **`ESORICS 2024`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70879-4_14)]
- Vulnerability Prediction using Pre-trained Models: An Empirical Evaluation. **`MASCOTS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10786510)] [[Code](https://sites.google.com/view/vpllm/)]
- StagedVulBERT: Multigranular Vulnerability Detection With a Novel Pretrained Code Model. **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10746847)] [[Code](https://github.com/YuanJiangGit/StagedVulBERT)]
- Applying Contrastive Learning to Code Vulnerability Type Classification. **`EMNLP 2024`** [[Paper](https://aclanthology.org/2024.emnlp-main.666/)]
- Enhancing Vulnerability Detection Efficiency: An Exploration of Light-weight LLMs with Hybrid Code Features. **`JISA 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212624002278)] [[Code](https://github.com/JNL-28/Enhancing-Vulnerability-Detection-Efficiency)]
- Streamlining Security Vulnerability Triage with Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.18908)] [[Code](https://zenodo.org/records/14776104)]
- DMVL4AVD: A Deep Multi-View Learning Model for Automated Vulnerability Detection. **`Neural Comput. Appl. 2025`** [[Paper](https://link.springer.com/article/10.1007/s00521-024-10892-x)] [[Code](https://drive.google.com/file/d/1-qWqmRuBi8kRAAE2yiG6JNiY8vLYxXlz/view)]
- Finetuning Large Language Models for Vulnerability Detection. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908394)] [[Code](https://github.com/rmusab/vul-llm-finetune)]
- Vulnerability Detection with Code Language Models: How Far are We?. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029911)] [[Code](https://github.com/DLVulDet/PrimeVul)]
- Trace Gadgets: Minimizing Code Context for Machine Learning-Based Vulnerability Prediction. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.13676)]
- Metamorphic-Based Many-Objective Distillation of LLMs for Code-Related Tasks. **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11029766)] [[Code](https://zenodo.org/records/14857610)]
- XGV-BERT: Leveraging Contextualized Language Model and Graph Neural Network for Efficient Software Vulnerability Detection. **`The Journal of Supercomputing 2023`** [[Paper](https://link.springer.com/article/10.1007/s11227-025-07198-7)]
- Leveraging Multi-Task Learning to Improve the Detection of SATD and Vulnerability. **`ICPC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11025930)] [[Code](https://github.com/moritzmock/multitask-vulberability-detection)]
- Closing the Gap: A User Study on the Real-world Usefulness of AI-powered Vulnerability Detection \& Repair in the IDE. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029760)] [[Code](https://figshare.com/articles/dataset/Closing_the_Gap_A_User_Study_on_the_Real-world_Usefulness_of_AI-powered_Vulnerability_Detection_Repair_in_the_IDE/26367139?file=52478936)]
- Human-Understandable Explanation for Software Vulnerability Prediction. **`JSS 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0164121225001232)] [[Code](https://github.com/quy-ng/human-xai-software-vulnerability-prediction)]
- LPASS: Linear Probes as Stepping Stones for Vulnerability Detection using Compressed LLMs. **`JISA 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212625001620)]
- Smart Cuts: Enhance Active Learning for Vulnerability Detection by Pruning Bad Seeds. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.20444)]
- CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2411.17274)] [[Code](https://github.com/yikun-li/CleanVul)]
- Line-level Semantic Structure Learning for Code Vulnerability Detection. **`Internetware 2024`** [[Paper](https://arxiv.org/abs/2407.18877)] [[Code](https://figshare.com/articles/dataset/CSLS_model_code_and_data/26391658)]
- VuL-MCBERT: A Vulnerability Detection Method Based on Self-Supervised Contrastive Learning. **`CAIBDA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11183103)]
- Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07390)] [[Code](https://github.com/Xin-Cheng-Wen/PO4Vul)]
- One-for-All Does Not Work! Enhancing Vulnerability Detection by Mixture-of-Experts (MoE). **`PACMSE 2025`** [[Paper](https://arxiv.org/abs/2501.16454)]
- An Automatic Classification Model for Long Code Vulnerabilities Based on the Teacher-Student Framework. **`QRS 2025`** [[Paper](https://ieeexplore.ieee.org/document/11216609)]
- Revisiting Pre-trained Language Models for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.16887)]
- Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.03051)]
- AI-Powered Vulnerability Detection in Code Using BERT-Based LLM with Transparency Measures. **`ITC-Egypt 2025`** [[Paper](https://ieeexplore.ieee.org/document/11186618)]
- Structural Semantic Enhancement: Better Integrating Code Semantics for Vulnerability Detection. **`InfSof 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0950584925001636?via%3Dihub)]
- Improving Software Security Through a LLM-Based Vulnerability Detection Model. **`DEXA 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-032-02049-9_9)]
- Out of Distribution, Out of Luck: How Well Can LLMs Trained on Vulnerability Datasets Detect Top 25 CWE Weaknesses?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.21817)] [[Code](https://github.com/yikun-li/TitanVul-BenchVul)]
- SAFE: A Novel Approach For Software Vulnerability Detection from Enhancing The Capability of Large Language Models. **`ASIACCS 2024`** [[Paper](https://arxiv.org/abs/2409.00882)]
- Software Vulnerability Detection using Large Language Models. **`SecureComm 2024`** [[Paper](https://arxiv.org/abs/2410.00249)]
- Data and Context Matter: Towards Generalizing AI-based Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.16625)]
- Utilizing Large Programming Language Models on Software Vulnerability Detection. **`ASYU 2025`** [[Paper](https://ieeexplore.ieee.org/document/11208282)]
- PIONEER: Improving the Robustness of Student Models when Compressing Pre-Trained Models of Code. **`ASE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10515-025-00560-2)] [[Code](https://github.com/illsui1on/PIONEER)]
- FuncVul: An Effective Function Level Vulnerability Detection Model using LLM and Code Chunk. **`ESORICS 2025`** [[Paper](https://arxiv.org/abs/2506.19453)] [[Code](https://github.com/sajalhalder/FuncVul)]
- Cross-Domain Evaluation of Transformer-Based Vulnerability Detection on Open and Industry Data. **`PROFES 2025`** [[Paper](https://arxiv.org/abs/2509.09313)] [[Code](https://github.com/CybersecurityLab-unibz/cross_domain_evaluation)]
- LOSVER: Line-Level Modifiability Signal-Guided Vulnerability Detection and Classification. **`ASE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11334430)] [[Code](https://github.com/waroad/losver)] [[Code](https://figshare.com/articles/conference_contribution/Backup_code_and_checkpoints_for_Localizer_and_Detector_from_paper_b_LOSVER_Line-Level_Modifiability_Signal-Guided_Vulnerability_Detection_and_Classification_b_/29192708)]
- Leveraging Self-Paced Learning for Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.09212)] [[Code](https://figshare.com/s/bef3211194fc18fe375e)]
- Retrieval-Augmented Few-Shot Prompting Versus Fine-Tuning for Code Vulnerability Detection. **`FLLM 2025`** [[Paper](https://ieeexplore.ieee.org/document/11391248)]
- Evaluating and Enhancing the Vulnerability Reasoning Capabilities of Large Language Models. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06687v1)]
- SecCodePRM: A Process Reward Model for Code Security. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.10418v1)] [[Code](https://github.com/viviable/seccodeprm)]
- From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.14012v1)] [[Code](https://github.com/youpengl/OpenVul)]
- Automated Software Vulnerability Detection via Pre-trained Context Encoder and Self Attention. **`ICDF2C 2021`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-06365-7_15)]
- BBVD: A BERT-based Method for Vulnerability Detection. **`IJACSA 2022`** [[Paper](https://www.proquest.com/docview/2770373789?pq-origsite=gscholar&fromopenview=true&sourcetype=Scholarly%20Journals)]
- Exploring Transformers for Multi-Label Classification of Java Vulnerabilities. **`QRS 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10062434)] [[Code](https://github.com/TQRG/VDET-for-Java)]
- Transformer-Based Language Models for Software Vulnerability Detection. **`ACSAC 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3564625.3567985)] [[Code](https://bitbucket.csiro.au/users/jan087/repos/acsac-2022-submission/browse)]
- Code Defect Detection Method Based on BERT and Ensemble. **`ICCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10507306)]
- Assessing the Effectiveness of Vulnerability Detection via Prompt Tuning: An Empirical Study. **`APSEC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10479384)] [[Code](https://github.com/P-E-Vul/prompt-empircial-vulnerability)]
- Optimizing Pre-trained Language Models for Efficient Vulnerability Detection in Code Snippets. **`ICCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10507456)]
- Vulnerability Detection in Popular Programming Languages with Language Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.15905)] [[Code](https://github.com/syafiq/llm_vd)]
- On the Compression of Language Models for Code: An Empirical Study on CodeBERT. **`SANER 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10992473)] [[Code](https://zenodo.org/records/14357478)]
- LLM-Based Approach for Buffer Overflow Detection in Source Code. **`CIT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11021816)]
- A Source Code Vulnerability Detection Method Based on Positive-Unlabeled Learning. **`RICAI 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10911761)]
- EnStack: An Ensemble Stacking Framework of Large Language Models for Enhanced Vulnerability Detection in Source Code. **`BigData 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10825609)]
- Software Vulnerability Detection Using LLM: Does Additional Information Help?. **`ACSAC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10917361)] [[Code](https://github.com/research7485/vulnerability_detection)]
- Enhancing Source Code Vulnerability Detection Using Flattened Code Graph Structures. **`ICFTIC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10913325)]
- MVD: A Multi-Lingual Software Vulnerability Detection Framework. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.06166)] [[Code](https://figshare.com/s/10ec70108294a225f391)]
- Python Source Code Vulnerability Detection Based on CodeBERT Language Model. **`ACAI 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10899694)]
- VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.07533)] [[Code](https://github.com/ucsb-mlsec/VulnLLM-R)]
- Intelligent Detection of Vulnerable Functions in Software through Neural Embedding-based Code Analysis. **`IJNM 2022`** [[Paper](https://onlinelibrary.wiley.com/doi/full/10.1002/nem.2198)] [[Code](https://cybercodeintelligence.github.io/CyberCI/)]
- Learning Defect Prediction from Unrealistic Data. **`SANER 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10589866)] [[Code](https://zenodo.org/records/10514652)]
- Python Source Code Vulnerability Detection with Named Entity Recognition. **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824001032)] [[Code](https://github.com/mmeberg/PyVulDet-NER)]
- Making Vulnerability Prediction more Practical: Prediction, Categorization, and Localization. **`IST 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0950584924000636)] [[Code](https://github.com/liucyy/VulPCL)]
- SecureFalcon: Are We There Yet in Automated Software Vulnerability Detection With LLMs?. **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10910240)]
- Impact of Identifier Normalization on Vulnerability  Detection Techniques. **`SANER 2025`** [[Paper](https://ieeexplore.ieee.org/document/11272061)] [[Code](https://github.com/tuhh-softsec/Impact-of-Identifier-Normalization-on-Vulnerability-Detection-Techniques)]
- You Only Train Once: A Flexible Training Framework for Code Vulnerability Detection Driven by Vul-Vector. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10988)]
- Security Vulnerability Detection Using Deep Learning Natural Language Processing. **`INFOCOM 2021`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9484500)]
- LineVul: A Transformer-based Line-level Vulnerability Prediction. **`MSR 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3524842.3528452)] [[Code](https://github.com/awsm-research/LineVul)]
- Transformer-based Vulnerability Detection in Code at EditTime: Zero-shot, Few-shot, or Fine-tuning?. **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2306.01754)]
- Keeping Pace with Ever-Increasing Data: Towards Continual Learning of Code Intelligence Models. **`ICSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10172346)] [[Code](https://github.com/ReliableCoding/REPEAT)]
- Detecting Vulnerabilities in IoT Software: New Hybrid Model and Comprehensive Data Analysis. **`JISA 2023`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212623000510)]
- VulDefend: A Novel Technique based on Pattern-exploiting Training for Detecting Software Vulnerabilities Using Language Models. **`JEEIT 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10185860)]
- DB-CBIL: A DistilBert-Based Transformer Hybrid Model Using CNN and BiLSTM for Software Vulnerability Detection. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10517582)]
- VulD-CodeBERT: CodeBERT-Based Vulnerability Detection Model for C/C++ Code. **`CISCE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10653337)]
- Automating the Detection of Code Vulnerabilities by Analyzing GitHub Issues. **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028308)]
- A Comparative Study of Machine Learning and Large Language Models for SQL and NoSQL Injection Vulnerability Detection. **`SIST 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11139190)]
- Adversarial Training for Robustness Enhancement in LLM-Based Code Vulnerability Detection. **`CISCE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11065803)]
- Learning to Focus: Context Extraction for Efficient Code Vulnerability Detection with Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.17460)]
- An Automated Code Review Framework Based on BERT and Qianwen Large Model. **`CCAI 2025`** [[Paper](https://ieeexplore.ieee.org/document/11189422)]
- VulDeBERT: A Vulnerability Detection System Using BERT. **`ISSRE 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9985089)] [[Code](https://github.com/SKKU-SecLab/VulDeBERT)]
- DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection. **`RAID 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3607199.3607242)] [[Code](https://github.com/wagner-group/diversevul)]
- Software Vulnerability Detection using Large Language Models. **`ISSRE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10301302)]
- Improving Long-Tail Vulnerability Detection Through Data Augmentation Based on Large Language Models. **`ICSME 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10795073)] [[Code](https://github.com/LuckyDengXiao/LERT)]
- VULREM: Fine-Tuned BERT-Based Source-Code Potential Vulnerability Scanning System to Mitigate Attacks in Web Applications. **`Applied Sciences 2024`** [[Paper](https://www.mdpi.com/2076-3417/14/21/9697)]
- Vul-LMGNNs: Fusing Language Models and Online-distilled Graph Neural Networks for Code Vulnerability Detection. **`Information Fusion 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253524005268)] [[Code](https://github.com/Vul-LMGNN/vul-LMGGNN)]
- SecureQwen: Leveraging LLMs for Vulnerability Detection in Python Codebases. **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824004565)]
- Multitask-Based Evaluation of Open-Source LLM on Software Vulnerability. **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10706805)] [[Code](https://github.com/vinci-grape/VulEmpirical)]
- Bridging Semantics \& Structure for Software Vulnerability Detection using Hybrid Network Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.10321)] [[Code](https://zenodo.org/records/17259519)]
- The Richer Representation Fallacy: Are We Just Adding Noise to LLM-based Software Vulnerability Detectors?. **`ICOCO 2025`** [[Paper](https://ieeexplore.ieee.org/document/11334069)]
- Transfer-Guided Konwledge Distillation for Enhancing Cross-Project Vulnerability Detection. **`CCNS 2025`** [[Paper](https://ieeexplore.ieee.org/document/11337967)]
- Sparse-MoE: Syntax-Aware Multi-view Mixture of Experts for Long-Sequence Software Vulnerability Detection. **`ADMA 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-981-95-3456-2_24)]
- CTVD: Collaborative Training of Deep Learning and Large Model for C/C++ Source Code Vulnerability Detection. **`SMC 2025`** [[Paper](https://ieeexplore.ieee.org/document/11343541)]

<a name="instruction-tuning"></a>
###### Instruction-Tuning
- Your Instructions Are Not Always Helpful: Assessing the Efficacy of Instruction Fine-tuning for Software Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.07466)]
- Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning. **`ACL 2024`** [[Paper](https://arxiv.org/abs/2406.03718)] [[Code](https://github.com/CGCL-codes/VulLLM)]
- Enhancing Source Code Security with LLMs: Demystifying The Challenges and Generating Reliable Repairs. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.00571)]
- Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.18260)] [[Code](https://github.com/SakiRinn/LLM4CVD)] [[Code](https://huggingface.co/datasets/xuefen/VulResource)]
- To Err is Machine: Vulnerability Detection Challenges LLM Reasoning. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- Case Study: Fine-tuning Small Language Models for Accurate and Private CWE Detection in Python Code. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.16584)] [[Code](https://huggingface.co/floxihunter/codegen-mono-CWEdetect)] [[Code](https://huggingface.co/datasets/floxihunter/synthetic_python_cwe)]
- Large Language Models for Multilingual Vulnerability Detection: How Far Are We?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07503)] [[Code](https://github.com/SpanShu96/Large-Language-Model-for-Multilingual-Vulnerability-Detection/tree/main)]
- VulnTeam: A Team Collaboration Framework for LLM-based Vulnerability Detection. **`IJCNN 2025`** [[Paper](https://ieeexplore.ieee.org/document/11229292)]
- Towards Explainable Vulnerability Detection With Large Language Models. **`TSE 2024`** [[Paper](https://arxiv.org/abs/2406.09701)]
- SQL Injection Vulnerability Detection Based on Pissa-Tuned Llama 3 Large Language Model. **`ICFTIC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10912886)]
- A Method of SQL Injection Attack Detection Based on Large Language Models. **`CNTEIE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10987904)]
- On the Effectiveness of Instruction-Tuning Local LLMs for Identifying Software Vulnerabilities. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.20062)]
- Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.01449)] [[Code](https://github.com/soarsmu/SVD-Bench)]
- Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10961)] [[Code](https://figshare.com/s/1514bc9a7aa64b46d94e)]

<a name="t2222-parameter-efficient-fine-tuning-peft"></a>
###### T2.2.2.2 Parameter-Efficient Fine-Tuning (PEFT)
<a name="t22221-selective"></a>
###### T2.2.2.2.1 Selective
<a name="selective"></a>
###### Selective
- Improving Vulnerability Type Prediction and Line-Level Detection via Adversarial Training-based Data Augmentation and Multi-Task Learning. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.23534)] [[Code](https://github.com/Karelye/EDAT-MLT)]
- VulTrLM: LLM-assisted Vulnerability Detection via AST Decomposition and Comment Enhancement. **`EMSE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10664-025-10738-7)]
- VLD-LP: Vulnerability Detection and Root Cause Localization with Large Language Model and Parameter-efficient Language Model Tuning. **`SMC 2025`** [[Paper](https://ieeexplore.ieee.org/document/11343151)]

<a name="t22222-additive"></a>
###### T2.2.2.2.2 Additive
<a name="adapter-tuning"></a>
###### Adapter-Tuning
- CLeVeR: Multi-modal Contrastive Learning for Vulnerability Code Representation. **`ACL 2025`** [[Paper](https://aclanthology.org/2025.findings-acl.414/)] [[Code](https://github.com/yoimiya-nlp/CLeVeR)]
- AutoAdapt: On the Application of AutoML for Parameter-Efficient Fine-Tuning of Pre-Trained Code Models. **`TOSEM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3734867)] [[Code](https://github.com/serval-uni-lu/AutoAdapt)]

<a name="prompt-tuning"></a>
###### Prompt-Tuning
- ProRLearn: Boosting Prompt Tuning-based Vulnerability Detection by Reinforcement Learning. **`ASE 2024`** [[Paper](https://link.springer.com/article/10.1007/s10515-024-00438-9)] [[Code](https://github.com/ProRLearn/ProRLearn001)]
- CGP-Tuning: Structure-Aware Soft Prompt Tuning for Code Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.04510)]
- VulPr: A Prompt Learning-based Method for Vulnerability Detection. **`EIT 2025`** [[Paper](https://ieeexplore.ieee.org/document/11231886)]
- Assessing the Effectiveness of Vulnerability Detection via Prompt Tuning: An Empirical Study. **`APSEC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10479384)] [[Code](https://github.com/P-E-Vul/prompt-empircial-vulnerability)]
- Fine-Tuning Pre-trained Model with Optimizable Prompt Learning for Code Vulnerability Detection. **`ISSRE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10771498)] [[Code](https://github.com/Exclusisve-V/PromptVulnerabilityDetection)]

<a name="additive-other"></a>
###### Additive-Other
- One Model, Many Skills: Parameter-Efficient Fine-Tuning for Multitask Code Analysis. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2603.09978)] [[Code](https://github.com/Amal-AK/multitask_PEFT)]
- Steering Large Language Models for Vulnerability Detection. **`ICASSP 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10887736)]

<a name="t22223-re-parameterized"></a>
###### T2.2.2.2.3 Re-parameterized
<a name="low-rank-decomposition"></a>
###### Low-Rank Decomposition
- Your Instructions Are Not Always Helpful: Assessing the Efficacy of Instruction Fine-tuning for Software Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.07466)]
- SCL-CVD: Supervised Contrastive Learning for Code Vulnerability Detection via GraphCodeBERT. **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824002992)]
- Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.16235)]
- Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning. **`ACL 2024`** [[Paper](https://arxiv.org/abs/2406.03718)] [[Code](https://github.com/CGCL-codes/VulLLM)]
- Can a Llama Be a Watchdog? Exploring Llama 3 and Code Llama for Static Application Security Testing. **`CSR 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10679444)]
- Outside the Comfort Zone: Analysing LLM Capabilities in Software Vulnerability Detection. **`ESORICS 2024`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70879-4_14)]
- RealVul: Can We Detect Vulnerabilities in Web Applications with LLM?. **`EMNLP 2024`** [[Paper](https://arxiv.org/abs/2410.07573)]
- Enhancing Vulnerability Detection Efficiency: An Exploration of Light-weight LLMs with Hybrid Code Features. **`JISA 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212624002278)] [[Code](https://github.com/JNL-28/Enhancing-Vulnerability-Detection-Efficiency)]
- Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.18260)] [[Code](https://github.com/SakiRinn/LLM4CVD)] [[Code](https://huggingface.co/datasets/xuefen/VulResource)]
- Sink Vulnerability Type Prediction Using Small Language Model (SLM). **`IC3ECSBHI 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10991300)]
- Finetuning Large Language Models for Vulnerability Detection. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908394)] [[Code](https://github.com/rmusab/vul-llm-finetune)]
- R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.04699)] [[Code](https://github.com/martin-wey/R2Vul)]
- Evaluating LLaMA 3.2 for Software Vulnerability Detection. **`EICC 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-94855-8_3)]
- Large Language Models for Multilingual Vulnerability Detection: How Far Are We?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07503)] [[Code](https://github.com/SpanShu96/Large-Language-Model-for-Multilingual-Vulnerability-Detection/tree/main)]
- Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07390)] [[Code](https://github.com/Xin-Cheng-Wen/PO4Vul)]
- VulnTeam: A Team Collaboration Framework for LLM-based Vulnerability Detection. **`IJCNN 2025`** [[Paper](https://ieeexplore.ieee.org/document/11229292)]
- LLMxCPG: Context-Aware Vulnerability Detection Through Code Property Graph-Guided Large Language Models. **`USENIX Security 2025`** [[Paper](https://arxiv.org/abs/2507.16585)] [[Code](https://github.com/qcri/llmxcpg)] [[Code](https://zenodo.org/records/15614095)]
- Revisiting Pre-trained Language Models for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.16887)]
- MalCodeAI: Autonomous Vulnerability Detection and Remediation via Language Agnostic Code Reasoning. **`IRI 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11153184)]
- Optimizing Code Vulnerability Detection via GRPO and SFT Fine-Tuning of Compact LLMs. **`DSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11360339)]
- VulPr: A Prompt Learning-based Method for Vulnerability Detection. **`EIT 2025`** [[Paper](https://ieeexplore.ieee.org/document/11231886)]
- An Advanced Detection Framework for Embedded System Vulnerabilities. **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11153853)]
- Utilizing Large Programming Language Models on Software Vulnerability Detection. **`ASYU 2025`** [[Paper](https://ieeexplore.ieee.org/document/11208282)]
- Towards Explainable Vulnerability Detection With Large Language Models. **`TSE 2024`** [[Paper](https://arxiv.org/abs/2406.09701)]
- The Semantic Trap: Do Fine-tuned LLMs Learn Vulnerability Root Cause or Just Functional Pattern?. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.22655)] [[Code](https://anonymous.4open.science/r/TrapEval)]
- LLMs in Code Vulnerability Analysis: A Proof of Concept. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.08691)] [[Code](https://figshare.com/s/a06ec09cd1bd98e6dd45)]
- One Model, Many Skills: Parameter-Efficient Fine-Tuning for Multitask Code Analysis. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2603.09978)] [[Code](https://github.com/Amal-AK/multitask_PEFT)]
- Beyond Function-Level Analysis: Context-Aware Reasoning for Inter-Procedural Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06751v1)] [[Code](https://github.com/yikun-li/CPRVul)]
- Enhancing Continual Learning for Software Vulnerability Prediction: Addressing Catastrophic Forgetting via Hybrid-Confidence-Aware Selective Replay for Temporal LLM Fine-Tuning. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.23834v1)]
- A Method of SQL Injection Attack Detection Based on Large Language Models. **`CNTEIE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10987904)]
- HALURust: Exploiting Hallucinations of Large Language Models to Detect Vulnerabilities in Rust. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.10793)]
- LLaVul: A Multimodal LLM for Interpretable Vulnerability Reasoning about Source Code. **`ICSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11140501)]
- Detecting Source Code Vulnerabilities Using Fine-Tuned Pre-Trained LLMs. **`ICSP 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10846595)]
- A Comprehensive Comparison of LLaMA 3.1 and Traditional ML Approaches in Automated Vulnerability Detection. **`AICCSA 2025`** [[Paper](https://ieeexplore.ieee.org/document/11315404)]

<a name="lora-derivates"></a>
###### LoRA Derivates
- Security Vulnerability Detection with Multitask Self-Instructed Fine-Tuning of Large Language Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.05892)] [[Code](https://zenodo.org/records/11403208)]
- Code Vulnerability Detection: A Comparative Analysis of Emerging Large Language Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.10490)]
- LPASS: Linear Probes as Stepping Stones for Vulnerability Detection using Compressed LLMs. **`JISA 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212625001620)]
- Ensembling Large Language Models for Code Vulnerability Detection: An Empirical Evaluation. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2509.12629)] [[Code](https://github.com/sssszh/ELVul4LLM)]
- An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems. **`CASCON 2025`** [[Paper](https://ieeexplore.ieee.org/document/11344502)]
- VulReaD: Knowledge-Graph-guided Software Vulnerability Reasoning and Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.10787v1)] [[Code](https://anonymous.4open.science/r/Vul-ReaD)]
- SQL Injection Vulnerability Detection Based on Pissa-Tuned Llama 3 Large Language Model. **`ICFTIC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10912886)]
- Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.01449)] [[Code](https://github.com/soarsmu/SVD-Bench)]
- Llama-Based Source Code Vulnerability Detection: Prompt Engineering vs Fine Tuning. **`ESORICS 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-032-07884-1_15)] [[Code](https://github.com/DynaSoumhaneOuchebara/Llama-based-vulnerability-detection)]
- Distilling Lightweight Language Models for C/C++ Vulnerabilities. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.06645)] [[Code](https://github.com/yangxiaoxuan123/FineSec_detect)]

<a name="t23-learning-paradigms"></a>
#### T2.3 Learning Paradigms
<a name="contrastive-learning"></a>
##### Contrastive Learning
- Multi-view Pre-trained Model for Code Vulnerability Identification. **`WASA 2022`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19211-1_11)]
- Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation. **`ESEC/FSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3611643.3616358)] [[Code](https://github.com/jacknichao/SVulD)]
- SCL-CVD: Supervised Contrastive Learning for Code Vulnerability Detection via GraphCodeBERT. **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824002992)]
- Applying Contrastive Learning to Code Vulnerability Type Classification. **`EMNLP 2024`** [[Paper](https://aclanthology.org/2024.emnlp-main.666/)]
- Vulnerability Detection with Code Language Models: How Far are We?. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029911)] [[Code](https://github.com/DLVulDet/PrimeVul)]
- VuL-MCBERT: A Vulnerability Detection Method Based on Self-Supervised Contrastive Learning. **`CAIBDA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11183103)]
- CLeVeR: Multi-modal Contrastive Learning for Vulnerability Code Representation. **`ACL 2025`** [[Paper](https://aclanthology.org/2025.findings-acl.414/)] [[Code](https://github.com/yoimiya-nlp/CLeVeR)]
- Joint Geometrical and Statistical Domain Adaptation for Cross-domain  Code Vulnerability Detection. **`EMNLP 2023`** [[Paper](https://aclanthology.org/2023.emnlp-main.788/)]
- Bridging Semantics \& Structure for Software Vulnerability Detection using Hybrid Network Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.10321)] [[Code](https://zenodo.org/records/17259519)]
- Sparse-MoE: Syntax-Aware Multi-view Mixture of Experts for Long-Sequence Software Vulnerability Detection. **`ADMA 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-981-95-3456-2_24)]

<a name="causal-learning"></a>
##### Causal Learning
- Towards Causal Deep Learning for Vulnerability Detection. **`ICSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3597503.3639170)] [[Code](https://figshare.com/s/0ffda320dcb96c249ef2?file=41801019)]

<a name="multi-task-learning"></a>
##### Multi-Task Learning
- An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph. **`EuroS&P 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10190505)] [[Code](https://github.com/pial08/SemVulDet)]
- AIBugHunter: A Practical Tool for Predicting, Classifying and Repairing Software Vulnerabilities. **`EMSE 2023`** [[Paper](https://link.springer.com/article/10.1007/s10664-023-10346-3)] [[Code](https://github.com/awsm-research/AIBugHunter)]
- TRACED: Execution-aware Pre-training for Source Code. **`ICSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3597503.3608140)] [[Code](https://github.com/ARiSE-Lab/TRACED_ICSE_24)]
- Security Vulnerability Detection with Multitask Self-Instructed Fine-Tuning of Large Language Models. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.05892)] [[Code](https://zenodo.org/records/11403208)]
- MultiVD: A Transformer-based Multitask Approach for Software Vulnerability Detection. **`SECRYPT 2024`** [[Paper](https://www.scitepress.org/Papers/2024/127194/127194.pdf)]
- Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning. **`ACL 2024`** [[Paper](https://arxiv.org/abs/2406.03718)] [[Code](https://github.com/CGCL-codes/VulLLM)]
- Leveraging Multi-Task Learning to Improve the Detection of SATD and Vulnerability. **`ICPC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11025930)] [[Code](https://github.com/moritzmock/multitask-vulberability-detection)]
- Closing the Gap: A User Study on the Real-world Usefulness of AI-powered Vulnerability Detection \& Repair in the IDE. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029760)] [[Code](https://figshare.com/articles/dataset/Closing_the_Gap_A_User_Study_on_the_Real-world_Usefulness_of_AI-powered_Vulnerability_Detection_Repair_in_the_IDE/26367139?file=52478936)]
- Improving Vulnerability Type Prediction and Line-Level Detection via Adversarial Training-based Data Augmentation and Multi-Task Learning. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.23534)] [[Code](https://github.com/Karelye/EDAT-MLT)]
- An Advanced Detection Framework for Embedded System Vulnerabilities. **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11153853)]
- Transformer-Based Semantic Embeddings and Hybrid Neural Networks for Robust Software Vulnerability Detection. **`i-PACT 2025`** [[Paper](https://ieeexplore.ieee.org/document/11307989)]
- One Model, Many Skills: Parameter-Efficient Fine-Tuning for Multitask Code Analysis. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2603.09978)] [[Code](https://github.com/Amal-AK/multitask_PEFT)]
- The Richer Representation Fallacy: Are We Just Adding Noise to LLM-based Software Vulnerability Detectors?. **`ICOCO 2025`** [[Paper](https://ieeexplore.ieee.org/document/11334069)]

<a name="knowledge-distillation"></a>
##### Knowledge Distillation
- Distilled and Contextualized Neural Models Benchmarked for Vulnerable Function Detection. **`Mathematics 2022`** [[Paper](https://www.mdpi.com/2227-7390/10/23/4482)]
- VulExplainer: A Transformer-Based Hierarchical  Distillation for Explaining Vulnerability Types. **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10220166)] [[Code](https://github.com/awsm-research/VulExplainer)]
- Greening Large Language Models of Code. **`ICSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3639475.3640097)] [[Code](https://github.com/soarsmu/Avatar)]
- Enhancing Vulnerability Detection Efficiency: An Exploration of Light-weight LLMs with Hybrid Code Features. **`JISA 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212624002278)] [[Code](https://github.com/JNL-28/Enhancing-Vulnerability-Detection-Efficiency)]
- Metamorphic-Based Many-Objective Distillation of LLMs for Code-Related Tasks. **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11029766)] [[Code](https://zenodo.org/records/14857610)]
- R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.04699)] [[Code](https://github.com/martin-wey/R2Vul)]
- An Automatic Classification Model for Long Code Vulnerabilities Based on the Teacher-Student Framework. **`QRS 2025`** [[Paper](https://ieeexplore.ieee.org/document/11216609)]
- SAFE: A Novel Approach For Software Vulnerability Detection from Enhancing The Capability of Large Language Models. **`ASIACCS 2024`** [[Paper](https://arxiv.org/abs/2409.00882)]
- PIONEER: Improving the Robustness of Student Models when Compressing Pre-Trained Models of Code. **`ASE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10515-025-00560-2)] [[Code](https://github.com/illsui1on/PIONEER)]
- VulReaD: Knowledge-Graph-guided Software Vulnerability Reasoning and Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.10787v1)] [[Code](https://anonymous.4open.science/r/Vul-ReaD)]
- On the Compression of Language Models for Code: An Empirical Study on CodeBERT. **`SANER 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10992473)] [[Code](https://zenodo.org/records/14357478)]
- VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.07533)] [[Code](https://github.com/ucsb-mlsec/VulnLLM-R)]
- VulDefend: A Novel Technique based on Pattern-exploiting Training for Detecting Software Vulnerabilities Using Language Models. **`JEEIT 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10185860)]
- Vul-LMGNNs: Fusing Language Models and Online-distilled Graph Neural Networks for Code Vulnerability Detection. **`Information Fusion 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253524005268)] [[Code](https://github.com/Vul-LMGNN/vul-LMGGNN)]
- Transfer-Guided Konwledge Distillation for Enhancing Cross-Project Vulnerability Detection. **`CCNS 2025`** [[Paper](https://ieeexplore.ieee.org/document/11337967)]
- Distilling Lightweight Language Models for C/C++ Vulnerabilities. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.06645)] [[Code](https://github.com/yangxiaoxuan123/FineSec_detect)]

<a name="continual-learning"></a>
##### Continual Learning
- Enhancing Continual Learning for Software Vulnerability Prediction: Addressing Catastrophic Forgetting via Hybrid-Confidence-Aware Selective Replay for Temporal LLM Fine-Tuning. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.23834v1)]
- SQL Injection Vulnerability Detection Based on Pissa-Tuned Llama 3 Large Language Model. **`ICFTIC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10912886)]
- MVD: A Multi-Lingual Software Vulnerability Detection Framework. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.06166)] [[Code](https://figshare.com/s/10ec70108294a225f391)]
- Keeping Pace with Ever-Increasing Data: Towards Continual Learning of Code Intelligence Models. **`ICSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10172346)] [[Code](https://github.com/ReliableCoding/REPEAT)]
- Distilling Lightweight Language Models for C/C++ Vulnerabilities. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.06645)] [[Code](https://github.com/yangxiaoxuan123/FineSec_detect)]

<a name="reinforcement-learning"></a>
##### Reinforcement Learning
- ProRLearn: Boosting Prompt Tuning-based Vulnerability Detection by Reinforcement Learning. **`ASE 2024`** [[Paper](https://link.springer.com/article/10.1007/s10515-024-00438-9)] [[Code](https://github.com/ProRLearn/ProRLearn001)]
- R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.04699)] [[Code](https://github.com/martin-wey/R2Vul)]
- Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2507.03051)]
- Enhancing Fine-Grained Vulnerability Detection With Reinforcement Learning. **`TSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11145224)] [[Code](https://github.com/YuanJiangGit/RLFD)]
- Optimizing Code Vulnerability Detection via GRPO and SFT Fine-Tuning of Compact LLMs. **`DSC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11360339)]
- Evaluating and Enhancing the Vulnerability Reasoning Capabilities of Large Language Models. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.06687v1)]
- From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2602.14012v1)] [[Code](https://github.com/youpengl/OpenVul)]
- Adversarial Training for Robustness Enhancement in LLM-Based Code Vulnerability Detection. **`CISCE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11065803)]

<a name="other-data-centric"></a>
##### Other Data-Centric
- When Less is Enough: Positive and Unlabeled Learning Model for Vulnerability Detection. **`ASE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10298363)] [[Code](https://github.com/PILOT-VD-2023/PILOT)]
- Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning. **`ACL 2024`** [[Paper](https://arxiv.org/abs/2406.03718)] [[Code](https://github.com/CGCL-codes/VulLLM)]
- Improving Vulnerability Type Prediction and Line-Level Detection via Adversarial Training-based Data Augmentation and Multi-Task Learning. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.23534)] [[Code](https://github.com/Karelye/EDAT-MLT)]
- Smart Cuts: Enhance Active Learning for Vulnerability Detection by Pruning Bad Seeds. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.20444)]
- PIONEER: Improving the Robustness of Student Models when Compressing Pre-Trained Models of Code. **`ASE 2025`** [[Paper](https://link.springer.com/article/10.1007/s10515-025-00560-2)] [[Code](https://github.com/illsui1on/PIONEER)]
- Leveraging Self-Paced Learning for Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2511.09212)] [[Code](https://figshare.com/s/bef3211194fc18fe375e)]
- PATVD: Vulnerability Detection Based on Pre-training Techniques and Adversarial Training. **`SmartWorld/UIC/ScalCom/DigitalTwin/PriComp/Meta 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10189687/)]
- A Source Code Vulnerability Detection Method Based on Positive-Unlabeled Learning. **`RICAI 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10911761)]
- Learning Defect Prediction from Unrealistic Data. **`SANER 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10589866)] [[Code](https://zenodo.org/records/10514652)]
- Adversarial Training for Robustness Enhancement in LLM-Based Code Vulnerability Detection. **`CISCE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11065803)]

<a name="t3-orchestration"></a>
### T3 Orchestration
<a name="multi-step"></a>
#### Multi-Step
- Software Vulnerability Detection with GPT and In-Context Learning. **`DSC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10381286)]
- Exploration On Prompting LLM With Code-Specific Information For Vulnerability Detection. **`SSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10664399)]
- May the Source Be with You: On ChatGPT, Cybersecurity, and Secure Coding. **`Information 2024`** [[Paper](https://www.mdpi.com/2078-2489/15/9/572)]
- Vulnerability Detection with Code Language Models: How Far are We?. **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029911)] [[Code](https://github.com/DLVulDet/PrimeVul)]
- Context-Enhanced Vulnerability Detection Based on Large Language Models. **`TOSEM 2025`** [[Paper](https://arxiv.org/abs/2504.16877)] [[Code](https://github.com/DoeSEResearch/PacVD)]
- Detecting Code Vulnerabilities using LLMs. **`DSN 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11068842)] [[Code](https://github.com/a24167566/LLMs-Code-Vulnerability-Detection)]
- SecureMind: A Framework for Benchmarking Large Language Models in Memory Bug Detection and Repair. **`ISMM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3735950.3735954)] [[Code](https://github.com/HuantWang/SecureMind)]
- An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.16310)] [[Code](https://zenodo.org/records/15572151)]
- LLM-GUARD: Large Language Model-Based Detection and Repair of Bugs and Security Vulnerabilities in C++ and Python. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.16419)] [[Code](https://github.com/NoujoudNader/LLM-Bugs-Detection)]
- DeepVulHunter: Enhancing the Code Vulnerability Detection Capability of LLMs through Multi-Round Analysis. **`JIIS 2025`** [[Paper](https://link.springer.com/article/10.1007/s10844-025-00982-0)]
- iCodeReviewer: Improving Secure Code Review with Mixture of Prompts. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.12186)]
- A Sequential Multi-Stage Approach for Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making. **`EMNLP 2025`** [[Paper](https://aclanthology.org/2025.emnlp-main.1071/)]

<a name="verification"></a>
#### Verification
- VulnGPT: Enhancing Source Code Vulnerability Detection Using AutoGPT and Adaptive Supervision Strategies. **`DCOSS-IoT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10621527)]
- M2CVD: Enhancing Vulnerability Understanding through Multi-Model Collaboration for Code Vulnerability Detection. **`TOSEM 2024`** [[Paper](https://arxiv.org/abs/2406.05940)] [[Code](https://github.com/HotFrom/M2CVD)]
- Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning. **`ACL 2024`** [[Paper](https://arxiv.org/abs/2406.03718)] [[Code](https://github.com/CGCL-codes/VulLLM)]
- Navigating (In)Security of AI-Generated Code. **`CSR 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10679468)]
- Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.01001)] [[Code](https://figshare.com/s/5da14b0776750c6fa787)]
- Harnessing Large Language Models for Software Vulnerability Detection: A Comprehensive Benchmarking Study. **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10879492)]
- Beyond Static Pattern Matching? Rethinking Automatic Cryptographic API Misuse Detection in the Era of LLMs. **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3728875)]
- Think Broad, Act Narrow: CWE Identification with Multi-Agent Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.01451)] [[Code](https://zenodo.org/records/15871507)]
- Leveraging Intra-and Inter-References in Vulnerability Detection using Multi-Agent Collaboration Based on LLMs. **`Cluster Computing 2025`** [[Paper](https://link.springer.com/article/10.1007/s10586-025-05721-2)]
- LLM-CloudSec: Large Language Model Empowered Automatic and Deep Vulnerability Analysis for Intelligent Clouds. **`INFOCOM 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10620804)] [[Code](https://github.com/DPCa0/LLM-CloudSec)]

<a name="agentic"></a>
#### Agentic
- VulnGPT: Enhancing Source Code Vulnerability Detection Using AutoGPT and Adaptive Supervision Strategies. **`DCOSS-IoT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10621527)]
- Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories. **`Unknown 2025`** [[Paper](https://arxiv.org/abs/2503.03586)]
- Think Broad, Act Narrow: CWE Identification with Multi-Agent Large Language Models. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2508.01451)] [[Code](https://zenodo.org/records/15871507)]
- MAVUL: Multi-Agent Vulnerability Detection via Contextual Reasoning and Interactive Refinement. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.00317)] [[Code](https://github.com/youpengl/MAVUL)]
- VulAgent: Hypothesis-Validation based Multi-Agent Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2509.11523)]
- Leveraging Intra-and Inter-References in Vulnerability Detection using Multi-Agent Collaboration Based on LLMs. **`Cluster Computing 2025`** [[Paper](https://link.springer.com/article/10.1007/s10586-025-05721-2)]
- An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems. **`CASCON 2025`** [[Paper](https://ieeexplore.ieee.org/document/11344502)]
- A Sequential Multi-Stage Approach for Code Vulnerability Detection via Confidence- and Collaboration-based Decision Making. **`EMNLP 2025`** [[Paper](https://aclanthology.org/2025.emnlp-main.1071/)]
- MulVul: Retrieval-augmented Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.18847)]
- VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.07533)] [[Code](https://github.com/ucsb-mlsec/VulnLLM-R)]
- LLM-CloudSec: Large Language Model Empowered Automatic and Deep Vulnerability Analysis for Intelligent Clouds. **`INFOCOM 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10620804)] [[Code](https://github.com/DPCa0/LLM-CloudSec)]
- SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.19828)] [[Code](https://github.com/basimbd/SecVulEval)]
- Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10961)] [[Code](https://figshare.com/s/1514bc9a7aa64b46d94e)]

<a name="ensemble"></a>
#### Ensemble
- An Enhanced Vulnerability Detection in Software Using a Heterogeneous Encoding Ensemble. **`ISCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10217978)]
- Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.16235)]
- VulSim: Leveraging Similarity of Multi-Dimensional Neighbor Embeddings for Vulnerability Detection. **`USENIX Security 2024`** [[Paper](https://www.usenix.org/conference/usenixsecurity24/presentation/shimmi)] [[Code](https://github.com/SamihaShimmi/VulSim)]
- Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques. **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.01001)] [[Code](https://figshare.com/s/5da14b0776750c6fa787)]
- DMVL4AVD: A Deep Multi-View Learning Model for Automated Vulnerability Detection. **`Neural Comput. Appl. 2025`** [[Paper](https://link.springer.com/article/10.1007/s00521-024-10892-x)] [[Code](https://drive.google.com/file/d/1-qWqmRuBi8kRAAE2yiG6JNiY8vLYxXlz/view)]
- An Ensemble Transformer Approach with Cross-Attention for Automated Code Security Vulnerability Detection and Documentation. **`ISDFS 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11012039)]
- EnStack: An Ensemble Stacking Framework of Large Language Models for Enhanced Vulnerability Detection in Source Code. **`BigData 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10825609)]
- Diverse LLMs vs. Vulnerabilities: Who Detects and Fixes Them Better?. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2512.12536)] [[Code](https://github.com/Erroristotle/DVDR_LLM)]
- You Only Train Once: A Flexible Training Framework for Code Vulnerability Detection Driven by Vul-Vector. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10988)]
- Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.01449)] [[Code](https://github.com/soarsmu/SVD-Bench)]
- VulD-CodeBERT: CodeBERT-Based Vulnerability Detection Model for C/C++ Code. **`CISCE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10653337)]
- Sparse-MoE: Syntax-Aware Multi-view Mixture of Experts for Long-Sequence Software Vulnerability Detection. **`ADMA 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-981-95-3456-2_24)]

<a name="controller"></a>
#### Controller
- Expert-in-the-Loop Systems with Cross-Domain and In-Domain Few-Shot Learning for Software Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10104)]
- One-for-All Does Not Work! Enhancing Vulnerability Detection by Mixture-of-Experts (MoE). **`PACMSE 2025`** [[Paper](https://arxiv.org/abs/2501.16454)]
- VulnTeam: A Team Collaboration Framework for LLM-based Vulnerability Detection. **`IJCNN 2025`** [[Paper](https://ieeexplore.ieee.org/document/11229292)]
- VulAgent: Hypothesis-Validation based Multi-Agent Vulnerability Detection. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2509.11523)]
- iCodeReviewer: Improving Secure Code Review with Mixture of Prompts. **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2510.12186)]
- MulVul: Retrieval-augmented Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution. **`arXiv 2026`** [[Paper](https://arxiv.org/abs/2601.18847)]
- Sparse-MoE: Syntax-Aware Multi-view Mixture of Experts for Long-Sequence Software Vulnerability Detection. **`ADMA 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-981-95-3456-2_24)]
