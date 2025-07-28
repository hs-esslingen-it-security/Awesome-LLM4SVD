# Awesome-LLM4SVD 🌟-🧠👩‍💻🔍

This repository contains the artifacts from the systematic literature review (SLR) on LLM-based software vulnerability detection ("A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models"). 
The SLR analyzes 227 studies published between January 2020 and June 2025 and provides a structured taxonomy of detection approaches, input representations, system architectures, adaptation techniques, and dataset usage.

To support reproducibility and structured comparison, we publicly release:
- A curated list of surveyed **papers**, along with their categorization can be found [here](https://github.com/hs-esslingen-it-security/Awesome-LLM4SVD/tree/main/taxonomy). This README will be continuously updated to track the latest papers.
- A list of the most commonly used **datasets**, including download sources

<br>

For details, see our publication: 

📚 S. Kaniewski, F. Schmidt, M. Enzweiler, M. Menth, und T. Heer, „A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models“, 2025.
```bibtex
@preprint{kaniewskiSLR-LLM4SVD2025,
    title={{A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models}}, 
    author={Kaniewski, Sabrina and Schmidt, Fabian and Enzweiler, Markus and Menth, Michael and Heer, Tobias},
    year={2025},
    note={tbd}
}
```




## Table of Contents

- 📝 [Surveyed Papers](#papers)
- 📝 [Selected Datasets](#datasets)
- 🤝 [Contribute to this repository](#contribution)
- ⚖️ [License](#license)


## Papers

### 2025
- Improving Vulnerability Type Prediction and Line-Level Detection via Adversarial Training-based Data Augmentation and Multi-Task Learning.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.23534)] [[Code](https://github.com/Karelye/EDAT-MLT)]
- Smart Cuts: Enhance Active Learning for Vulnerability Detection by Pruning Bad Seeds.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.20444)]
- FuncVul: An Effective Function Level Vulnerability Detection Model using LLM and Code Chunk.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.19453)] [[Code](https://github.com/sajalhalder/FuncVul)]
- Detecting Code Vulnerabilities using LLMs.  **`DSN 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11068842)] [[Code](https://github.com/a24167566/LLMs-Code-Vulnerability-Detection)]
- Beyond Static Pattern Matching? Rethinking Automatic Cryptographic API Misuse Detection in the Era of LLMs.  **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3728875)]
- LPASS: Linear Probes as Stepping Stones for Vulnerability Detection using Compressed LLMs.  **`JISA 2025`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212625001620)]
- SAVANT: Vulnerability Detection in Application Dependencies through Semantic-Guided Reachability Analysis.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.17798)]
- SafeGenBench: A Benchmark Framework for Security Vulnerability Detection in LLM-Generated Code.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.05692)]
- Large Language Models for In-File Vulnerability Localization Can Be "Lost in the End".  **`PACMSE 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3715758)] [[Code](https://zenodo.org/records/14840519)]
- Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2406.11147)] [[Code](https://github.com/knowledgerag4llmvuld/knowledgerag4llmvuld)]
- Evaluating LLaMA 3.2 for Software Vulnerability Detection.  **`EICC 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-94855-8_3)]
- How Well Do Large Language Models Serve as End-to-End Secure Code Agents for Python?.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2408.10495)] [[Code](https://github.com/jianian0318/LLMSecureCode)]
- SecureMind: A Framework for Benchmarking Large Language Models in Memory Bug Detection and Repair.  **`ISMM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3735950.3735954)] [[Code](https://github.com/HuantWang/SecureMind)]
- Expert-in-the-Loop Systems with Cross-Domain and In-Domain Few-Shot Learning for Software Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10104)]
- Large Language Models for Multilingual Vulnerability Detection: How Far Are We?.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07503)] [[Code](https://github.com/SpanShu96/Large-Language-Model-for-Multilingual-Vulnerability-Detection/tree/main)]
- Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.07390)] [[Code](https://github.com/Xin-Cheng-Wen/PO4Vul)]
- CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2411.17274)] [[Code](https://github.com/yikun-li/CleanVul)]
- LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2401.16185)] [[Code](https://anonymous.4open.science/r/LLM4Vuln/README.md)]
- An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2401.16310)] [[Code](https://zenodo.org/records/15572151)]
- ANVIL: Anomaly-based Vulnerability Identification without Labelled Training Data.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2408.16028)] [[Code](https://anonymous.4open.science/r/anvil)]
- RepoAudit: An Autonomous LLM-Agent for Repository-Level Code Auditing.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.18160)] [[Code](https://github.com/PurCL/RepoAudit)]
- SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.19828)] [[Code](https://github.com/basimbd/SecVulEval)]
- VADER: A Human-Evaluated Benchmark for Vulnerability Assessment, Detection, Explanation, and Remediation.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.19395)] [[Code](https://github.com/AfterQuery/vader)]
- Learning to Focus: Context Extraction for Efficient Code Vulnerability Detection with Language Models.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.17460)]
- Are Sparse Autoencoders Useful for Java Function Bug Detection?.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10375)]
- Leveraging Large Language Models for Command Injection Vulnerability Analysis in Python: An Empirical Study on Popular Open-Source Projects.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.15088)]
- Code Vulnerability Repair with Large Language Model Using Context-Aware Prompt Tuning.  **`SPW 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11050839)]
- Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10494)] [[Code](https://github.com/MurrayTom/CoV-Eval)]
- SV-TrustEval-C: Evaluating Structure and Semantic Reasoning in Large Language Models for Source Code Vulnerability Analysis.  **`SP 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11023455)] [[Code](https://github.com/Jackline97/SV-TrustEval-C)] [[Code](https://huggingface.co/datasets/LLMs4CodeSecurity/SV-TrustEval-C-1.0)]
- AutoAdapt: On the Application of AutoML for Parameter-Efficient Fine-Tuning of Pre-Trained Code Models.  **`TOSEM 2025`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3734867)] [[Code](https://github.com/serval-uni-lu/AutoAdapt)]
- Adversarial Training for Robustness Enhancement in LLM-Based Code Vulnerability Detection.  **`CISCE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11065803)]
- AutoPatch: Multi-Agent Framework for Patching Real-World CVE Vulnerabilities.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.04195)] [[Code](https://github.com/ai-llm-research/autopatch)]
- Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2505.10961)] [[Code](https://figshare.com/s/1514bc9a7aa64b46d94e)]
- GraphCodeBERT-Augmented Graph Attention Networks for Code Vulnerability Detection.  **`CAI 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11050748)]
- Automating the Detection of Code Vulnerabilities by Analyzing GitHub Issues.  **`LLM4Code 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028308)]
- ♪ With a Little Help from My (LLM) Friends: Enhancing Static Analysis with LLMs to Detect Software Vulnerabilities.  **`LLM4Code 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11028575)]
- Leveraging Multi-Task Learning to Improve the Detection of SATD and Vulnerability.  **`ICPC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11025930)] [[Code](https://github.com/moritzmock/multitask-vulberability-detection)]
- Vulnerability Detection with Code Language Models: How Far are We?.  **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029911)] [[Code](https://github.com/DLVulDet/PrimeVul)]
- Metamorphic-Based Many-Objective Distillation of LLMs for Code-Related Tasks.  **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/document/11029766)] [[Code](https://zenodo.org/records/14857610)]
- Closing the Gap: A User Study on the Real-world Usefulness of AI-powered Vulnerability Detection & Repair in the IDE.  **`ICSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11029760)] [[Code](https://figshare.com/articles/dataset/Closing_the_Gap_A_User_Study_on_the_Real-world_Usefulness_of_AI-powered_Vulnerability_Detection_Repair_in_the_IDE/26367139?file=52478936)]
- An Ensemble Transformer Approach with Cross-Attention for Automated Code Security Vulnerability Detection and Documentation.  **`ISDFS 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11012039)]
- Case Study: Fine-tuning Small Language Models for Accurate and Private CWE Detection in Python Code.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.16584)] [[Code](https://huggingface.co/floxihunter/codegen-mono-CWEdetect)] [[Code](https://huggingface.co/datasets/floxihunter/synthetic_python_cwe)]
- Context-Enhanced Vulnerability Detection Based on Large Language Model.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.16877)] [[Code](https://github.com/DoeSEResearch/PacVD)]
- Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.13474)]
- Trace Gadgets: Minimizing Code Context for Machine Learning-Based Vulnerability Prediction.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.13676)]
- XGV-BERT: Leveraging Contextualized Language Model and Graph Neural Network for Efficient Software Vulnerability Detection.  **`The Journal of Supercomputing 2025`** [[Paper](https://link.springer.com/article/10.1007/s11227-025-07198-7)]
- SSRFSeek: An LLM-based Static Analysis Framework for Detecting SSRF Vulnerabilities in PHP Applications.  **`AINIT 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11035424)]
- R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2504.04699)] [[Code](https://github.com/martin-wey/R2Vul)]
- IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2405.17238)] [[Code](https://github.com/iris-sast/iris)]
- Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities.  **`ICST 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10988968)] [[Code](https://github.com/seal-research/secvul-llm-study/)]
- Reasoning with LLMs for Zero-Shot Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.17885)] [[Code](https://github.com/Erroristotle/VulnSage)]
- A Comprehensive Study of LLM Secure Code Generation.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.15554)]
- Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.03586)]
- Assessing the Effectiveness of LLMs in Android Application Vulnerability Analysis.  **`ADIoT 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-85593-1_9)]
- HALURust: Exploiting Hallucinations of Large Language Models to Detect Vulnerabilities in Rust.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.10793)]
- You Only Train Once: A Flexible Training Framework for Code Vulnerability Detection Driven by Vul-Vector.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2506.10988)]
- Steering Large Language Models for Vulnerability Detection.  **`ICASSP 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10887736)]
- SecureFalcon: Are We There Yet in Automated Software Vulnerability Detection With LLMs?.  **`TSE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10910240)]
- Castle: Benchmarking Dataset for Static Code Analyzers and LLMs towards CWE Detection.  **`TASE 2025`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-98208-8_15)] [[Code](https://github.com/CASTLE-Benchmark)]
- Benchmarking Large Language Models for Multi-Language Software Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2503.01449)] [[Code](https://github.com/soarsmu/SVD-Bench)]
- Finetuning Large Language Models for Vulnerability Detection.  **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908394)] [[Code](https://github.com/rmusab/vul-llm-finetune)]
- AIDetectVul: Software Vulnerability Detection Method Based on Feature Fusion of Pre-trained Models.  **`ICCECE 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10985370)]
- Fine-Tuning Transformer LLMs for Detecting SQL Injection and XSS Vulnerabilities.  **`ICAIIC 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10920868)]
- Manual Prompt Engineering is Not Dead: A Case Study on Large Language Models for Code Vulnerability Detection with DSPy.  **`CDMA 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10908746)]
- One-for-All Does Not Work! Enhancing Vulnerability Detection by Mixture-of-Experts (MoE).  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.16454)]
- Harnessing Large Language Models for Software Vulnerability Detection: A Comprehensive Benchmarking Study.  **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10879492)]
- Streamlining Security Vulnerability Triage with Large Language Models.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.18908)] [[Code](https://zenodo.org/records/14776104)]
- Towards Explainable Vulnerability Detection with Large Language Models.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2406.09701)]
- Can LLM Prompting Serve as a Proxy for Static Analysis in Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2412.12039)]
- CGP-Tuning: Structure-Aware Soft Prompt Tuning for Code Vulnerability Detection.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2501.04510)]
- Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2412.14841)]
- To Err is Machine: Vulnerability Detection Challenges LLM Reasoning.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2403.17218)] [[Code](https://figshare.com/articles/dataset/Data_Package_for_LLM_Vulnerability_Detection_Study/27368025)]
- Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study.  **`arXiv 2025`** [[Paper](https://arxiv.org/abs/2412.18260)] [[Code](https://github.com/SakiRinn/LLM4CVD)] [[Code](https://huggingface.co/datasets/xuefen/VulResource)]
- Leveraging an Enhanced CodeBERT-Based Model for Multiclass Software Defect Prediction via Defect Classification.  **`IEEE Access 2025`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10820528)]

### 2024
- Vulnerability Detection in Popular Programming Languages with Language Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.15905)] [[Code](https://github.com/syafiq/llm_vd)]
- LLM-Based Approach for Buffer Overflow Detection in Source Code.  **`ICCIT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/11021816)]
- Python Source Code Vulnerability Detection Based on CodeBERT Language Model.  **`ACAI 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10899694)]
- On the Compression of Language Models for Code: An Empirical Study on CodeBERT.  **`SANER 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10992473)] [[Code](https://zenodo.org/records/14357478)]
- Evaluating Large Language Models in Vulnerability Detection Under Variable Context Windows.  **`ICMLA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10903489)]
- EnStack: An Ensemble Stacking Framework of Large Language Models for Enhanced Vulnerability Detection in Source Code.  **`BigData 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10825609)]
- Enhancing Source Code Vulnerability Detection Using Flattened Code Graph Structures.  **`ICFTIC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10913325)]
- SQL Injection Vulnerability Detection Based on Pissa-Tuned Llama 3 Large Language Model.  **`ICFTIC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10912886)]
- Software Vulnerability Detection Using LLM: Does Additional Information Help?.  **`ACSAC Workshops 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10917361)] [[Code](https://github.com/research7485/vulnerability_detection)]
- MVD: A Multi-Lingual Software Vulnerability Detection Framework.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.06166)] [[Code](https://figshare.com/s/10ec70108294a225f391)]
- A Source Code Vulnerability Detection Method Based on Positive-Unlabeled Learning.  **`RICAI 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10911761)]
- A Method of SQL Injection Attack Detection Based on Large Language Models.  **`CNTEIE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10987904)]
- Enhancing Vulnerability Detection Efficiency: An Exploration of Light-weight LLMs with Hybrid Code Features.  **`JISA 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212624002278)] [[Code](https://github.com/JNL-28/Enhancing-Vulnerability-Detection-Efficiency)]
- Enhanced LLM-Based Framework for Predicting Null Pointer Dereference in Source Code.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2412.00216)]
- Boosting Cybersecurity Vulnerability Scanning based on LLM-supported Static Application Security Testing.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.15735)]
- Research on the LLM-Driven Vulnerability Detection System Using LProtector.  **`ICDSCA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10859408)]
- Line-level Semantic Structure Learning for Code Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.18877)] [[Code](https://figshare.com/articles/dataset/CSLS_model_code_and_data/26391658)]
- StagedVulBERT: Multigranular Vulnerability Detection With a Novel Pretrained Code Model.  **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10746847)] [[Code](https://github.com/YuanJiangGit/StagedVulBERT)]
- Enhancing Reverse Engineering: Investigating and Benchmarking Large Language Models for Vulnerability Analysis in Decompiled Binaries.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2411.04981)]
- AutoSafeCoder: A Multi-Agent Framework for Securing LLM Code Generation through Static Analysis and Fuzz Testing.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.10737)] [[Code](https://github.com/SecureAIAutonomyLab/AutoSafeCoder)]
- Fight Fire With Fire: How Much Can We Trust ChatGPT on Source Code-Related Tasks?.  **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10745266)] [[Code](https://figshare.com/s/4b51f0b8a2cda17d08be)]
- Applying Contrastive Learning to Code Vulnerability Type Classification.  **`EMNLP 2024`** [[Paper](https://aclanthology.org/2024.emnlp-main.666/)]
- Fine-Tuning Pre-trained Model with Optimizable Prompt Learning for Code Vulnerability Detection.  **`ISSRE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10771498)] [[Code](https://github.com/Exclusisve-V/PromptVulnerabilityDetection)]
- Exploring AI for Vulnerability Detection and Repair.  **`CARS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10778769)]
- A Qualitative Study on Using ChatGPT for Software Security: Perception vs. Practicality.  **`TPS-ISA 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10835695)] [[Code](https://figshare.com/articles/dataset/Reproduction_package_for_paper_A_Qualitative_Study_on_Using_ChatGPT_for_Software_Security_Perception_vs_Practicality_/24452365?file=48008890)]
- Vul-LMGNNs: Fusing Language Models and Online-distilled Graph Neural Networks for Code Vulnerability Detection.  **`Information Fusion 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253524005268)] [[Code](https://github.com/Vul-LMGNN/vul-LMGGNN)]
- Detecting Source Code Vulnerabilities Using Fine-Tuned Pre-Trained LLMs.  **`ICSP 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10846595)]
- DetectBERT: Code Vulnerability Detection.  **`GCCIT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10862235)]
- VulnerAI: GPT Based Web Application Vulnerability Detection.  **`ICAMAC 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10828788)]
- VULREM: Fine-Tuned BERT-Based Source-Code Potential Vulnerability Scanning System to Mitigate Attacks in Web Applications.  **`Applied Sciences 2024`** [[Paper](https://www.mdpi.com/2076-3417/14/21/9697)]
- Vulnerability Prediction using Pre-trained Models: An Empirical Evaluation.  **`MASCOTS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10786510)] [[Code](https://sites.google.com/view/vpllm/)]
- From Solitary Directives to Interactive Encouragement! LLM Secure Code Generation by Natural Language Prompting.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2410.14321)]
- SecureQwen: Leveraging LLMs for Vulnerability Detection in Python Codebases.  **`Computers \& Security 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824004565)]
- DLAP: A Deep Learning Augmented Large Language Model Prompting framework for software vulnerability detection.  **`JSS 2024`** [[Paper](nan)] [[Code](https://github.com/Yang-Yanjing/DLAP)]
- RealVul: Can We Detect Vulnerabilities in Web Applications with LLM?.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2410.07573)]
- Multitask-Based Evaluation of Open-Source LLM on Software Vulnerability.  **`TSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10706805)] [[Code](https://github.com/vinci-grape/VulEmpirical)]
- Improving Long-Tail Vulnerability Detection Through Data Augmentation Based on Large Language Models.  **`ICSME 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10795073)] [[Code](https://github.com/LuckyDengXiao/LERT)]
- Enhancing Pre-Trained Language Models for Vulnerability Detection via Semantic-Preserving Data Augmentation.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2410.00249)]
- May the Source Be with You: On ChatGPT, Cybersecurity, and Secure Coding.  **`Information 2024`** [[Paper](https://www.mdpi.com/2078-2489/15/9/572)]
- Code Vulnerability Detection: A Comparative Analysis of Emerging Large Language Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.10490)]
- VulnLLMEval: A Framework for Evaluating Large Language Models in Software Vulnerability Detection and Patching.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.10756)]
- Bridge and Hint: Extending Pre-trained Language Models for Long-Range Code.  **`ISSTA 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3650212.3652127)] [[Code](https://anonymous.4open.science/r/EXPO/README.md)]
- SCALE: Constructing Structured Natural Language Comment Trees for Software Vulnerability Detection.  **`ISSTA 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3650212.3652124)] [[Code](https://github.com/Xin-Cheng-Wen/Comment4Vul)]
- Outside the Comfort Zone: Analysing LLM Capabilities in Software Vulnerability Detection.  **`ESORICS 2024`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70879-4_14)]
- Navigating (In)Security of AI-Generated Code.  **`CSR 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10679468)]
- Can a Llama Be a Watchdog? Exploring Llama 3 and Code Llama for Static Application Security Testing.  **`CSR 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10679444)]
- SAFE: Advancing Large Language Models in Leveraging Semantic and Syntactic Relationships for Software Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.00882)]
- Beyond ChatGPT: Enhancing Software Quality Assurance Tasks with Diverse LLMs and Validation Techniques.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.01001)] [[Code](https://figshare.com/s/5da14b0776750c6fa787)]
- Enhancing Source Code Security with LLMs: Demystifying The Challenges and Generating Reliable Repairs.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.00571)]
- Unintentional Security Flaws in Code: Automated Defense via Root Cause Analysis.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2409.00199)]
- VulDetectBench: Evaluating the Deep Capability of Vulnerability Detection with Large Language Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.07595)] [[Code](https://github.com/Sweetaroo/VulDetectBench)]
- Learning-based Models for Vulnerability Detection: An Extensive Study.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.07526)] [[Code](https://figshare.com/s/bde8e41890e8179fbe5f?file=41894784)]
- Uncovering the Limits of Machine Learning for Automatic Vulnerability Detection.  **`USENIX Security 2024`** [[Paper](https://www.usenix.org/conference/usenixsecurity24/presentation/risse)] [[Code](https://github.com/niklasrisse/USENIX_2024)] [[Code](https://github.com/niklasrisse/VPP)]
- Large Language Models for Secure Code Assessment: A Multi-Language Empirical Study.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.06428)]
- VulSim: Leveraging Similarity of {Multi-Dimensional.  **`USENIX Security 2024`** [[Paper](https://www.usenix.org/conference/usenixsecurity24/presentation/shimmi)] [[Code](https://github.com/SamihaShimmi/VulSim)]
- From Generalist to Specialist: Exploring CWE-Specific Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.02329)]
- Automated Software Vulnerability Static Code Analysis Using Generative Pre-Trained Transformer Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2408.00197)]
- DFEPT: Data Flow Embedding for Enhancing Pre-Trained Model Based Vulnerability Detection.  **`Internetware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3671016.3671388)] [[Code](https://github.com/GCVulnerability/DFEPT)]
- Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2407.16235)]
- SCL-CVD: Supervised Contrastive Learning for Code Vulnerability Detection via GraphCodeBERT.  **`Computers \& Security 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824002992)]
- M2CVD: Enhancing Vulnerability Semantic through Multi-Model Collaboration for Code Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.05940)] [[Code](https://github.com/HotFrom/M2CVD)]
- Effectiveness of ChatGPT for Static Analysis: How Far Are We?.  **`AIware 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3664646.3664777)] [[Code](https://zenodo.org/records/10828316)]
- MultiVD: A Transformer-based Multitask Approach for Software Vulnerability Detection.  **`SECRYPT 2024`** [[Paper](https://www.scitepress.org/Papers/2024/127194/127194.pdf)]
- Exploration On Prompting LLM With Code-Specific Information For Vulnerability Detection.  **`SSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10664399)]
- Enhancing Software Code Vulnerability Detection Using GPT-4o and Claude-3.5 Sonnet: A Study on Prompt Engineering Techniques.  **`Electronics 2024`** [[Paper](https://www.mdpi.com/2079-9292/13/13/2657)]
- Vulnerability Classification on Source Code Using Text Mining and Deep Learning Techniques.  **`QRS-C 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10727022)] [[Code](https://sites.google.com/view/vulnerabilityclassification/)]
- Parameter-efficient Multi-classification Software Defect Detection Method based on Pre-trained LLMs.  **`IJCIS 2024`** [[Paper](https://link.springer.com/article/10.1007/s44196-024-00551-3)] [[Code](https://gitee.com/wxyzjp123/msdd-ia3/)]
- Software Vulnerability Prediction in Low-Resource Languages: An Empirical Study of CodeBERT and ChatGPT.  **`EASE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3661167.3661281)] [[Code](https://github.com/lhmtriet/LLM4Vul)]
- Security Vulnerability Detection with Multitask Self-Instructed Fine-Tuning of Large Language Models.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.05892)] [[Code](https://zenodo.org/records/11403208)]
- SVulDetector: Vulnerability Detection based on Similarity using Tree-based Attention and Weighted Graph Embedding Mechanisms.  **`COSE 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824002335)] [[Code](https://figshare.com/s/426156a96a83da1d38d0)]
- Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2406.03718)] [[Code](https://github.com/CGCL-codes/VulLLM)]
- Greening Large Language Models of Code.  **`ICSE-SEIS 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3639475.3640097)] [[Code](https://github.com/soarsmu/Avatar)]
- Evaluating the Impact of Conventional Code Analysis Against Large Language Models in API Vulnerability Detection.  **`EICC 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3655693.3655701)]
- Large Language Model for Vulnerability Detection: Emerging Results and Future Directions.  **`ICSE-NIER 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3639476.3639762)] [[Code](https://github.com/soarsmu/ChatGPT-VulDetection)]
- LLM-CloudSec: Large Language Model Empowered Automatic and Deep Vulnerability Analysis for Intelligent Clouds.  **`INFOCOM WKSHPS 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10620804)] [[Code](https://github.com/DPCa0/LLM-CloudSec)]
- LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks.  **`SP 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10646663/)] [[Code](https://github.com/ai4cloudops/SecLLMHolmes)]
- VulD-CodeBERT: CodeBERT-Based Vulnerability Detection Model for C/C++ Code.  **`CISCE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10653337)]
- DB-CBIL: A DistilBert-Based Transformer Hybrid Model Using CNN and BiLSTM for Software Vulnerability Detection.  **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10517582)]
- VulnGPT: Enhancing Source Code Vulnerability Detection Using AutoGPT and Adaptive Supervision Strategies.  **`DCOSS-IoT 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10621527)]
- Enhancing Static Analysis for Practical Bug Detection: An LLM-Integrated Approach.  **`PACMPL 2024`** [[Paper](https://dl.acm.org/doi/full/10.1145/3649828)] [[Code](https://sites.google.com/view/llift-open/home)]
- VulEval: Towards Repository-Level Evaluation of Software Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2404.15596)]
- ProRLearn: Boosting Prompt Tuning-based Vulnerability Detection by Reinforcement Learning.  **`ASE 2024`** [[Paper](https://link.springer.com/article/10.1007/s10515-024-00438-9)] [[Code](https://github.com/ProRLearn/ProRLearn001)]
- Towards Causal Deep Learning for Vulnerability Detection.  **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3597503.3639170)] [[Code](https://figshare.com/s/0ffda320dcb96c249ef2?file=41801019)]
- Pre-training by Predicting Program Dependencies for Vulnerability Analysis Tasks.  **`ICSE 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10548173)] [[Code](https://zenodo.org/records/10140638)]
- BiT5: A Bidirectional NLP Approach for Advanced Vulnerability Detection in Codebase.  **`Procedia Computer Science 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S1877050924006306)]
- Software Vulnerability and Functionality Assessment using Large Language Models.  **`NLBSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3643787.3648036)]
- GRACE: Empowering LLM-based Software Vulnerability Detection with Graph Structure and In-Context Learning.  **`JSS 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0164121224000748)] [[Code](https://github.com/P-E-Vul/GRACE)]
- Python Source Code Vulnerability Detection with Named Entity Recognition.  **`Computers \& Security 2024`** [[Paper](https://www.sciencedirect.com/science/article/pii/S0167404824001032)] [[Code](https://github.com/mmeberg/PyVulDet-NER)]
- Chain-of-Thought Prompting of Large Language Models for Discovering and Fixing Software Vulnerabilities.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2402.17230)]
- LLbezpeky: Leveraging Large Language Models for Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.01269)]
- TRACED: Execution-aware Pre-training for Source Code.  **`ICSE 2024`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3597503.3608140)] [[Code](https://github.com/ARiSE-Lab/TRACED_ICSE_24)]
- DP-CCL: A Supervised Contrastive Learning Approach Using CodeBERT Model in Software Defect Prediction.  **`IEEE Access 2024`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10422975)] [[Code](https://github.com/saharsadia/DP-CCL)]
- A Preliminary Study on Using Large Language Models in Software Pentesting.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.17459)]
- Your Instructions Are Not Always Helpful: Assessing the Efficacy of Instruction Fine-tuning for Software Vulnerability Detection.  **`arXiv 2024`** [[Paper](https://arxiv.org/abs/2401.07466)]

### 2023
- How Far Have We Gone in Vulnerability Detection Using Large Language Models.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2311.12420)] [[Code](https://github.com/Hustcw/VulBench)]
- Enhancing Code Security Through Open-source Large Language Models: A Comparative Study.  **`FPS 2023`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-57537-2_15)]
- Code Defect Detection Method Based on BERT and Ensemble.  **`ICCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10507306)]
- Optimizing Pre-trained Language Models for Efficient Vulnerability Detection in Code Snippets.  **`ICCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10507456)]
- Exploring the Limits of ChatGPT in Software Security Applications.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2312.05275)]
- ChatGPT for Vulnerability Detection, Classification, and Repair: How Far Are We?.  **`APSEC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10479409)] [[Code](https://github.com/awsm-research/ChatGPT4Vul)]
- Assessing the Effectiveness of Vulnerability Detection via Prompt Tuning: An Empirical Study.  **`APSEC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10479384)] [[Code](https://github.com/P-E-Vul/prompt-empircial-vulnerability)]
- Joint Geometrical and Statistical Domain Adaptation for Cross-domain  Code Vulnerability Detection.  **`EMNLP 2023`** [[Paper](https://aclanthology.org/2023.emnlp-main.788/)]
- The EarlyBIRD Catches the Bug: On Exploiting Early Layers of Encoder Models for More Efficient Code Classification.  **`ESEC/FSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3611643.3616304)] [[Code](https://zenodo.org/records/10499843)]
- Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation.  **`ESEC/FSE 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3611643.3616358)] [[Code](https://github.com/jacknichao/SVulD)]
- Software Defect Prediction via Code Language Models.  **`ICCTIT 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10435711)]
- AIBugHunter: A Practical Tool for Predicting, Classifying and Repairing Software Vulnerabilities.  **`EMSE 2023`** [[Paper](https://link.springer.com/article/10.1007/s10664-023-10346-3)] [[Code](https://github.com/awsm-research/AIBugHunter)]
- Do Language Models Learn Semantics of Code? A Case Study in Vulnerability Detection.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2311.04109)] [[Code](https://figshare.com/s/4a16a528d6874aad51a0)]
- Software Vulnerabilities Detection Based on a Pre-trained Language Model.  **`TrustCom 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10538979)]
- Enhancing Large Language Models for Secure Code Generation: A Dataset-driven Study on Vulnerability Mitigation.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2310.16263)]
- DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection.  **`RAID 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3607199.3607242)] [[Code](https://github.com/wagner-group/diversevul)]
- Software Vulnerability Detection using Large Language Models.  **`ISSREW 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10301302)]
- PTLVD:Program Slicing and Transformer-based Line-level Vulnerability Detection System.  **`SCAM 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10356694)] [[Code](https://github.com/chenshixu/PTLVD)]
- Prompt Tuning in Code Intelligence: An Experimental Evaluation.  **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10269066)] [[Code](https://github.com/adf1178/PT4Code)]
- A New Approach to Web Application Security: Utilizing GPT Language Models for Source Code Inspection.  **`Future Internet 2023`** [[Paper](https://www.mdpi.com/1999-5903/15/10/326)]
- DefectHunter: A Novel LLM-Driven Boosted-Conformer-based Code Vulnerability Detection Mechanism.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2309.15324)] [[Code](https://github.com/WJ-8/DefectHunter)]
- Function-Level Vulnerability Detection Through Fusing Multi-Modal Knowledge.  **`ASE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10298584)] [[Code](https://github.com/jacknichao/MVulD)]
- When Less is Enough: Positive and Unlabeled Learning Model for Vulnerability Detection.  **`ASE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10298363)] [[Code](https://github.com/PILOT-VD-2023/PILOT)]
- Using ChatGPT as a Static Application Security Testing Tool.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.14434)] [[Code](https://github.com/abakhshandeh/ChatGPTasSAST)]
- Deep Learning-Based Framework for Automated Vulnerability Detection in Android Applications.  **`IBCAST 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10713017)]
- Can Large Language Models Find And Fix Vulnerable Software?.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2308.10345)]
- Software Vulnerability Detection with GPT and In-Context Learning.  **`DSC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10381286)]
- VulExplainer: A Transformer-Based Hierarchical  Distillation for Explaining Vulnerability Types.  **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10220166)] [[Code](https://github.com/awsm-research/VulExplainer)]
- VulDetect: A novel technique for detecting software vulnerabilities using Language Models.  **`CSR 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10224924)]
- Leveraging Deep Learning Models for Cross-function Null Pointer Risks Detection.  **`AITest 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10229470)]
- An Enhanced Vulnerability Detection in Software Using a Heterogeneous Encoding Ensemble.  **`ISCC 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10217978)]
- An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph.  **`EuroS&P 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10190505)] [[Code](https://github.com/pial08/SemVulDet)]
- Vulnerability Detection by Learning From Syntax-Based Execution Paths of Code.  **`TSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10153647)] [[Code](https://zenodo.org/records/7123322)]
- New Tricks to Old Codes: Can AI Chatbots Replace Static Code Analysis Tools?.  **`EICC 2023`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3590777.3590780)] [[Code](https://github.com/New-Tricks-to-Old-Codes/Replace-Static-Analysis-Tools)]
- Transformer-based Vulnerability Detection in Code at EditTime: Zero-shot, Few-shot, or Fine-tuning?.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2306.01754)]
- VulDefend: A Novel Technique based on Pattern-exploiting Training for Detecting Software Vulnerabilities Using Language Models.  **`JEEIT 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10185860)]
- Detecting Vulnerabilities in IoT Software: New Hybrid Model and Comprehensive Data Analysis.  **`JISA 2023`** [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212623000510)]
- Keeping Pace with Ever-Increasing Data: Towards Continual Learning of Code Intelligence Models.  **`ICSE 2023`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10172346)] [[Code](https://github.com/ReliableCoding/REPEAT)]
- Evaluation of ChatGPT Model for Vulnerability Detection.  **`arXiv 2023`** [[Paper](https://arxiv.org/abs/2304.07232)]

### 2022
- PATVD: Vulnerability Detection Based on Pre-training Techniques and Adversarial Training.  **`SmartWorld/UIC/ScalCom/DigitalTwin/PriComp/Meta 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10189687/)]
- Exploring Transformers for Multi-Label Classification of Java Vulnerabilities.  **`QRS 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/10062434)] [[Code](https://github.com/TQRG/VDET-for-Java)]
- BBVD: A BERT-based Method for Vulnerability Detection.  **`IJACSA 2022`** [[Paper](https://www.proquest.com/docview/2770373789?pq-origsite=gscholar&fromopenview=true&sourcetype=Scholarly%20Journals)]
- Transformer-Based Language Models for Software Vulnerability Detection.  **`ACSAC 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3564625.3567985)] [[Code](https://bitbucket.csiro.au/users/jan087/repos/acsac-2022-submission/browse)]
- Distilled and Contextualized Neural Models Benchmarked for Vulnerable Function Detection.  **`Mathematics 2022`** [[Paper](https://www.mdpi.com/2227-7390/10/23/4482)]
- BERT-Based Vulnerability Type Identification with Effective Program Representation.  **`WASA 2022`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19208-1_23#citeas)]
- VulDeBERT: A Vulnerability Detection System Using BERT.  **`ISSREW 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9985089)] [[Code](https://github.com/SKKU-SecLab/VulDeBERT)]
- VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection.  **`IJCNN 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9892280)] [[Code](https://github.com/ICL-ml4csec/VulBERTa)]
- Cyber Security Vulnerability Detection Using Natural Language Processing.  **`AIIoT 2022`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9817336)]
- fuLineVulTransformerbasedLineLevel2022.  **`MSR 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3524842.3528452)] [[Code](https://github.com/awsm-research/LineVul)]
- LineVD: Statement-level Vulnerability Detection using Graph Neural Networks.  **`MSR 2022`** [[Paper](https://dl.acm.org/doi/abs/10.1145/3524842.3527949)] [[Code](https://github.com/davidhin/linevd)]
- Intelligent Detection of Vulnerable Functions in Software through Neural Embedding-based Code Analysis.  **`IJNM 2022`** [[Paper](https://onlinelibrary.wiley.com/doi/full/10.1002/nem.2198)] [[Code](https://cybercodeintelligence.github.io/CyberCI/)]
- Software Defect Prediction Employing BiLSTM and BERT-based Semantic Feature.  **`Soft Computing 2022`** [[Paper](https://link.springer.com/article/10.1007/s00500-022-06830-5)]
- Deep Neural Embedding for Software Vulnerability Discovery: Comparison and Optimization.  **`Security and Communication Networks 2022`** [[Paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/5203217)] [[Code](https://cybercodeintelligence.github.io/CyberCI/)]

### 2021
- Automated Software Vulnerability Detection via Pre-trained Context Encoder and Self Attention.  **`ICDF2C 2021`** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-06365-7_15)]
- Detecting Integer Overflow Errors in Java Source Code via Machine Learning.  **`ICTAI 2021`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9643278)]
- Unified Pre-training for Program Understanding and Generation.  **`NAACL HLT 2021`** [[Paper](https://par.nsf.gov/servlets/purl/10336701)] [[Code](https://github.com/wasiahmad/PLBART)]
- An Empirical Study on Software Defect Prediction Using CodeBERT Model.  **`Applied Sciences 2021`** [[Paper](https://www.mdpi.com/2076-3417/11/11/4793)] [[Code](https://gitee.com/penguinc/applsci-code-bert-defect-prediciton)]
- Security Vulnerability Detection Using Deep Learning Natural Language Processing.  **`INFOCOM WKSHPS 2021`** [[Paper](https://ieeexplore.ieee.org/abstract/document/9484500)]

### 2020
- Exploring Software Naturalness through Neural Language Models.  **`arXiv 2020`** [[Paper](https://arxiv.org/abs/2006.12641)]




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

Please cite our paper if you use this resource.
