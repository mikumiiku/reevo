# ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution

Haoran Ye $^{1}$ , Jiarui Wang $^{2}$ , Zhiguang Cao $^{3}$ , Federico Berto $^{4}$ , Chuanbo Hua $^{4}$ , Haeyeon Kim $^{4}$ , Jinkyoo Park $^{4}$ , Guojie Song $^{15*}$

$^{1}$ National Key Laboratory of General Artificial Intelligence, School of Intelligence Science and Technology, Peking University  
$^{2}$ Southeast University  $^{3}$ Singapore Management University  $^{4}$ KAIST  $^{5}$ PKU-Wuhan Institute for Artificial Intelligence AI4CO†

hrye@stu.pku.edu.cn, jiarui_wang@seu.edu.cn, zgcao@smu.edu.sg {fberto, cbhua, haeyeonkim, jinkyoo PARK}@kaist.ac.kr, gjsong@pku.edu.cn

Project Website: https://ai4co.github.io/reeveo

# Abstract

The omnipresence of NP-hard combinatorial optimization problems (COPs) compels domain experts to engage in trial-and-error heuristic design. The long-standing endeavor of design automation has gained new momentum with the rise of large language models (LLMs). This paper introduces Language Hyper-Heuristics (LHHs), an emerging variant of Hyper-Heuristics that leverages LLMs for heuristic generation, featuring minimal manual intervention and open-ended heuristic spaces. To empower LHHs, we present Reflective Evolution (ReEvo), a novel integration of evolutionary search for efficiently exploring the heuristic space, and LLM reflections to provide verbal gradients within the space. Across five heterogeneous algorithmic types, six different COPs, and both white-box and black-box views of COPs, ReEvo yields state-of-the-art and competitive meta-heuristics, evolutionary algorithms, heuristics, and neural solvers, while being more sample-efficient than prior LHHs.

# 1 Introduction

NP-hard combinatorial optimization problems (COPs) pervade numerous real-world systems, each characterized by distinct constraints and objectives. The intrinsic complexity and heterogeneity of these problems compel domain experts to laboriously develop heuristics for their approximate solutions [23]. Automation of heuristic designs represents a longstanding pursuit.

Classic Hyper-Heuristics (HHs) automate heuristic design by searching for the best heuristic (combination) from a set of heuristics or heuristic components [64]. Despite decades of development, HHs are limited by heuristic spaces predefined by human experts [64]. The rise of large language models (LLMs) opens up new possibilities for HHs. This paper introduces the general concept of Language Hyper-Heuristics (LHH) to advance beyond preliminary attempts in individual COP settings [68, 46]. LHH constitutes an emerging variant of HH that utilizes LLMs for heuristic generations. It features minimal human intervention and open-ended heuristic spaces, showing promise to comprehensively shift the HH research paradigm.

Pure LHH (e.g., LLM generations alone) is sample-inefficient and exhibits limited inference capability for black-box COPs. This work elicits the power of LHH with Reflective Evolution (ReEvo). ReEvo couples evolutionary search for efficiently exploring heuristic spaces, with self-reflections to boost the reasoning capabilities of LLMs. It emulates human experts by reflecting on the relative performance of two heuristics and gathering insights across iterations. This reflection approach is analogous to interpreting genetic cues and providing "verbal gradient" within search spaces. We introduce fitness landscape analysis and black-box prompting for reliable evaluation of LHHs. The dual-level reflections are shown to enhance heuristic search and induce verbal inference for black-box COPs, enabling ReEvo to outperform prior state-of-the-art (SOTA) LHH [47].

We introduce novel applications of LHHs and yield SOTA solvers with ReEvo: (1) We evolve penalty heuristics for Guided Local Search (GLS), which outperforms SOTA learning-based [52, 24, 75] and knowledge-based [1] (G)LS solvers. (2) We enhance Ant Colony Optimization (ACO) by evolving its heuristic measures, surpassing both neural-enhanced heuristics [94] and expert-designed heuristics [71, 6, 72, 17, 39]. (3) We refine the genetic algorithm (GA) for Electronic Design Automation (EDA) by evolving genetic operators, outperforming expert-designed GA [63] and the SOTA neural solver [31] for the Decap Placement Problem (DPP). (4) Compared to a classic HH [15], ReEvo generates superior constructive heuristics for the Traveling Salesman Problem (TSP). (5) We enhance the generalization of SOTA neural combinatorial optimization (NCO) solvers [37, 51] by evolving heuristics for attention reshaping. For example, we improve the optimality gap of POMO [37] from  $52\%$  to  $29\%$  and LEHD [51] from  $3.2\%$  to  $3.0\%$  on TSP1000, with negligible additional time overhead and no need for tuning neural models.

We summarize our contributions as follows. (1) We propose the concept of Language HyperHeuristics (LHHs), which bridges emerging attempts using LLMs for heuristic generation with a methodological group that enjoys decades of development. (2) We present Reflective Evolution (ReEvo), coupling evolutionary computation with humanoid reflections to elicit the power of LHHs. We introduce fitness landscape analysis and black-box prompting for reliable LHH evaluations, where ReEvo achieves SOTA sample efficiency. (3) We introduce novel applications of LHHs and present SOTA COP solvers with ReEvo, across five heterogeneous algorithmic types and six different COPs.

# 2 Related work

Traditional Hyper-Heuristics. Traditional HHs select the best performing heuristic from a predefined set [13] or generate new heuristics through the combination of simpler heuristic components [15, 104]. HHs offer a higher level of generality in solving various optimization problems [109, 96, 19, 44, 103, 58], but are limited by the heuristic space predefined by human experts.

Neural Combinatorial Optimization. Recent advances of NCO show promise in learning end-to-end solutions for COPs [2, 93, 3]. NCO can be regarded as a variant of HH, wherein neural architectures and solution pipelines define a heuristic space, and training algorithms search within it. A well-trained neural network (NN), under certain solution pipelines, represents a distinct heuristic. From this perspective, recent advancements in NCO HHs have led to better-aligned neural architectures [28, 51, 34, 73] and advanced solution pipelines [32, 52, 42, 89, 95, 12, 5] to define effective heuristic spaces, and improved training algorithms to efficiently explore heuristic spaces [33, 27, 14, 76, 18, 90, 79, 35], while targeting increasingly broader applications [9, 107, 54, 77]. In this work, we show that ReEvo-generated heuristics can outperform or enhance NCO methods.

LLMs for code generation and optimization. The rise of LLMs introduces new prospects for diverse fields [88, 82, 105, 25, 50, 99]. Among others, code generation capabilities of LLMs are utilized for code debugging [10, 49], enhancing code performance [55], solving algorithmic competition challenges [41, 70], robotics [38, 43, 81], and general task solving [92, 102]. Interleaving LLM generations with evaluations [74] yields powerful methods for prompt optimization [108, 83, 20], reinforcement learning (RL) reward design [53], algorithmic (self-)improvement [98, 48, 45], neural architecture search [8], and general solution optimization [91, 4, 80], with many under evolutionary frameworks [57, 87, 21, 7, 40]. Most related to ReEvo, concurrent efforts by Liu et al. [47] and Romera-Paredes et al. [68] leverage LLMs to develop heuristics for COPs. We go beyond and propose generic LHH for COPs, along with better sample efficiency, broader applications, more reliable evaluations, and improved heuristics. In addition, ReEvo contributes to a smoother fitness landscape,

showing the potential to enhance other tasks involving LLMs for optimization. We present further discussions in Appendix A.

Self-reflections of LLMs. Shinn et al. [70] propose to reinforce language agents via linguistic feedback, which is subsequently harnessed for various tasks [56, 84]. While Shinn et al. [70] leverage binary rewards indicating passing or failing test cases in programming, ReEvo extends the scope of verbal RL feedback to comparative analysis of two heuristics, analogous to verbal gradient information [66] within heuristic spaces. Also, ReEvo incorporates reflection within an evolutionary framework, presenting a novel and powerful integration.

# 3 Language Hyper-Heuristics for Combinatorial Optimization

HHs explore a search space of heuristic configurations to select or generate effective heuristics, indirectly optimizing the underlying COP. This dual-level framework is formally defined as follows.

Definition 3.1 (Hyper-Heuristic). For COP with solution space  $S$  and objective function  $f: S \to \mathbb{R}$ , a Hyper-Heuristic (HH) searches for the optimal heuristic  $h^*$  in a heuristic space  $H$  such that a meta-objective function  $F: H \to \mathbb{R}$  is minimized, i.e.,  $h^* = \operatorname*{argmin}_{h \in H} F(h)$ .

Depending on how the heuristic space  $H$  is defined, traditional HHs can be categorized into selection and generation HHs, both entailing manually defined heuristic primitives. Here, we introduce a novel variant of HHs, Language Hyper-Heuristics (LHH), wherein heuristics in  $H$  are generated by LLMs. LHHs dispense with the need for predefined  $H$ , and instead leverage LLMs to explore an open-ended heuristic space. We recursively define LHHs as follows.

Definition 3.2 (Language Hyper-Heuristic). A Language Hyper-Heuristic (LHH) is an HH variant where heuristics in  $H$  are generated by LLMs.

In this work, we define the meta-objective function  $F$  as the expected performance of a heuristic  $h$  for certain COP. It is estimated by the average performance on a dataset of problem instances.

# 4 Language Hyper-Heuristic with ReEvo

LHH takes COP specifications as input and outputs the best inductive heuristic found for this COP. Vanilla LHH can be repeated LLM generations to randomly search the heuristic space, which is sample-inefficient and lacks reasoning capabilities for complex and black-box problems (see § 6). Therefore, we propose Reflective Evolution (ReEvo) to interpret genetic cues of evolutionary search and unleash the power of LHHs.

ReEvo is schematically illustrated in Fig. 1. Under an evolutionary framework, LLMs assume two roles: a generator  $LLM$  for generating individuals and a reflector  $LLM$  for guiding the generation with reflections. ReEvo, as an LHH, features a distinct individual encoding, where each individual is the code snippet of a heuristic. Its evolution begins with population initialization, followed by five iterative steps: selection, short-term reflection, crossover, long-term reflection, and elitist mutation. We evaluate the meta-objective of all heuristics, both after crossover and mutation. Our prompts are gathered in Appendix B.

Individual encoding. ReEvo optimizes toward best-performing heuristics via an evolutionary process, specifically a Genetic Programming (GP). It diverges from traditional GPs in that (1) individuals are code snippets generated by LLMs, and (2) individuals are not constrained by any predefined encoding format, except for adhering to a specified function signature.

Population initialization. ReEvo initializes a heuristic population by prompting the generator LLM with a task specification. A task specification contains COP descriptions (if available), heuristic designation, and heuristic functionality. Optionally, including seed heuristics, either trivial or expertly crafted to improve upon, can provide in-context examples that encourage valid heuristic generation and bias the search toward more promising directions.

A ReEvo iteration contains the following five sequential steps.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/9214877439ee6225b59b08d58b779de7994143562068a2d19615eaaefc55e883.jpg)

Figure 1: An illustration of ReEvo.  
![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/27f9fa9bd90f89276a13fcc9380b5a2bf5e5cd9869b104c19df465dc1b86ab75.jpg)  
(a) ReEvo pipeline. Top: ReEvo evolves a population of heuristics. Insights and knowledge are verbalized as long-term reflections and accumulated throughout iterations. Bottom: A ReEvo iteration contains five sequential steps: selection, short-term reflection, crossover, long-term reflection, and elitist mutation.  
(b) Examples of reflections for black-box TSP. Heuristics are designed for Ant Colony Optimization (see § 5.2). Left: Given a pair parent heuristics, ReEvo correctly infers the TSP objective and generates a better offspring accordingly. Right: Given the elite heuristic and accumulated long-term reflections, ReEvo incorporates the suggested statistics and yields a better mutated heuristic.

Selection. ReEvo selects parent pairs from successfully executed heuristics at random, while avoiding pairing heuristics with an identical meta-objective value  $F$ .

Short-term reflection. For each pair of heuristic parents, the reflector LLM reflects upon their relative performance and gives hints accordingly for improved design. Unlike prior work [70], ReEvo integrates the reflections into evolutionary search and reflects by performing comparative analyses. Our proposed approach is analogous to interpreting genetic cues and providing verbal gradients within search spaces, which leads to smoother fitness landscapes and better search results (see § 6.1).

Crossover. ReEvo prompts the generator LLM to generate an offspring heuristic, given task specifications, a pair of parent heuristics, explicit indications of their relative performance, short-term reflections over the pair, and generation instructions.

Long-term reflection. ReEvo accumulates expertise in improving heuristics via long-term reflections. The reflector LLM, given previous long-term reflections and newly gained short-term ones, summarizes them and gives hints for improved heuristic design.

Elitist mutation. ReEvo employs an elitist mutation approach. Based on long-term reflections, the generator LLM samples multiple heuristics to improve the current best one. A mutation prompt consists of task specifications, the elite heuristic, long-term reflections, and generation instructions.

Viewing ReEvo from the perspective of an LLM agentic architecture [88], short-term reflections interpret the environmental feedback from each round of interaction. Long-term reflections distill accumulated experiences and knowledge, enabling them to be loaded into the inference context without causing memory blowups.

# 5 Heuristic generation with ReEvo

This section presents novel applications of LHH across heterogeneous algorithmic types and diverse COPs. With ReEvo, we yield state-of-the-art and competitive meta-heuristics, evolutionary algorithms, heuristics, and neural solvers.

Hyperparameters of ReEvo and detailed experimental setup are given in Appendix C. We apply ReEvo to different algorithmic types across six diverse COPs representative of different areas: Traveling Salesman Problem (TSP), Capacitated Vehicle Routing Problem (CVRP), and Orienteering Problem (OP) for routing problems; Multiple Knapsack Problem (MKP) for subset problems; Bin Packing Problem (BPP) for grouping problems; and Decap Placement Problem (DPP) for electronic design automation (EDA) problems. Details of the benchmark COPs are given in Appendix D. The best ReEvo-generated heuristics are collected in Appendix E.

# 5.1 Penalty heuristics for Guided Local Search

We evolve penalty heuristics for Guided Local Search (GLS) [1]. GLS interleaves local search with solution perturbation. The perturbation is guided by the penalty heuristics to maximize its utility. ReEvo searches for the penalty heuristic that leads to the best GLS performance.

We implement the best heuristic generated by ReEvo within KGLS [1] and refer to such coupling as KGLS-ReEvo. In Table 1, we compare KGLS-ReEvo with the original KGLS, other GLS variants [24, 75, 47], and SOTA NCO method that learns to improve a solution [52]. The results show that ReEvo can improve KGLS and outperform SOTA baselines. In addition, we use a single heuristic for TSP20 to 200, while NCO baselines require training models specific to each problem size.

Table 1: Evaluation results of different local search (LS) variants. We report optimality gaps and per-instance execution time.  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="2">TSP20</td><td colspan="2">TSP50</td><td colspan="2">TSP100</td><td colspan="2">TSP200</td></tr><tr><td>Opt. gap (%)</td><td>Time (s)</td><td>Opt. gap (%)</td><td>Time (s)</td><td>Opt. gap (%)</td><td>Time (s)</td><td>Opt. gap (%)</td><td>Time (s)</td></tr><tr><td>NeuOpt* [52]</td><td>LS+RL</td><td>0.000</td><td>0.124</td><td>0.000</td><td>1.32</td><td>0.027</td><td>2.67</td><td>0.403</td><td>4.81</td></tr><tr><td>GNNGLS [24]</td><td>GLS+SL</td><td>0.000</td><td>0.116</td><td>0.052</td><td>3.83</td><td>0.705</td><td>6.78</td><td>3.522</td><td>9.92</td></tr><tr><td>NeuralGLS† [75]</td><td>GLS+SL</td><td>0.000</td><td>10.005</td><td>0.003</td><td>10.01</td><td>0.470</td><td>10.02</td><td>3.622</td><td>10.12</td></tr><tr><td>EoH [47]</td><td>GLS+LHH</td><td>0.000</td><td>0.563</td><td>0.000</td><td>1.90</td><td>0.025</td><td>5.87</td><td>0.338</td><td>17.52</td></tr><tr><td>KGLS‡ [1]</td><td>GLS</td><td>0.004</td><td>0.001</td><td>0.017</td><td>0.03</td><td>0.002</td><td>1.55</td><td>0.284</td><td>2.52</td></tr><tr><td>KGLS-ReEvo‡</td><td>GLS+LHH</td><td>0.000</td><td>0.001</td><td>0.000</td><td>0.03</td><td>0.000</td><td>1.55</td><td>0.216</td><td>2.52</td></tr></table>

*: All instances are solved in one batch. D2A=1; T=500, 4000, 5000, and 5000 for 4 problem sizes, respectively.  
†: The results are drawn from the original literature. ‡: They are based on our own GLS implementation.

# 5.2 Heuristic measures for Ant Colony Optimization

Solutions to COPs can be stochastically sampled, with heuristic measures indicating the promise of solution components and biasing the sampling. Ant Colony Optimization (ACO), which interleaves stochastic solution sampling with pheromone update, builds on this idea. We generate such heuristic measures for five different COPs: TSP, CVRP, OP, MKP, and BPP.

Under the ACO framework, we evaluate the best ReEvo-generated heuristics against the expert-designed ones and neural heuristics specifically learned for ACO [94]. The evolution curves displayed

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/33d5e573c5f86c38697b883fd7c7c4d0b95f5b8906e09273d210557beb052d72.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/8fdb76cee1ed9c4c43c8aca22eb5e2be344cee2183d4f9eec5a4e4711283010e.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/a9b35ed6716aa61f6a90532c3dcccd82d10f91df6cbbdc2e03d4a15df0edc094.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/93678d72265134b98568c150bb2d57d9436916ed8207362984040014bf8793e4.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/8688998356bef4b2f66c3f202986ccf19b237a02204901a537885289df698d38.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/871c49a0503b3584750195a1349da91ba9647b60474a5e928c8c20605aaf59a5.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/34b21c752227c6ea3ef880c781fb98ee99fde46f5f54727d401fc4ec83a0ba10.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/7aab970b89c6a25040176380322fd5782c8e466b599525ed780a2a995a94dffc.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/d5d2c971a03c5d4bdf2b626f7a94e3cb1c26d5a541ba249c2243714cf0dfc5c0.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/129ddf907d138ca505f81823d96b5d9ee3f6305c4eb0debbe59cf9c535c27ab7.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/184a4b166ef014e8e30338a6b9eddaa25795712b8074bb4fae5e6c718407cdeb.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/abb20e677398429ac43b367603bf2b82a43c437b11a855ecf7962ba05f0fde8e.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/88b1190bca19e91ad8f67c5f076ab15e0904a03e142d98473cc552d8cc071f1d.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/142f9ac1d78a870853ea2b6c95d8dc21a971b1c3f458ab7a0846ff09210714bc.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/197af9410cce19b401faaf34ddd162451f5c473fc50fc3a409f1bd06236e035c.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/022fffd618161bc772e0f91b63046a248ae2263a9fbab77a590af1419cb49076.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/3e944db4dffd7a64e8f5ed44b5dd01f48c241ebdbc49b1b272b2e7d475dd8afc.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/6daf4393ba7b2c768ac1b7d8712636e11674c381a244b8755677dbda01447fe4.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/f750ac37b4faa5b95e2fe914cc0d4df4c93e41fa4203b983f92d1b31221f3b3c.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/97717195f87e5f1d205bf2482082a9e1bba3721cb4dbf82e22482d1f34dd4a0d.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/4ddf7470f5bdc4ebbbdeb7c2c8dadf9ea19c5902d7ab8d3d9edf4efd6e9e8fc2.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/7ffbe566b418b90da7659819b8ebcf1db92fc55a08e667a59db9ce1080c39bc1.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/6a60bec6899f85051e9dfba1c3f0601cdb6b3e2b788f78d7215f055047264671.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/8e495ec02fc53b18848765aafe7186463175ecaec69f3c46973d3167bd2726e2.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/e0fb42d2d8d09e952dafea3ac4406b346db9cd759527388b0d6ce1230fde34ed.jpg)  
Figure 2: Comparative evaluations of ACO using expert-designed heuristics [71, 6, 72, 17, 39], neural heuristics [94], and ReEvo heuristics. For each COP, the same neural heuristic or the ReEvo heuristic is applied across all problem sizes; both heuristics are trained exclusively on the smallest problem size among the five. Left: Relative performance improvement of DeepACO and ReEvo over human baselines w.r.t. problem sizes. Right: ACO evolution curves, plotting the all-time best objective value w.r.t. the number of solution evaluations. The curves are averaged over three runs in which only small variances are observed (e.g.,  $\sim 0.01$  for TSP50).

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/3fe7047b3119a2ef9bf21a3c707cf77fe052b88e00ad2344208dc14558ac4a36.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/99b0882385004b9dec4a0640b0491bee6f17a30ac98393c4dedcd3f549c4abfd.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/3b150c969cecc965725c2bee51e87ef0963f257a1f4e3d845f651eabcf184f08.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/9daf9aaef04b1f9f4d3f3e75bd58d167abc6fe32574b43263e2027f6b48eca1e.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/14c2b4e003bcc823375f60d908e5b53a73d3899f9216c37ac2c87bff5593f783.jpg)

in Fig. 2 verify the consistent superiority of ReEvo across COPs and problem sizes. Notably, on 3 out of 5 COPs, ReEvo outperforms DeepACO [94] even when the latter overfits the test problem size (TSP50, OP50, and MKP100). We observe that most ReEvo-generated heuristics show consistent performance across problem sizes and distributions. Hence, their advantages grow as the distributional shift increases for neural heuristics.

# 5.3 Genetic operators for Electronic Design Automation

Expert-designed GAs are widely adopted in EDA [69, 97, 11, 26]. Besides directly solving EDA problems, GA-generated solutions can be used to train amortized neural solvers [31]. Here, we show that ReEvo can improve the expert-designed GAs and outperform DevFormer [31], the SOTA solver for the DPP problem. We sequentially evolve with ReEvo the crossover and mutation operators for the GA expert-designed by Park et al. [63]. Fig. 3 compares online and offline learned methods, DevFormer, the original expert-designed GA, and the GA with ReEvo-generated operators, showing that the ReEvo-designed GA outperforms previous methods and, importantly, both the expert-designed GA and DevFormer.

# 5.4 Constructive heuristics for the Traveling Salesman Problem

Heuristics can be used for deterministic solution construction by sequentially assigning values to each decision variable. We evaluate the constructive heuristic for TSP generated by ReEvo on real-world benchmark instances from TSPLIB [67] in Table 2. ReEvo can generate better heuristics than GHPP [15], a classic HH based on GP.

# 5.5 Attention reshaping for Neural Combinatorial Optimization

Autoregressive NCO solvers suffer from limited scaling-up generalization [29], partially due to the dispersion of attention scores [85]. Wang et al. [85] design a distance-aware heuristic to reshape

Table 2: Comparisons of constructive heuristics designed by human, GHPP [15], and ReEvo. We report the average optimality gap of each instance, where the baseline results are drawn from [15] and the results of ReEvo are averaged over 3 runs with different starting nodes.  

<table><tr><td>Instance</td><td>Nearest Neighbour</td><td>GHPP [15]</td><td>ReEvo</td></tr><tr><td>ts225</td><td>16.8</td><td>7.7</td><td>6.6</td></tr><tr><td>rat99</td><td>21.8</td><td>14.1</td><td>12.4</td></tr><tr><td>rl1889</td><td>23.7</td><td>21.1</td><td>17.5</td></tr><tr><td>u1817</td><td>22.2</td><td>21.2</td><td>16.6</td></tr><tr><td>d1655</td><td>23.9</td><td>18.7</td><td>17.5</td></tr><tr><td>bier127</td><td>23.3</td><td>15.6</td><td>10.8</td></tr><tr><td>lin318</td><td>25.8</td><td>14.3</td><td>16.6</td></tr><tr><td>eil51</td><td>32.0</td><td>10.2</td><td>6.5</td></tr><tr><td>d493</td><td>24.0</td><td>15.6</td><td>13.4</td></tr><tr><td>kroB100</td><td>26.3</td><td>14.1</td><td>12.2</td></tr><tr><td>kroC100</td><td>25.8</td><td>16.2</td><td>15.9</td></tr></table>

<table><tr><td>Instance</td><td>Nearest Neighbour</td><td>GHPP [15]</td><td>ReEvo</td></tr><tr><td>ch130</td><td>25.7</td><td>14.8</td><td>9.4</td></tr><tr><td>pr299</td><td>31.4</td><td>18.2</td><td>20.6</td></tr><tr><td>fl417</td><td>32.4</td><td>22.7</td><td>19.2</td></tr><tr><td>d657</td><td>29.7</td><td>16.3</td><td>16.0</td></tr><tr><td>kroA150</td><td>26.1</td><td>15.6</td><td>11.6</td></tr><tr><td>fl1577</td><td>25.0</td><td>17.6</td><td>12.1</td></tr><tr><td>u724</td><td>28.5</td><td>15.5</td><td>16.9</td></tr><tr><td>pr264</td><td>17.9</td><td>24.0</td><td>16.8</td></tr><tr><td>pr226</td><td>24.6</td><td>15.5</td><td>18.0</td></tr><tr><td>pr439</td><td>27.4</td><td>21.4</td><td>19.3</td></tr><tr><td>Avg. opt. gap</td><td>25.4</td><td>16.7</td><td>14.6</td></tr></table>

the attention scores, which improves the generalization of NCO solvers without additional training. However, the expert-designed attention-reshaping can be suboptimal and does not generalize across neural models or problem distributions.

Here we show that ReEvo can automatically and efficiently tailor attention reshaping for specific neural models and problem distributions of interest. We apply attention reshaping designed by experts [85] and ReEvo to two distinct model architectures: POMO with heavy encoder and light decoder [37], and LEHD with light encoder and heavy decoder [51]. On TSP and CVRP, Table 3 compares the original NCO solvers [37, 51], those with expert-designed attention reshaping [85], and those with ReEvo-designed attention reshaping. The results reveal that the ReEvo-generated heuristics can improve the original models and outperform their expert-designed counterparts. Note that implementing ReEvo-generated attention reshaping takes negligible additional time; e.g., solving a CVRP1000 with LEHD takes 50.0 seconds with reshaping, compared to 49.8 seconds without.

# 6 Evaluating ReEvo

# 6.1 Fitness landscape analysis

The fitness landscape of a searching algorithm depicts the structure and characteristics of its search space  $F: H \to \mathbb{R}$  [59]. This understanding is essential for designing effective HHs. Here we introduce this technique to LHHs and evaluate the impact of reflections on the fitness landscape.

Traditionally, the neighborhood of a solution is defined as a set of solutions that can be reached after a single move of a certain heuristic. However, LHHs feature a probabilistic nature and open-ended search space, and we redefine its neighborhood as follows.

Definition 6.1 (Neighborhood). Let  $LLM$  denote an LHH move,  $x$  a specific prompt, and  $h_c$  the current heuristic. Given  $LLM$  and  $x$ , the neighborhood of  $h_c$  is defined as a set  $\mathcal{N}$ , where each

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/6cb5dd095168dfcf7f6d14eaeffeee42f3b6045b95eb69ee44af34713ccc6ffe.jpg)  
Figure 3: Left: Comparison of DevFormer [31], the expert-designed GA [63] and our ReEvo-designed GA on DPP. The evolution curves plot the best objective value over generations; the horizontal line indicates the reward of end-to-end solutions generated by DevFormer. Right: Evaluation results of DPP solvers. We report the number of solution generations and the average objective value of 100 test problems.

<table><tr><td>Method</td><td># of shots</td><td>Obj. ↑</td></tr><tr><td>Pointer-PG [30]</td><td>10,000</td><td>9.66 ± 0.206</td></tr><tr><td>AM-PG [60]</td><td>10,000</td><td>9.63 ± 0.587</td></tr><tr><td>CNN-DQN [61]</td><td>10,000</td><td>9.79 ± 0.267</td></tr><tr><td>CNN-DDQN [100]</td><td>10,000</td><td>9.63 ± 0.150</td></tr><tr><td>Pointer-CRL [30]</td><td>Zero Shot</td><td>9.59 ± 0.232</td></tr><tr><td>AM-CRL [62]</td><td>Zero Shot</td><td>9.56 ± 0.471</td></tr><tr><td>DevFormer-CSE [31]</td><td>Zero Shot</td><td>12.88 ± 0.003</td></tr><tr><td>GA-expert [63]</td><td>400</td><td>12.41 ± 0.026</td></tr><tr><td>GA-ReEvo (ours)</td><td>400</td><td>12.98 ± 0.018</td></tr></table>

Table 3: Evaluation results for NCO solvers with and without different attention-reshaping heuristics.  

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">n = 200</td><td colspan="2">n = 500</td><td colspan="2">n = 1000</td></tr><tr><td>Obj.</td><td>Opt. gap (%)</td><td>Obj.</td><td>Opt. gap (%)</td><td>Obj.</td><td>Opt. gap (%)</td></tr><tr><td rowspan="6">TSP</td><td>POMO [37]</td><td>11.16</td><td>4.40</td><td>22.21</td><td>34.43</td><td>35.19</td><td>52.11</td></tr><tr><td>POMO + DAR [85]</td><td>11.12</td><td>3.98</td><td>21.63</td><td>30.95</td><td>33.32</td><td>44.05</td></tr><tr><td>POMO + ReEvo [75]</td><td>11.12</td><td>4.02</td><td>20.54</td><td>24.32</td><td>29.86</td><td>29.08</td></tr><tr><td>LEHD [51]</td><td>10.79</td><td>0.87</td><td>16.78</td><td>1.55</td><td>23.87</td><td>3.17</td></tr><tr><td>LEHD + DAR [85]</td><td>10.79</td><td>0.89</td><td>16.79</td><td>1.62</td><td>23.87</td><td>3.19</td></tr><tr><td>LEHD + ReEvo</td><td>10.77</td><td>0.74</td><td>16.78</td><td>1.55</td><td>23.82</td><td>2.97</td></tr><tr><td rowspan="6">CVRP</td><td>POMO [37]</td><td>22.39</td><td>10.93</td><td>50.12</td><td>33.76</td><td>145.40</td><td>289.48</td></tr><tr><td>POMO + DAR [85]</td><td>22.36</td><td>10.78</td><td>50.23</td><td>34.05</td><td>144.24</td><td>286.37</td></tr><tr><td>POMO + ReEvo</td><td>22.30</td><td>10.48</td><td>47.10</td><td>25.70</td><td>118.80</td><td>218.22</td></tr><tr><td>LEHD [51]</td><td>20.92</td><td>3.68</td><td>38.61</td><td>3.03</td><td>39.12</td><td>4.79</td></tr><tr><td>LEHD + DAR [85]</td><td>21.13</td><td>4.67</td><td>39.16</td><td>4.49</td><td>39.70</td><td>6.35</td></tr><tr><td>LEHD + ReEvo</td><td>20.85</td><td>3.30</td><td>38.57</td><td>2.94</td><td>39.11</td><td>4.76</td></tr></table>

element  $h \in \mathcal{N}$  represents a heuristic that  $LLM$  can mutate  $h_c$  into, in response to  $x$ :

$$
\mathcal {N} \left(h _ {c}\right) = \{h \mid L L M \left(h \mid h _ {c}, x\right) > \xi \}. \tag {1}
$$

Here,  $LLM(h|h_c,x)$  denotes the probability of generating  $h$  after prompting with  $h_c$  and  $x$ , and  $\xi$  is a small threshold value. In practice, the neighborhood can be approximated by sampling from the distribution  $LLM(\cdot|h_c,x)$  for a large number of times.

We extend the concept of autocorrelation to LHHs under our definition of neighborhood. Autocorrelation reflects the ruggedness of a landscape, indicating the difficulty of a COP [59, 22].

Definition 6.2 (Autocorrelation). Autocorrelation measures the correlation structure of a fitness landscape. It is derived from the autocorrelation function  $r$  of a time series of fitness values, which are generated by a random walk on the landscape via neighboring points:

$$
r _ {i} = \frac {\sum_ {t = 1} ^ {T - i} \left(f _ {t} - \bar {f}\right) \left(f _ {t + i} - \bar {f}\right)}{\sum_ {t = 1} ^ {T} \left(f _ {t} - \bar {f}\right) ^ {2}}, \tag {2}
$$

where  $\bar{f}$  is the mean fitness of the points visited,  $T$  is the size of the random walk, and  $i$  is the time lag between points in the walk.

Based on the autocorrelation function, correlation length is defined below [86].

Definition 6.3 (Correlation Length). Given an autocorrelation function  $r$ , the correlation length  $l$  is formulated as  $l = -1 / \ln(|r_1|)$  for  $r_1 \neq 0$ . It reflects the ruggedness of a landscape, and smaller values indicate a more rugged landscape.

To perform autocorrelation analysis for ReEvo, we conduct random walks based on the neighborhood established with our crossover prompt either with or without short-term reflections. In practice, we set the population size to 1 and skip invalid heuristics; the selection always picks the current and last heuristics for short-term reflection and crossover, and we do not implement mutation.

Table 4 presents the correlation length and the average objective value of the random walks, where we generate ACO heuristics for TSP50. The correlation length is averaged over 3 runs each with 40 random walk steps, while the objective value is averaged over all  $3 \times 40$  heuristics. The results verify that implementing reflection leads to a less

rugged landscape and better search results. As discussed in  $\S 4$ , reflections can function as verbal gradients that lead to better neighborhood structures.

Table 4: Autocorrelation analysis of ReEvo.  

<table><tr><td></td><td>Correlation length ↑</td><td>Objective ↓</td></tr><tr><td>w/o reflection</td><td>0.28 ± 0.07</td><td>12.08 ± 7.15</td></tr><tr><td>w/ reflection</td><td>1.28 ± 0.62</td><td>6.53 ± 0.60</td></tr></table>

# 6.2 Ablation studies

In this section, we investigate the effects of the proposed components of ReEvo with both white and black-box prompting.

Black-box prompting. We do not reveal any information related to the COPs and prompt LHHs in general forms (e.g., edge_attr in place of distance_matrix). Black-box settings allow reliable evaluations of LHHs in designing effective heuristics for novel and complex problems, rather than merely retrieving code tailored for prominent COPs from their parameterized knowledge.

We evaluate sampling LLM generations without evolution (LLM) and ReEvo without long-term reflections, short-term reflections, crossover, or mutation on generating ACO heuristics for TSP100. Table 5 shows that ReEvo enhances sample efficiency, and all its components positively contribute to its performance, both in white-box and black-box prompting.

# 6.3 Comparative evaluations

Table 5: Ablation study of ReEvo components with both white and black-box prompting.  

<table><tr><td>Method</td><td>White-box ↓</td><td>Black-box ↓</td></tr><tr><td>LLM</td><td>8.64 ± 0.13</td><td>9.74 ± 0.54</td></tr><tr><td>w/o long-term reflections</td><td>8.61 ± 0.21</td><td>9.32 ± 0.71</td></tr><tr><td>w/o short-term reflections</td><td>8.46 ± 0.01</td><td>9.05 ± 0.83</td></tr><tr><td>w/o crossover</td><td>8.45 ± 0.02</td><td>9.47 ± 1.40</td></tr><tr><td>w/o mutation</td><td>8.83 ± 0.09</td><td>9.34 ± 0.96</td></tr><tr><td>ReEvo</td><td>8.40 ± 0.02</td><td>8.96 ± 0.82</td></tr></table>

This section compares ReEvo with EoH [47], a recent SOTA LHH that is more sample-efficient than FunSearch [68]. We adhere to the original code and (hyper)parameters of EoH. Our experiments apply both LHHs to generate ACO heuristics for TSP, CVRP, OP, MKP, and BPP, using black-box prompting and three LLMs: GPT-3.5 Turbo, GPT-4 Turbo, and Llama 3 (70B).

Fig. 4 compares EoH and ReEvo, and shows that ReEvo demonstrates superior sample efficiency. Besides the better neighborhood structure (§ 6.1), reflections facilitate explicit verbal inference of underlying black-box COP structures; we depict an example in Fig. 1 (b). The enhanced sample efficiency and inference capabilities of ReEvo are particularly useful for complex real-world problems, where the objective function is usually black-box and expensive to evaluate.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/7a71eae5ca0b46f7f21d6a6f0d33a7e23ea6dfbf044f42255f23cda8de50c076.jpg)  
(a) LHH evolution results using different LLMs.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-23/14d379b9-88f0-4db2-aad7-587914e7d2d2/04b820a5ac69ba5d5270a7e442a0d2e302788fd66aeee57a30d381c5255824e6.jpg)  
(b) LHH evolution curves using GPT-3.5 Turbo.  
Figure 4: Comparisons between EoH [47] and ReEvo on five COPs with black-box prompting and using different LLMs. We perform three runs for each setting.

# 7 Discussions and limitations

When to use ReEvo as an LHH. Our experiments limit the number of heuristic evaluations to 100 shots and the results do not necessarily scale up. ReEvo is designed for scenarios where sample efficiency is crucial, such as real-world applications where heuristic evaluation can be costly. Allowing a large number of heuristic evaluations could obscure the impact of reflection or other prompting techniques, as reported by Zhang et al. [101].

When to use ReEvo as an (alternative to) NCO/ML4CO method. LHH can be categorized as an NCO/ML4CO method. However, to facilitate our discussion, we differentiate LHHs from

"traditional" NCO methods that usually train NN-parameterized heuristics via parameter adjustment. In § 5, we demonstrate that ReEvo can either outperform or enhance NCO methods. Below, we explore the complementary nature of LHH and NCO methods.

- **Rule-based vs. NN-parameterized policies.** LHHs generate interpretable and rule-based heuristics (code snippets), while NCO generates black-box NN-parameterized policies. Interpretable heuristics offer insights for human designers and can be more reliable in practice when faced with dynamic environments, limited data, distributional shifts, or adversarial attacks. However, they may not be as expressive as neural networks and may underfit in complex environments.  
- Evolution and training. LHHs require only less than 100 heuristic evaluations and about 5 minutes to evolve a strong heuristic, while many NCO methods usually require millions of samples and days of training. LHHs are more practical when solution evaluation is expensive.  
- Inference. LHHs generate heuristics that are less demanding in terms of computational resources, as they do not require GPU during deployment. NCO methods require GPU for training and deployment, but they can also leverage the parallelism of GPU to potentially speed up inference.  
- Engineering efforts and inductive biases. LHHs only need some text-based (and even black-box) explanations to guide the search. NCO requires the development of NN architectures, hyperparameters, and training strategies, where informed inductive biases and manual tuning are crucial to guarantee performance.

The choice of LLMs for ReEvo. Reflection is more effective when using capable LLMs, such as GPT-3.5 Turbo and its successors, as discussed by Shinn et al. [70]. Currently, many open-source LLMs are not capable enough to guarantee statistically significant improvement of reflections [101]. However, as LLM capabilities improve, we only expect this paradigm to get better over time [70]. One can refer to [101] for extended evaluations based on more LLMs and problem settings.

Benchmarking LHHs based on heuristic evaluations. We argue that benchmarking LHHs should prioritize the number of heuristic evaluations rather than LLM query budgets [101] due to the following reasons.

- Prioritizing scenarios where heuristic evaluations are costly leads to meaningful comparisons between LHHs. The performance of different LHH methods becomes nearly indistinguishable when a large number of heuristic evaluations are allowed [101].  
- The overhead of LLM queries is negligible compared to real-world heuristic evaluations. LLM inference—whether via local models or commercial APIs—is highly cost-effective nowadays, with expenses averaging around $0.0003 per call in ReEvo using GPT-3.5-turbo, and response times of under one second on average for asynchronous API calls or batched inference. These costs are negligible compared to real-world heuristic evaluations, which, taking the toy EDA problem in this paper as an example, exceeds 20 minutes per evaluation.  
- Benchmarking LHHs based on LLM inference costs presents additional challenges. Costs and processing time are driven by token usage rather than the number of queries, complicating the benchmarking process. For instance, EoH [47] requires heuristic descriptions before code generation, resulting in higher token usage. In contrast, although ReEvo involves more queries for reflections, it is more token-efficient when generating heuristics.

# 8 Conclusion

This paper presents Language Hyper-Heuristics (LHHs), a rising variant of HHs, alongside Reflective Evolution (ReEvo), an evolutionary framework to elicit the power of LHHs. Applying ReEvo across five heterogeneous algorithmic types, six different COPs, and both white-box and black-box views of COPs, we yield state-of-the-art and competitive meta-heuristics, evolutionary algorithms, heuristics, and neural solvers. Comparing against SOTA LHH [47], ReEvo demonstrates superior sample efficiency. The development of LHHs is still at its emerging stage. It is promising to explore their broader applications, better dual-level optimization architectures, and theoretical foundations. We also expect ReEvo to enrich the landscape of evolutionary computation, by showing that genetic cues can be interpreted and verbalized using LLMs.

# Acknowledgments and disclosure of funding

We are very grateful to Yuan Jiang, Yining Ma, Yifan Yang, AI4CO community, anonymous reviewers, and the area chair for valuable discussions and feedback. This work was supported by the National Natural Science Foundation of China (Grant No. 62276006); Wuhan East Lake High-Tech Development Zone National Comprehensive Experimental Base for Governance of Intelligent Society; the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG3-RP-2022-031); the Institute of Information & Communications Technology Planning & Evaluation (IITP)-Innovative Human Resource Development for Local Intellectualization program grant funded by the Korea government (MSIT) (IITP-2024-RS-2024-00436765).

# References

[1] F. Arnold and K. Sorensen. Knowledge-guided local search for the vehicle routing problem. Computers & Operations Research, 105:32-46, 2019.  
[2] Y. Bengio, A. Lodi, and A. Prouvost. Machine learning for combinatorial optimization: a methodological tour d'horizon. European Journal of Operational Research, 290(2):405-421, 2021.  
[3] F. Berto, C. Hua, J. Park, M. Kim, H. Kim, J. Son, H. Kim, J. Kim, and J. Park. RL4CO: a unified reinforcement learning for combinatorial optimization library. In NeurIPS 2023 Workshop: New Frontiers in Graph Learning, 2023.  
[4] E. Brooks, L. A. Walls, R. Lewis, and S. Singh. Large language models can implement policy iteration. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.  
[5] F. Bu, H. Jo, S. Y. Lee, S. Ahn, and K. Shin. Tackling prevalent conditions in unsupervised combinatorial optimization: Cardinality, minimum, covering, and more. arXiv preprint arXiv:2405.08424, 2024.  
[6] J. Cai, P. Wang, S. Sun, and H. Dong. A dynamic space reduction ant colony optimization for capacitated vehicle routing problem. Soft Computing, 26(17):8745-8756, 2022.  
[7] W. Chao, J. Zhao, L. Jiao, L. Li, F. Liu, and S. Yang. A match made in consistency heaven: when large language models meet evolutionary algorithms, 2024.  
[8] A. Chen, D. M. Dohan, and D. R. So. Evoprompting: Language models for code-level neural architecture search. arXiv preprint arXiv:2302.14838, 2023.  
[9] J. Chen, J. Wang, Z. Zhang, Z. Cao, T. Ye, and C. Siyuan. Efficient meta neural heuristic for multi-objective combinatorial optimization. In Advances in Neural Information Processing Systems, 2023.  
[10] X. Chen, M. Lin, N. Scharli, and D. Zhou. Teaching large language models to self-debug. arXiv preprint arXiv:2304.05128, 2023.  
[11] F. de Paulis, R. Cecchetti, C. Olivieri, and M. Buecker. Genetic algorithm pdn optimization based on minimum number of decoupling capacitors applied to arbitrary target impedance. In 2020 IEEE International Symposium on Electromagnetic Compatibility & Signal/Power Integrity (EMCSI), pages 428-433. IEEE, 2020.  
[12] T. Dernedde, D. Thyssens, S. Dittrich, M. Stubbemann, and L. Schmidt-Thieme. Moco: A learnable meta optimizer for combinatorial optimization. arXiv preprint arXiv:2402.04915, 2024.  
[13] J. H. Drake, A. Kheiri, E. Ozcan, and E. K. Burke. Recent advances in selection hyperheuristics. European Journal of Operational Research, 285(2):405-428, 2020.  
[14] D. Drakulic, S. Michel, F. Mai, A. Sors, and J.-M. Andreoli. Bq-nco: Bisimulation quotienting for efficient neural combinatorial optimization. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.