# WKL
Code for the paper: **Stage-Adaptive Knowledge Distillation via KL-Wasserstein Hybrid Distribution Alignment**

**Abstract**: Knowledge distillation typically aligns the teacher and student output distributions using Kullback-Leibler (KL) divergence. Recent studies, however, have shown that Wasserstein distance can offer complementary geometric alignment. Yet, the behavior of these two divergences across different stages of distillation remains insufficiently characterized, limiting principled designs that leverage their respective strengths. In this paper, we first develop a set of quantitative alignment diagnostics that decompose the teacher-student discrepancy into target-class and non-target-class components. We then analyze their stage-dependent optimization behaviors from both loss and gradient perspectives. Based on these observations, we propose a stage-adaptive mixed distillation objective that smoothly interpolates between Wasserstein distance and KL divergence via a dynamic weighting schedule. The resulting training strategy emphasizes global geometric alignment in the early epochs and progressively shifts toward local, fine-grained probability matching in the later epochs, leading to more coordinated teacher-student distribution alignment. Extensive experiments on CIFAR-100 and ImageNet datasets demonstrate that the proposed method consistently improves student performance across diverse settings, outperforming distillation baselines using either KL divergence or Wasserstein distance alone, and yielding favorable results compared to recent Wasserstein-based distillation approaches. The code is available at [https://github.com/Chang-Heyu/WKL](https://github.com/Chang-Heyu/WKL).

![Overview of the proposed stage-adaptive mixed distillation objective](figures/fig-main.jpg)  
*Fig. 1: Overview of the proposed stage-adaptive mixed distillation objective. The training process transitions from Wasserstein-dominant global structure alignment to KL-dominant local detail refinement via a dynamic weighting schedule.*

### Loss Evolution Analysis

<div style="display: flex; flex-wrap: wrap; gap: 10px;"> 
  <div style="flex: 1 1 48%;">
    <img src="figures/loss_kl.png" width="20%" />  
    <p>(a) loss_kl</p>
  </div>
  <div style="flex: 1 1 48%;">
    <img src="figures/loss_tc.png" width="20%" />  
    <p>(b) loss_tc</p>
  </div>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <div style="flex: 1 1 48%;">
    <img src="figures/loss_nc.png" width="48%" />  
    <p>(c) loss_nc</p>
  </div>
  <div style="flex: 1 1 48%;">
    <img src="figures/loss_w.png" width="48%" />  
    <p>(d) loss_w</p>
  </div>
</div>
