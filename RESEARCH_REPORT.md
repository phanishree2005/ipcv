# $M^2$-ReID: Adaptive Multi-Model Cross-Camera Person Re-Identification via Quality-Conditioned Neural Gating and Spatio-Temporal Consistency

**Abstract**  
Person Re-Identification (Re-ID) across distributed, non-overlapping camera networks faces critical challenges stemming from severe viewpoint disparities, illumination variations, synthetic occlusions, and background clutter. While existing approaches rely either on purely convolutional feature extractors (CNNs) or vision transformers (ViTs), individual model paradigms exhibit distinct failure modes under unconstrained capture conditions. In this research work, we propose **$M^2$-ReID** (*Adaptive Multi-Model Person Re-Identification*), a unified cross-camera retrieval framework that adaptively harmonizes representations from:
1. Multi-scale convolutional feature learning (**OSNet**),
2. Vision Transformer patch-based self-attention with side-information embeddings (**TransReID**), and
3. Zero-shot visual representations from a Vision-Language foundation model (**CLIP**).

Our primary research contribution is an **Adaptive Multi-Model Feature Fusion Module** with a neural gating mechanism conditioned on a 7-dimensional image quality vector (sharpness, brightness, contrast, detection confidence, bbox area, aspect ratio, occlusion score) and a 17-joint anatomical human pose representation. The gating network dynamically estimates sample-specific softmax attention weights ($w_{osnet} + w_{transreid} + w_{clip} = 1$), dynamically down-weighting degraded modalities and amplifying resilient features. Retrieved gallery candidates from a high-throughput **FAISS vector database** are subsequently refined via **Multi-Context Re-Ranking** incorporating spatio-temporal Gaussian travel-time likelihoods and camera transition priors. On the **Market-1501** benchmark, $M^2$-ReID achieves **98.1% Rank-1 accuracy** and **94.2% mAP**, outperforming the single-model OSNet baseline by $+3.3\%$ Rank-1 and $+9.3\%$ mAP. Furthermore, cross-dataset generalization experiments on the **MSMT17** benchmark demonstrate superior domain-shift resilience.

---

## 1. Introduction

Person Re-Identification (Re-ID) aims to identify and match individual pedestrians across non-overlapping surveillance camera feeds at different timestamps. Modern Re-ID faces four fundamental visual challenges:
- **Intra-Class Variation:** Significant variations in lighting, pedestrian orientation, scale, and body posture across camera viewpoints.
- **Inter-Class Ambiguity:** Different pedestrians wearing similar clothing color schemes in crowded environments.
- **Camera Domain Discrepancy:** Differences in sensor sensitivity, optical distortion, and exposure between cameras.
- **Occlusion and Resolution Degradation:** Distant camera views leading to blurred, low-resolution crops with partial obstructions.

Existing research has primarily explored individual architectural paradigms:
1. *CNNs (e.g., OSNet)* capture local spatial hierarchies and omni-scale receptive fields effectively.
2. *Transformers (e.g., TransReID)* excel at global patch self-attention and viewpoint token conditioning.
3. *Vision-Language Models (e.g., CLIP)* encode robust, semantic visual representations trained on web-scale datasets.

However, each individual representation possesses specific blind spots. For instance, Vision Transformers may degrade under extreme occlusions or small crop sizes, whereas CNNs struggle with drastic viewpoint shifts. Fixed linear fusion fails because crop quality and environmental degradation vary dynamically per image.

To address these limitations, $M^2$-ReID introduces an **Adaptive Multi-Model Feature Fusion** architecture conditioned on real-time image quality metrics and skeletal pose vectors, integrated with a FAISS vector index, spatial-temporal camera topology modeling, and a Global Identity Manager.

---

## 2. Methodology & Mathematical Formulation

### 2.1 Multi-Model Representation Extraction
Given an input cropped person image $I \in \mathbb{R}^{H \times W \times 3}$ and associated camera identifier $c \in \{1, \dots, C\}$:
- **OSNet:** Generates multi-scale convolutional embedding $f_{os} = \mathcal{F}_{OS}(I) \in \mathbb{R}^{512}$.
- **TransReID:** Generates transformer token representation with side-information camera embedding $f_{tr} = \mathcal{F}_{TR}(I, c) \in \mathbb{R}^{768}$.
- **CLIP Visual Encoder:** Generates zero-shot foundation visual embedding $f_{cl} = \mathcal{F}_{CLIP}(I) \in \mathbb{R}^{512}$.

Each embedding is projected into a common fused dimension $D = 512$:
$$\tilde{f}_{os} = \phi_{os}(f_{os}), \quad \tilde{f}_{tr} = \phi_{tr}(f_{tr}), \quad \tilde{f}_{cl} = \phi_{cl}(f_{cl})$$

### 2.2 Image Quality Vector ($\mathbf{q} \in \mathbb{R}^7$)
To inform the neural gating network of degradation severity, we extract a 7-dimensional metric vector:
1. **Laplacian Blur Variance:** $q_1 = 1 - \exp(-\text{Var}(\nabla^2 I_{gray}) / \sigma_{blur})$
2. **Normalized Brightness:** $q_2 = 1 - 2|\mu_{intensity} - 0.5|$
3. **Contrast Score:** $q_3 = 1 - \exp(-2\sigma_{intensity})$
4. **Detector Confidence:** $q_4 = \text{Conf}_{det} \in [0, 1]$
5. **Bounding Box Area Ratio:** $q_5 = \text{Area}_{crop} / \text{Area}_{frame}$
6. **Aspect Ratio Deviation:** $q_6 = \exp(-|H/W - 2.0|)$
7. **Spatial Occlusion Score:** $q_7 = 1 - \min(\text{Var}_{segments}) / \max(\text{Var}_{segments})$

### 2.3 Pose Keypoint Representation ($\mathbf{p} \in \mathbb{R}^{34}$)
17 COCO skeletal keypoints $\mathbf{k}_i = (x_i, y_i) \in [0, 1]^2$ are extracted for Head, Shoulders, Elbows, Wrists, Hips, Knees, and Ankles, normalized relative to the crop center, producing $\mathbf{p} = [\mathbf{k}_1, \dots, \mathbf{k}_{17}] \in \mathbb{R}^{34}$.

### 2.4 Adaptive Neural Gating Mechanism
The gating network $\mathcal{G}$ accepts the concatenated representation:
$$\mathbf{z} = [\tilde{f}_{os} \,\|\, \tilde{f}_{tr} \,\|\, \tilde{f}_{cl} \,\|\, \psi_q(\mathbf{q}) \,\|\, \psi_p(\mathbf{p})]$$
and computes dynamic logits $\mathbf{g} = [g_1, g_2, g_3] \in \mathbb{R}^3$. Softmax normalization yields sample-specific weights:
$$w_i = \frac{\exp(g_i / \tau)}{\sum_{j=1}^3 \exp(g_j / \tau)}, \quad \sum_{i=1}^3 w_i = 1.0$$
The initial fused representation is formed as:
$$f_{init} = w_1 \tilde{f}_{os} + w_2 \tilde{f}_{tr} + w_3 \tilde{f}_{cl}$$
Followed by a residual quality-pose refinement block $\mathcal{R}$:
$$f_{fused} = \text{L2-Norm}\left(f_{init} + \mathcal{R}([f_{init} \,\|\, \psi_q(\mathbf{q}) \,\|\, \psi_p(\mathbf{p})])\right)$$

### 2.5 Training Objectives
The adaptive fusion network is trained end-to-end using a combined objective:
$$\mathcal{L}_{total} = \mathcal{L}_{CE\_smooth}(y, \hat{y}) + \mathcal{L}_{Triplet}(f_{fused}, y) + \lambda_{reg} \mathcal{H}(\mathbf{w})$$
where $\mathcal{L}_{Triplet}$ uses Batch-Hard mining with margin $\alpha=0.3$, and $\mathcal{H}(\mathbf{w}) = -\sum_{i=1}^3 w_i \log(w_i)$ regularizes weight entropy.

### 2.6 Spatial-Temporal Multi-Context Re-Ranking
For gallery retrieval, candidate similarity from FAISS is adjusted via physical transition likelihoods:
$$S_{final} = w_{app} S_{app} + w_{pose} S_{pose} + w_{temp} S_{temp}(\Delta t) + w_{cam} P(c_q \to c_g) + w_{qual} S_{qual}$$
where $S_{temp}(\Delta t) = \exp\left(-\frac{1}{2} \left(\frac{\Delta t - \mu_{ij}}{\sigma_{ij}}\right)^2\right)$ models physical travel feasibility between cameras.

---

## 3. Experimental Evaluation

### 3.1 Benchmark Datasets
1. **Market-1501:** 1,501 identities across 6 cameras (751 train IDs, 750 test IDs, 3,368 query images, 19,732 gallery images).
2. **MSMT17:** 4,101 identities across 15 cameras (1,041 train IDs, 3,060 test IDs, 126,441 images across morning, noon, and afternoon).

### 3.2 Quantitative Results (Experiments E1 to E10 on Market-1501)

| Exp ID | Method / Architecture | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mAP (%) | Latency (ms) | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | OSNet (CNN Baseline) | 94.8 | 98.2 | 98.9 | 84.9 | 14.2 | 70.4 |
| **E2** | TransReID (ViT Baseline) | 95.2 | 98.6 | 99.1 | 88.8 | 28.5 | 35.1 |
| **E3** | CLIP (Vision-Language) | 89.4 | 95.8 | 97.2 | 78.3 | 22.0 | 45.5 |
| **E4** | OSNet + TransReID (Pairwise) | 95.9 | 98.8 | 99.2 | 89.7 | 30.2 | 33.1 |
| **E5** | OSNet + CLIP (Pairwise) | 95.1 | 98.3 | 99.0 | 86.4 | 26.5 | 37.7 |
| **E6** | TransReID + CLIP (Pairwise) | 95.6 | 98.7 | 99.1 | 89.1 | 31.8 | 31.4 |
| **E7** | Fixed 3-Model Fusion (0.4/0.4/0.2) | 96.1 | 98.9 | 99.3 | 90.2 | 32.1 | 31.2 |
| **E8** | **Adaptive 3-Model Fusion (Ours)** | **97.4** | **99.2** | **99.6** | **92.6** | 34.8 | 28.7 |
| **E9** | **Adaptive Fusion + Pose** | **97.8** | **99.4** | **99.7** | **93.4** | 35.6 | 28.1 |
| **E10**| **$M^2$-ReID (Full Pipeline)** | **98.1** | **99.5** | **99.8** | **94.2** | 36.5 | 27.4 |

---

## 4. Component Ablation Study

| Ablation Stage | Component Added | Rank-1 (%) | $\Delta$ Rank-1 | mAP (%) | $\Delta$ mAP |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Stage 1** | Baseline (OSNet) | 94.8 | - | 84.9 | - |
| **Stage 2** | + Multi-Model Fixed Fusion | 96.1 | +1.3% | 90.2 | +5.3% |
| **Stage 3** | + Adaptive Neural Gating | 97.4 | +1.3% | 92.6 | +2.4% |
| **Stage 4** | + Pose Feature Conditioning | 97.8 | +0.4% | 93.4 | +0.8% |
| **Stage 5** | + Spatio-Temporal Consistency | 98.0 | +0.2% | 93.9 | +0.5% |
| **Stage 6** | + Multi-Context Re-Ranking | **98.1** | **+0.1%** | **94.2** | **+0.3%** |

---

## 5. Perturbation Robustness Analysis

We subjected probe queries to 5 severe real-world perturbation axes:

| Perturbation Condition | Severity / Setting | OSNet Baseline Rank-1 | $M^2$-ReID (Ours) Rank-1 | Resilience Gain ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Gaussian Blur** | Level 1 (Kernel 3x3) | 90.2% | **95.6%** | +5.4% |
| **Gaussian Blur** | Level 3 (Kernel 7x7) | 78.4% | **88.2%** | **+9.8%** |
| **Low Illumination** | 0.4x Brightness Factor | 81.1% | **91.4%** | **+10.3%** |
| **High Illumination** | 1.6x Brightness Factor | 86.3% | **94.1%** | +7.8% |
| **Low Contrast** | 0.4x Contrast Scaling | 82.5% | **92.0%** | +9.5% |
| **Synthetic Occlusion** | 10% Surface Area | 87.2% | **94.8%** | +7.6% |
| **Synthetic Occlusion** | 25% Surface Area | 74.6% | **87.5%** | **+12.9%** |
| **Low Resolution** | 4x Downsampling | 71.3% | **84.9%** | **+13.6%** |

*Finding:* Under severe occlusions ($25\%$) and low resolution ($4\times$), adaptive gating dynamically shifts higher attention weights to CLIP and OSNet multi-scale features, mitigating catastrophic failure in Vision Transformer tokens.

---

## 6. Cross-Dataset Generalization (Market-1501 $\to$ MSMT17)

| Method | Trained On | Evaluated On | Rank-1 (%) | mAP (%) | Domain Transfer Drop |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **OSNet Baseline** | Market-1501 | MSMT17 | 76.2 | 51.4 | -18.6% Rank-1 |
| **$M^2$-ReID (Ours)** | Market-1501 | MSMT17 | **83.5** | **63.8** | **-14.6% Rank-1 (Robust)** |

*Domain Shift Analysis:* MSMT17 introduces 15 cameras, complex outdoor illumination variations (morning vs afternoon shadows), and diverse architectural backgrounds. The incorporation of CLIP's pre-trained semantic domain invariance alongside learned adaptive gating preserves matching fidelity across extreme camera domain transitions.

---

## 7. Automated Error Diagnostics & Failure Modes

An empirical diagnosis of false matches across the test gallery reveals the following distribution of Rank-1 failure modes:
1. **Similar Clothing Appearance (43.8%):** Confusing color histograms (e.g. solid white shirts and dark jeans across different identities).
2. **Heavy Occlusion (21.9%):** Occlusion by bicycles, luggage, or fellow pedestrians blocking lower limbs.
3. **Lighting & Shadow Disparity (15.6%):** Extreme specular reflection in bright outdoor cameras vs dark corridor views.
4. **Low Resolution / Distant Subject (9.4%):** Pedestrians captured beyond 30 meters.
5. **Extreme Pose / Viewpoint Shift (6.2%):** Direct frontal view matching against direct rear view.
6. **Cross-Camera Ambiguity (3.1%):** Rare synchronization anomalies between distant cameras.

---

## 8. Conclusion

$M^2$-ReID demonstrates that combining complementary CNN, Transformer, and Vision-Language models via condition-aware adaptive neural gating significantly improves Person Re-Identification accuracy, robustness against environmental perturbations, and cross-dataset generalization. The integration of high-speed FAISS vector retrieval, spatial-temporal travel modeling, and an interactive Streamlit dashboard offers an end-to-end framework suitable for research and controlled video analytics.
