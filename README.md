# 📊 Empirical Employee Attrition Modeling & High-Dimensional Diagnostic Visualization

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Deep%20Learning-PyTorch%202.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20Evaluation-green.svg)]()
[![Course Code](https://img.shields.io/badge/Course-U21ADP05%20(EDA%20%26%20Vis)-orange.svg)]()
[![Institution](https://img.shields.io/badge/Institution-KPR%20Institute%20of%20Engineering%20%26%20Technology-purple.svg)]()

> **Course**: U21ADP05 – Exploratory Data Analysis and Visualization (Assignment II, 40 Marks)  
> **Institution**: Department of Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology  
> **Candidate**: Suja S (`Roll No: 24AD120`) | **Batch**: 2023–2027 | **Year/Sem**: III Year / V Semester  

---

## 📌 Executive Summary

Voluntary employee attrition presents severe organizational vulnerabilities, incurring replacement overhead ranging between **50% and 200% of an individual employee's annualized compensation**. This repository contains a publication-grade, mathematically grounded case study analyzing the real-world **IBM HR Analytics Employee Attrition Dataset** (1,470 employees across 35 attributes). 

Following **Ben Fry's 7 Stages of Data Visualization**, the project implements non-parametric hypothesis testing (Mann-Whitney $U$, $\chi^2$ test of independence), statistical moment diagnostics (skewness, kurtosis), outlier detection via Tukey's Interquartile Range (IQR) fences, and a **Deep Multi-Layer Perceptron (PyTorch MLP)** with regularized architecture and model-agnostic Permutation Feature Importance.

---

## 📂 Repository Architecture

```text
EDA2/
├── figures/                                    # Publication-Grade Figures (300 DPI)
│   ├── fig1_attrition_distribution.png         # Target class imbalance & donut decomposition
│   ├── fig2_continuous_kde_distributions.png   # Density estimation & statistical moments
│   ├── fig3_bivariate_boxplots_mannwhitney.png # Boxplots with Mann-Whitney U p-values
│   ├── fig4_categorical_attrition_chisq.png    # Overtime & Job Role Chi-Square diagnostics
│   ├── fig5_correlation_heatmap.png            # Triangular Pearson correlation matrix
│   ├── fig6_multivariate_pairplot.png          # High-dimensional pairwise interactions
│   ├── fig7_training_val_loss_curve.png        # Training vs. Validation BCE Loss trajectory
│   ├── fig8_training_val_accuracy_curve.png    # Training vs. Validation Accuracy trajectory
│   ├── fig9_confusion_matrix.png               # Out-of-sample Test Confusion Matrix
│   └── fig10_roc_auc_curve.png                 # ROC Benchmark Curve (MLP vs. Logistic Regression)
├── attrition_pipeline.py                       # Standalone, end-to-end reproducible pipeline
├── best_attrition_mlp.pt                       # Serialized optimal PyTorch model weights
├── benchmark_metrics.json                      # Comprehensive experiment logs & metrics
├── emp_attrition.csv                           # Archived IBM HR Attrition dataset
├── REPORT.md                                   # Complete 11-Section Academic Case Study Report
└── README.md                                   # Project documentation & execution guide
```

---

## 🔬 Key Empirical Discoveries

1. **Target Imbalance**: The workforce displays an attrition rate of **16.12% (237 voluntary resignations vs. 1,233 retained staff)**, yielding a class imbalance ratio of **5.2 : 1**.
2. **Work-Life Depletion**: Mandatory `OverTime` is the single strongest empirical predictor ($\chi^2 = 87.56, p = 8.16 \times 10^{-21}$). Employees working overtime exhibit an attrition rate of **30.5%**, compared to **10.4%** for non-overtime staff.
3. **Compensation Gradient**: Attriters earn a median monthly income of **\$3,202**, compared to **\$5,204** for retained colleagues (Mann-Whitney $U = 100,620.5, p = 2.95 \times 10^{-14}$).
4. **Vulnerable Job Roles**: Sales Representatives (39.8%) and Laboratory Technicians (23.9%) undergo severe turnover compared to Research Directors (2.5%) and Healthcare Representatives (6.9%).
5. **Model Performance**: The Deep Learning MLP achieves **84.16% test accuracy**, an **ROC-AUC of 0.8284**, and **66.67% sensitivity (recall)**, successfully identifying two-thirds of at-risk employees, outperforming the linear baseline's 44.44% recall.

---

## 📈 Empirical Benchmark Comparison

Evaluated on the strictly holdout out-of-sample test partition ($N = 221$, stratified 15%):

| Evaluation Metric | Deep Learning MLP (PyTorch) | Regularized Logistic Regression | Clinical / Strategic Advantage |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **84.16%** | **87.33%** | High baseline discrimination across full sample |
| **Precision** | **51.06%** | **66.67%** | Controlled false alarm intervention rate |
| **Recall (Sensitivity)** | **66.67% (24 / 36)** | **44.44% (16 / 36)** | **+22.23% lift in capturing true leavers** |
| **Specificity** | **87.57% (162 / 185)** | **95.68% (177 / 185)** | Preserves non-attriter operational baseline |
| **F1-Score** | **0.5783** | **0.5333** | **+0.0450 superior harmonic balance** |
| **ROC-AUC** | **0.8284** | **0.8407** | Strong ranking discrimination above random chance |
| **True Positives (TP)** | **24** | **16** | 8 additional flight-risk employees protected |
| **False Negatives (FN)**| **12** | **20** | **40% reduction in catastrophic undetected turnover** |

---

## 🚀 Reproduction & Setup Guide

### 1. Clone & Navigate
```bash
git clone https://github.com/SujaS-24AD120/IBM-HR-Attrition-EDA-DL.git
cd IBM-HR-Attrition-EDA-DL
```

### 2. Environment Setup
Create an isolated Python 3.10+ virtual environment and install dependencies:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install torch numpy pandas scipy matplotlib seaborn scikit-learn
```

### 3. Run the Automated Pipeline
Execute the master reproducible script:
```bash
python attrition_pipeline.py
```
This single script will:
- Download and validate the raw IBM HR dataset.
- Calculate Tukey's IQR fences and compute statistical hypothesis tests ($U$-test and $\chi^2$).
- Generate all 10 high-resolution figures (`figures/fig1` through `figures/fig10`).
- Execute hyperparameter tuning across candidate MLP configurations.
- Train the optimal deep neural network, checkpoint weights to `best_attrition_mlp.pt`.
- Perform permutation feature importance over 5 repeats.
- Export all diagnostic metrics into `benchmark_metrics.json`.

---

## 📊 Visual Gallery

| Figure Index & Title | Visualization Preview |
| :--- | :--- |
| **Figure 1**: Target Class Distribution & Donut Ratio | `figures/fig1_attrition_distribution.png` |
| **Figure 2**: Continuous Probability Density (Moments) | `figures/fig2_continuous_kde_distributions.png` |
| **Figure 3**: Bivariate Boxplots & Mann-Whitney $U$ Tests | `figures/fig3_bivariate_boxplots_mannwhitney.png` |
| **Figure 4**: Categorical Contingency & Chi-Square Tests | `figures/fig4_categorical_attrition_chisq.png` |
| **Figure 5**: Lower-Bound Pearson Correlation Heatmap | `figures/fig5_correlation_heatmap.png` |
| **Figure 6**: Multivariate High-Dimensional Pairplot | `figures/fig6_multivariate_pairplot.png` |
| **Figure 7**: Training vs. Validation Loss Trajectory | `figures/fig7_training_val_loss_curve.png` |
| **Figure 8**: Training vs. Validation Accuracy Trajectory | `figures/fig8_training_val_accuracy_curve.png` |
| **Figure 9**: Out-of-Sample Test Confusion Matrix | `figures/fig9_confusion_matrix.png` |
| **Figure 10**: Receiver Operating Characteristic (ROC-AUC) | `figures/fig10_roc_auc_curve.png` |

---

## 📜 Academic Integrity & Attribution
This project was authored independently by **Suja S** (`Roll No: 24AD120`) for the course **U21ADP05 – Exploratory Data Analysis and Visualization**, Department of Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology. All code, statistical formulations, and visual representations are original and reproducible.
