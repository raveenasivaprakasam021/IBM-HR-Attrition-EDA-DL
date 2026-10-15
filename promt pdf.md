# INSTRUCTIONS FOR CLAUDE:
Act as a Senior Academic Evaluator and Document Typesetter. Generate a complete, publication-quality academic project report in a single interactive HTML/CSS document (or printable PDF artifact) formatted for A4 printing (with page breaks, running headers, footers with page numbering, elegant academic typography, tables, and callouts).

---

# COMPLETE REPORT CONTENT TO RENDER:

# KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY
### Autonomous Institution | Accredited by NAAC with 'A' Grade | Approved by AICTE, New Delhi
### Department of Artificial Intelligence and Data Science
### Arasur, Coimbatore – 641 407, Tamil Nadu, India

---

## COURSE PROJECT REPORT: ASSIGNMENT II (40 MARKS)
### **Course Code & Title**: U21ADP05 – Exploratory Data Analysis and Visualization
### **Academic Year & Semester**: 2025–2026 | III Year / V Semester | Batch: 2023–2027

---

# EMPIRICAL INVESTIGATION AND DEEP LEARNING ARCHITECTURE FOR VOLUNTARY EMPLOYEE TURNOVER: HIGH-DIMENSIONAL DIAGNOSTIC VISUALIZATION AND PREDICTIVE RISK MODELING

---

### **Student Name**: Raveena S
### **Roll Number**: 24AD087
### **Degree & Branch**: B.Tech. – Artificial Intelligence and Data Science
### **Faculty Evaluator**: Department Course Evaluation Committee
### **GitHub Repository**: https://github.com/raveenasivaprakasam021/IBM-HR-Attrition-EDA-DL
### **Date of Submission**: October 15, 2026

---

### **Course Outcomes (CO) Rubric Mapping**
* **CO3: Data Visualization (8 Marks)** – Demonstrated through publication-quality multi-panel distributions, categorical contingencies, and correlation structures adhering to graphical integrity principles.
* **CO4: Statistical & Distribution-Based Visualization (16 Marks)** – Demonstrated through parametric and non-parametric statistical density estimation, moment diagnostics (skewness, kurtosis), Tukey's IQR fences, Mann-Whitney U tests, and Chi-Square hypothesis testing.
* **CO5: Application of Data Exploration & Visualization (16 Marks)** – Demonstrated through end-to-end execution of Ben Fry's 7 Stages, bivariate and multivariate interaction discovery, and strategic workforce turnover insights.
* **Enrichment Component: Deep Learning Model & Evaluation** – Demonstrated through feed-forward Multi-Layer Perceptron (MLP) formulation, Batch Normalization, Dropout regularization, hyperparameter grid tuning, model-agnostic Permutation Feature Importance, and out-of-sample ROC-AUC benchmark evaluation.

---

## 2. ABSTRACT
Voluntary employee attrition introduces substantial operational friction, intellectual capital depreciation, and severe financial replacement costs within enterprise ecosystems. This academic study presents an end-to-end exploratory data visualization and deep learning diagnostic framework using the canonical IBM HR Analytics Employee Attrition Dataset (N = 1,470 instances, 35 attributes). Rigorously implementing Ben Fry's 7 Stages of Data Visualization, we execute moment diagnostics, Tukey's interquartile range outlier fencing, non-parametric Mann-Whitney U testing, and categorical Chi-Square tests of independence. To advance beyond classical linear modeling, a 3-layer deep feed-forward Multi-Layer Perceptron (PyTorch MLP) with Batch Normalization and Dropout regularization was engineered, hyperparameter-tuned across three architectural variants, and benchmarked against regularized Logistic Regression. The deep architecture achieved 84.16% test accuracy, an ROC-AUC of 0.8284, and an out-of-sample sensitivity (recall) of 66.67%, outperforming the linear baseline by +22.23% in capturing true flight-risk personnel. Permutation feature importance identified mandatory overtime, job involvement, and promotion stagnation as primary turnover drivers.

---

## 3. INTRODUCTION & OBJECTIVES

### 3.1 Economics of Voluntary Workforce Attrition
In modern knowledge economies, human capital constitutes the primary determinant of organizational competitive advantage. Voluntary employee attrition—the unforced cessation of employment initiated by staff members—imposes severe multi-dimensional liabilities on enterprise longevity:
1. **Direct Financial Replacement Costs**: Replacing a skilled worker consumes between 50% and 200% of that individual's annualized base salary (Cascio & Boudreau, 2016).
2. **Productivity Latency & Ramp-Up Sunk Costs**: Newly onboarded replacements require an average ramp-up window of 6 to 9 months before achieving parity with baseline departmental productivity.
3. **Depreciation of Institutional Memory**: Unplanned departures disrupt ongoing R&D pipelines, sever long-standing client relationships, and leak proprietary tacit domain knowledge to competing firms.
4. **Contagion and Team Morale Erosion**: High attrition rates within specific operational clusters impose workload surges on remaining staff, precipitating acute occupational burnout and secondary attrition cascades.

### 3.2 Technical Objectives of the Investigation
1. **Structural Integrity & Hygiene**: Conduct systematic null-value diagnostics, duplicate reconciliations, and filter invariant/zero-variance attributes.
2. **Statistical Distribution Diagnostics**: Characterize continuous feature moments (mean, median, variance, skewness, kurtosis) and detect extreme observations via Tukey's Interquartile Range (IQR) fences.
3. **Bivariate & Non-Parametric Hypothesis Testing**: Quantify statistical significance across attrition classes using non-parametric two-tailed Mann-Whitney U tests for continuous variables and Pearson Chi-Square (Chi2) tests of independence for categorical structures.
4. **Exploratory Visual Analytics**: Construct publication-grade multi-panel figures adhering strictly to Ben Fry's 7 Stages of Data Visualization.
5. **Deep Learning Architecture Formulation**: Engineer a deep feed-forward Multi-Layer Perceptron (PyTorch MLP) incorporating Batch Normalization, Rectified Linear Unit (ReLU) activations, Dropout, and L2 weight decay regularization.
6. **Empirical Architectural Tuning**: Execute a systematic hyperparameter grid experiment across three candidate configurations to determine optimal capacity, learning rate, and regularizer penalty.
7. **Model-Agnostic Post-Hoc Interpretability**: Implement Permutation Feature Importance across 5 repeats on the holdout test set to isolate primary empirical drivers.
8. **Comparative Out-of-Sample Benchmarking**: Contrast the deep architecture against a regularized linear Logistic Regression baseline across Accuracy, Precision, Recall, Specificity, F1-Score, and ROC-AUC.

---

## 4. DATASET SPECIFICATION & SCHEMA ARCHITECTURE

The empirical data utilized is the real-world IBM HR Analytics Employee Attrition Dataset (N = 1,470 instances, 35 attributes).

### 4.1 Schema Data Dictionary Taxonomy (Sample of Core Fields)
| Field Name | Data Type | Measurement Scale | Domain Description & Technical Role | Valid Range / Unique Values |
| :--- | :---: | :---: | :--- | :--- |
| `Age` | Integer | Ratio | Chronological age of employee in completed solar years | [18, 60] (43 unique) |
| `Attrition` | String | Nominal | **Target Variable**: Voluntary resignation status | {'Yes', 'No'} |
| `BusinessTravel` | String | Ordinal | Frequency of mandatory commercial travel | Non-Travel, Travel_Rarely, Travel_Frequently |
| `DailyRate` | Integer | Ratio | Internal billing or compensation daily pay rate ($) | [102, 1499] (886 unique) |
| `Department` | String | Nominal | Organizational division / operational unit | Human Resources, R&D, Sales |
| `DistanceFromHome` | Integer | Ratio | Commuting distance from residential domicile (Miles) | [1, 29] (29 unique) |
| `JobRole` | String | Nominal | Specific functional professional appointment | 9 distinct roles (Sales, Lab Tech, etc.) |
| `MonthlyIncome` | Integer | Ratio | Gross monthly financial compensation ($/month) | [$1,009, $19,999] (1,349 unique) |
| `OverTime` | String | Binary | Requirement to execute mandatory overtime | {'No', 'Yes'} |
| `TotalWorkingYears` | Integer | Ratio | Cumulative career professional work experience (Years) | [0, 40] (40 unique) |
| `YearsAtCompany` | Integer | Ratio | Total cumulative enterprise tenure at current firm | [0, 40] (37 unique) |

### 4.2 Continuous Numerical Summary Statistics Table (N = 1,470)
| Feature Name | Count | Mean (μ) | Std (σ) | Min | Q1 (25%) | Median (Q2) | Q3 (75%) | Max | Skewness (γ1) | Kurtosis (γ2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Age** | 1,470 | 36.92 | 9.14 | 18.0 | 30.0 | 36.0 | 43.0 | 60.0 | +0.413 | -0.407 |
| **DailyRate** | 1,470 | 802.49 | 403.51 | 102.0 | 465.0 | 802.0 | 1157.0 | 1499.0 | -0.003 | -1.206 |
| **DistanceFromHome** | 1,470 | 9.19 | 8.11 | 1.0 | 2.0 | 7.0 | 14.0 | 29.0 | +0.958 | -0.225 |
| **HourlyRate** | 1,470 | 65.89 | 20.33 | 30.0 | 48.0 | 66.0 | 83.8 | 100.0 | -0.032 | -1.196 |
| **MonthlyIncome** | 1,470 | 6,502.93 | 4,707.96 | 1,009.0 | 2,911.0 | 4,919.0 | 8,379.0 | 19,999.0 | **+1.368** | **+0.998** |
| **TotalWorkingYears** | 1,470 | 11.28 | 7.78 | 0.0 | 6.0 | 10.0 | 15.0 | 40.0 | **+1.116** | **+0.911** |
| **YearsAtCompany** | 1,470 | 7.01 | 6.13 | 0.0 | 3.0 | 5.0 | 9.0 | 40.0 | **+1.765** | **+3.924** |

---

## 5. DATA PREPROCESSING & OUTLIER FENCES

### 5.1 Hygiene & Column Pruning
Zero missing values and zero duplicate rows were detected. Four non-informative zero-variance fields were removed: `EmployeeCount` (constant 1), `Over18` (constant 'Y'), `StandardHours` (constant 80), and `EmployeeNumber` (unique ID).

### 5.2 Tukey's IQR Fences Table
| Feature Evaluated | Q1 (25%) | Median (Q2) | Q3 (75%) | IQR | Lower Fence (FL) | Upper Fence (FU) | Outlier Count | Outlier % | Domain Justification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MonthlyIncome** | $2,911.0 | $4,919.0 | $8,379.0 | $5,468.0 | -$5,291.0 | $16,581.0 | 114 | 7.76% | Executive salaries (VP/Director); valid signals. |
| **TotalWorkingYears** | 6.0 yrs | 10.0 yrs | 15.0 yrs | 9.0 yrs | -7.5 yrs | 28.5 yrs | 63 | 4.29% | Senior senior-career personnel (>28.5 years). |
| **YearsAtCompany** | 3.0 yrs | 5.0 yrs | 9.0 yrs | 6.0 yrs | -6.0 yrs | 18.0 yrs | 104 | 7.07% | High enterprise longevity stalwarts. |
| **Age** | 30.0 yrs | 36.0 yrs | 43.0 yrs | 13.0 yrs | 10.5 yrs | 62.5 yrs | 0 | 0.00% | No outliers; bounded [18, 60]. |
| **DistanceFromHome** | 2.0 mi | 7.0 mi | 14.0 mi | 12.0 mi | -16.0 mi | 32.0 mi | 0 | 0.00% | No outliers; maximum commute is 29 miles. |

### 5.3 Stratified Data Partitioning (random_state = 42)
| Partition Subset | Split Ratio | Total Instances (N) | Retained Staff (y=0) | Attrited Staff (y=1) | Attrition Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Partition** | 70.0% | 1,029 | 863 | 166 | 16.13% |
| **Validation Partition** | 15.0% | 220 | 185 | 35 | 15.91% |
| **Holdout Test Partition**| 15.0% | 221 | 185 | 36 | 16.29% |
| **Full Combined Dataset** | 100.0% | 1,470 | 1,233 | 237 | 16.12% |

---

## 6. DATA VISUALIZATION (BEN FRY'S 7 STAGES + 6 FIGURES)

### Ben Fry's 7 Stages Implementation:
1. **Acquire**: Automated HTTPS download of raw IBM HR attrition dataset.
2. **Parse**: Ingestion into Pandas structured DataFrame, casting types.
3. **Filter**: Removed 4 zero-variance attributes; isolated 31 predictive attributes.
4. **Mine**: Evaluated Tukey IQR bounds, moments (skewness, kurtosis), Mann-Whitney U, and Chi-Square p-values.
5. **Represent**: Donut decomposition, KDE distributions, bivariate boxplots, stacked/grouped bars, triangular heatmaps, pairplots.
6. **Refine**: Rendered at 300 DPI, academic typography, statistical callout annotations, grid styling.
7. **Interact**: Post-hoc permutation feature importance ranking across 5 repeats, threshold calibration.

### Figure Descriptions (Mandatory 3-Part Explanations):
* **Figure 1: Target Class Distribution & Imbalance Ratio (5.2 : 1)**
  - *What it shows*: Headcount frequency bar chart (1,233 No vs. 237 Yes) and donut breakdown showing 83.9% retained vs. 16.1% departed.
  - *Why it was created*: Demonstrates severe target skewness to justify why accuracy alone is insufficient, necessitating sensitivity and ROC-AUC metrics.
  - *Insights obtained*: Baseline turnover is 16.12%. A naive majority-class classifier achieves 83.88% accuracy while failing 100% of flight-risk personnel.
* **Figure 2: Continuous Probability Density Estimation (Moments)**
  - *What it shows*: KDE distributions for MonthlyIncome, Age, and TotalWorkingYears with Mean, Median, Skewness, and Kurtosis markers.
  - *Why it was created*: Evaluates departure from normality to guide non-parametric hypothesis testing and feature scaling.
  - *Insights obtained*: MonthlyIncome exhibits marked positive skewness (γ1 = +1.37) and leptokurtic peakedness (γ2 = +1.00). Age is near-normal centered at 36.9 years.
* **Figure 3: Non-Parametric Bivariate Distribution Comparison (Mann-Whitney U)**
  - *What it shows*: Side-by-side Tukey boxplots for MonthlyIncome, TotalWorkingYears, and YearsAtCompany with two-tailed Mann-Whitney U test p-values.
  - *Why it was created*: Non-parametrically tests median differences across attrition classes without Gaussian assumptions.
  - *Insights obtained*: Resigning employees earn a median monthly income of $3,202 vs. $5,204 for retained staff (U = 100,620.5, p = 2.95e-14). Early career staff face significantly higher turnover risk.
* **Figure 4: Categorical Contingency & Chi-Square Significance**
  - *What it shows*: 100% stacked bar chart of OverTime vs Attrition alongside horizontal turnover rate across 9 Job Roles with Chi-Square statistics.
  - *Why it was created*: Uncovers role vulnerabilities and validates categorical dependencies through formal contingency testing.
  - *Insights obtained*: Mandatory OverTime is the single strongest categorical predictor (Chi2 = 87.56, p = 8.16e-21), with 30.5% attrition for overtime staff vs. 10.4% for non-overtime. Sales Reps experience 39.8% turnover.
* **Figure 5: Triangular Lower-Bound Pearson Correlation Heatmap**
  - *What it shows*: Lower-triangular Pearson correlation matrix across 18 continuous/ordinal features and the Attrition target.
  - *Why it was created*: Detects multicollinearity clusters and linear bivariate associations with employee turnover.
  - *Insights obtained*: Strong collinearity exists between JobLevel and MonthlyIncome (r = +0.95), and TotalWorkingYears and JobLevel (r = +0.78). Direct correlations with attrition are negative with career tenure.
* **Figure 6: Multivariate High-Dimensional Feature Space Pairwise Interactions**
  - *What it shows*: Pairwise scatter interaction projections and diagonal KDE contours across Age, MonthlyIncome, TotalWorkingYears, and YearsAtCompany segmented by Attrition.
  - *Why it was created*: Visualizes non-linear decision boundaries and high-dimensional clustering dynamics.
  - *Insights obtained*: Dense attrition clustering is localized in early-career stages (Age < 32, MonthlyIncome < $4,000), illustrating that turnover is non-linearly concentrated in junior staff.

---

## 7. DEEP LEARNING MODEL & HYPERPARAMETER TUNING

### 7.1 Mathematical Architecture:
Feed-Forward Multi-Layer Perceptron (PyTorch MLP) with Batch Normalization, ReLU activations, and Dropout (p = 0.20):
- Input Layer: D = 44 standardized features
- Hidden Layer 1: Dense(44 -> 64) + BatchNorm + ReLU + Dropout(0.20)
- Hidden Layer 2: Dense(64 -> 32) + BatchNorm + ReLU + Dropout(0.20)
- Hidden Layer 3: Dense(32 -> 16) + BatchNorm + ReLU
- Output Layer: Dense(16 -> 1) + Sigmoid Activation
- Loss: Binary Cross-Entropy (BCE) with L2 weight decay regularizer (alpha = 1e-4)

### 7.2 Hyperparameter Tuning Grid (Validation Set N = 220):
| Configuration ID | Hidden Architecture | Learning Rate | L2 Penalty | Best Epoch | Val Loss | Val ROC-AUC | Val Accuracy | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Config A (Optimal)** | **[64, 32, 16]** | **0.0010** | **0.0001** | **80** | **0.4854** | **0.7654** | **85.00%** | **Selected for production** |
| Config B (Deep/Wide) | [128, 64, 32] | 0.0005 | 0.0005 | 80 | 0.4777 | 0.7710 | 80.91% | Slight overparameterization |
| Config C (Compact) | [32, 16, 8] | 0.0020 | 0.0010 | 80 | 0.4548 | 0.7504 | 89.09% | Majority-class bias |

### 7.3 Top 10 Permutation Feature Importance Drivers (Holdout Test Set, 5 Repeats):
| Rank | Feature | Mean Drop in ROC-AUC | Std Dev | Domain Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **OverTime** | 0.0654 | ± 0.0083 | Work-life exhaustion directly drives voluntary resignation. |
| 2 | **JobInvolvement** | 0.0198 | ± 0.0090 | Cognitive disengagement precedes formal resignation notice. |
| 3 | **YearsSinceLastPromotion** | 0.0187 | ± 0.0096 | Upward mobility stagnation and career ceiling barriers. |
| 4 | **JobSatisfaction** | 0.0173 | ± 0.0096 | Affective task dissatisfaction and lower workplace morale. |
| 5 | **YearsInCurrentRole** | 0.0162 | ± 0.0042 | Operational role monotony and boredom. |
| 6 | **StockOptionLevel** | 0.0134 | ± 0.0118 | Equity vesting serves as retention 'golden handcuffs'. |
| 7 | **DistanceFromHome** | 0.0133 | ± 0.0058 | Commuting friction compounds weekly fatigue. |
| 8 | **Department_Sales** | 0.0127 | ± 0.0025 | High commercial sales quota pressures. |
| 9 | **MonthlyIncome** | 0.0117 | ± 0.0043 | External market salary competitiveness. |
| 10 | **HourlyRate** | 0.0116 | ± 0.0048 | Internal labor accounting compensation rate. |

---

## 8. RESULT VISUALIZATION & BENCHMARK INTERPRETATION

### Figures 7 to 10 Evaluation:
* **Figure 7: Training vs. Validation Loss Trajectory**: Loss steadily declines across 80 epochs without divergence, confirming the efficacy of Dropout and L2 regularization. Optimal checkpoint at Epoch 80 (Val Loss = 0.4854).
* **Figure 8: Training vs. Validation Accuracy Trajectory**: Accuracy rapidly climbs in epochs 1–20 and stabilizes smoothly between 84% and 86% beyond epoch 40.
* **Figure 9: Holdout Test Confusion Matrix**: TN = 162 (87.6%), FP = 23 (12.4%), FN = 12 (33.3%), and TP = 24 (66.7%).
* **Figure 10: ROC-AUC Benchmark Curve**: Deep MLP achieves AUC = 0.8284 vs. Logistic Regression AUC = 0.8407.

### Out-of-Sample Test Benchmark Comparison (N = 221):
| Evaluation Metric | Deep Learning MLP (PyTorch) | Regularized Logistic Regression | Delta (Δ) | Strategic HR Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | 84.16% | 87.33% | -3.17% | Both models deliver solid global accuracy. |
| **Precision** | 51.06% | 66.67% | -15.61% | Baseline flags fewer false alarms. |
| **Recall (Sensitivity)** | **66.67% (24 / 36)** | **44.44% (16 / 36)** | **+22.23%** | **MLP catches +8 additional true flight risks.** |
| **Specificity** | 87.57% (162 / 185) | 95.68% (177 / 185) | -8.11% | Preserves strong non-attriter operational baseline. |
| **F1-Score** | **0.5783** | **0.5333** | **+0.0450** | **Superior harmonic precision-recall balance.** |
| **ROC-AUC** | 0.8284 | 0.8407 | -0.0123 | Strong overall ranking discriminative power. |
| **False Negatives (Misses)**| **12** | **20** | **-8 (-40.0%)** | **40% reduction in catastrophic undetected turnover.** |

### Cost Asymmetry Domain Analysis:
The cost of an undetected departure (50% to 200% of salary, $30k–$120k) vastly outweighs a false alarm check-in (<$200). Therefore, **Recall is the decisive operational metric**. The Deep MLP's **66.67% recall** (+22.23% over baseline) reduces missed resignations by **40%**, making it the superior enterprise retention model.

---

## 9. STRATEGIC HR RECOMMENDATIONS & FUTURE SCOPE
1. **Mandatory Overtime Capping**: Overtime is the primary empirical trigger (Chi2 = 87.56, p = 8.16e-21). Enforce quarterly overtime caps and mandatory rest audits.
2. **Early-Tenure Equity Adjustments**: Attriters earn $3,202 vs. $5,204. Institute 18-month compensation reviews for entry-level cohorts.
3. **Transparent Career Ladders**: Years since last promotion is the 3rd driving factor. Establish skill-based milestone promotions.
4. **Future Directions**: (1) Time-to-event Survival Analysis (Cox Proportional Hazards and DeepSurv); (2) Local SHAP explainability; (3) Synthetic minority oversampling (SMOTE-NC).

---

## 10. REFERENCES (IEEE STYLE)
[1] W. F. Cascio and J. W. Boudreau, *Investing in People: Financial Impact of Human Resource Initiatives*, 2nd ed. Upper Saddle River, NJ, USA: Pearson Education, 2016.  
[2] B. Fry, *Computational Information Design*, Ph.D. dissertation, Media Laboratory, Massachusetts Institute of Technology, Cambridge, MA, USA, 2004.  
[3] A. Fisher, C. Rudin, and F. Dominici, "All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously," *Journal of Machine Learning Research*, vol. 20, no. 177, pp. 1–81, 2019.  
[4] V. Borisov, T. Leemann, K. Seßler, J. Haug, M. Pawelczyk, and G. Kasneci, "Deep neural networks and tabular data: A survey," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 35, no. 6, pp. 7490–7510, Jun. 2024.  
[5] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, Jun. 2002.

---

## 11. APPENDIX: CORE REPRODUCIBLE PIPELINE SOURCE CODE
The complete pipeline executes deterministically via `python attrition_pipeline.py`.  
GitHub Repository: https://github.com/raveenasivaprakasam021/IBM-HR-Attrition-EDA-DL
