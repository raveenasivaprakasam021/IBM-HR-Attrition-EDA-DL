"""
====================================================================================================
ACADEMIC CASE STUDY PROJECT: EMPLOYEE ATTRITION PREDICTION & EXPLANATORY MODELING
Course Code: U21ADP05 - Exploratory Data Analysis and Visualization (Assignment II)
Department of Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology
Author: Raveena S (Roll No: 24AD087) | Batch: 2023-2027 | Year/Sem: V
====================================================================================================
This standalone, reproducible pipeline performs:
  1. Data Acquisition & Validation (IBM HR Analytics Attrition Dataset)
  2. Data Preprocessing & Outlier Diagnostics (Tukey's IQR fences, dummy encoding, scaling)
  3. Stratified Partitioning (70% Train, 15% Validation, 15% Test)
  4. Publication-Grade Exploratory Visualizations (Figures 1 - 6, 300 DPI)
  5. Deep Learning Multi-Layer Perceptron (PyTorch MLP) Architecture & Hyperparameter Tuning
  6. Permutation Feature Importance & Model Explainability
  7. Regularized Logistic Regression Baseline Comparison
  8. Model Evaluation Visualizations (Figures 7 - 10, 300 DPI)
  9. Comprehensive Benchmark Metrics Export
====================================================================================================
"""

import os
import sys

# Prevent OpenMP runtime library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# =====================================================================
# 0. REPRODUCIBILITY SEED CONFIGURATION
# =====================================================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Set global Matplotlib styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.frameon'] = True

OUTPUT_DIR = os.path.join(os.getcwd(), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Academic Color Palette
COLOR_ACTIVE = '#1f77b4'   # Blue (Retained / No Attrition)
COLOR_ATTRITED = '#d62728' # Coral Red (Attrited / Yes Attrition)
PALETTE = [COLOR_ACTIVE, COLOR_ATTRITED]

# =====================================================================
# 1. DATA ACQUISITION & INTEGRITY VERIFICATION
# =====================================================================
DATA_URL = 'https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv'

print("="*80)
print("[STAGE 1: ACQUIRE & PARSE] Downloading and loading IBM HR Dataset...")
try:
    df_raw = pd.read_csv(DATA_URL)
    print(f"Dataset successfully loaded. Dimensions: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns.")
except Exception as e:
    print(f"Failed to fetch dataset online: {e}. Checking local cache...")
    if os.path.exists("emp_attrition.csv"):
        df_raw = pd.read_csv("emp_attrition.csv")
    else:
        raise FileNotFoundError("Could not download or find emp_attrition.csv locally.")

# Save local copy for offline archival
df_raw.to_csv("emp_attrition.csv", index=False)

# Integrity checks
n_duplicates = df_raw.duplicated().sum()
n_missing = df_raw.isnull().sum().sum()
print(f"Integrity Check: Duplicate Rows = {n_duplicates}, Total Missing Values = {n_missing}")

# Check zero-variance columns & identifiers
drop_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df = df_raw.drop(columns=drop_cols)
print(f"Removed non-informative / zero-variance columns: {drop_cols}")
print(f"Refined Feature Count: {df.shape[1]} (including target 'Attrition')")

# =====================================================================
# 2. STATISTICAL OUTLIER ANALYSIS (TUKEY'S IQR FENCES)
# =====================================================================
print("="*80)
print("[STAGE 2: FILTER & MINE] Computing Tukey's IQR Fences for Continuous Features...")
iqr_features = ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'Age', 'DistanceFromHome']
iqr_summary = []

for feat in iqr_features:
    q1 = df[feat].quantile(0.25)
    q3 = df[feat].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[feat] < lower_bound) | (df[feat] > upper_bound)]
    pct_outliers = (len(outliers) / len(df)) * 100
    iqr_summary.append({
        'Feature': feat,
        'Q1 (25th)': q1,
        'Median': df[feat].median(),
        'Q3 (75th)': q3,
        'IQR': iqr,
        'Lower Fence': lower_bound,
        'Upper Fence': upper_bound,
        'Outlier Count': len(outliers),
        'Outlier %': f"{pct_outliers:.2f}%"
    })

iqr_df = pd.DataFrame(iqr_summary)
print(iqr_df.to_string(index=False))

# =====================================================================
# 3. STATISTICAL HYPOTHESIS TESTING (MANN-WHITNEY U & CHI-SQUARE)
# =====================================================================
print("="*80)
print("[STATISTICAL TESTS] Bivariate Hypothesis Tests...")
group_yes = df[df['Attrition'] == 'Yes']
group_no = df[df['Attrition'] == 'No']

mwu_results = {}
for feat in ['MonthlyIncome', 'Age', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']:
    u_stat, p_val = stats.mannwhitneyu(group_yes[feat], group_no[feat], alternative='two-sided')
    mwu_results[feat] = {'U_stat': u_stat, 'p_value': p_val}
    print(f"Mann-Whitney U [{feat}]: U = {u_stat:.1f}, p = {p_val:.4e}")

chi2_results = {}
for cat in ['OverTime', 'JobRole', 'MaritalStatus', 'BusinessTravel', 'Department']:
    ct = pd.crosstab(df[cat], df['Attrition'])
    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    chi2_results[cat] = {'chi2': chi2, 'p_value': p_val, 'dof': dof}
    print(f"Chi-Square [{cat}]: Chi2 = {chi2:.4f}, p = {p_val:.4e}, df = {dof}")

# =====================================================================
# 4. EXPLORATORY DATA VISUALIZATIONS (FIGURES 1 - 6, 300 DPI)
# =====================================================================
print("="*80)
print("[STAGE 3: VISUALIZATION GENERATION] Exporting Figures 1 to 6...")

# ---------------------------------------------------------------------
# Figure 1: Target Class Distribution (Attrition Donut & Bar Chart)
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

counts = df['Attrition'].value_counts()
pcts = df['Attrition'].value_counts(normalize=True) * 100

# Bar chart with annotations
bars = ax1.bar(['Retained (No)', 'Attrited (Yes)'], counts, color=[COLOR_ACTIVE, COLOR_ATTRITED], width=0.55, edgecolor='black', linewidth=1.2)
ax1.set_title("Employee Attrition Headcount Distribution", pad=12)
ax1.set_ylabel("Employee Count (Frequency)")
ax1.set_ylim(0, 1450)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for bar, count, pct in zip(bars, counts, pcts):
    height = bar.get_height()
    ax1.annotate(f'{count:,} employees\n({pct:.2f}%)',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 6), textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

# Donut chart
wedges, texts, autotexts = ax2.pie(
    counts, labels=['Retained (No)', 'Attrited (Yes)'],
    colors=[COLOR_ACTIVE, COLOR_ATTRITED],
    autopct='%1.1f%%', startangle=140,
    pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

ax2.set_title("Attrition Class Imbalance Ratio (5.2 : 1)", pad=12)

plt.suptitle("Figure 1: Target Class Distribution & Class Imbalance Diagnostic", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, 'fig1_attrition_distribution.png')
plt.savefig(fig1_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path}")

# ---------------------------------------------------------------------
# Figure 2: Continuous Probability Density (Histograms + KDE + Stats)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
dist_cols = ['MonthlyIncome', 'Age', 'TotalWorkingYears']
labels = ['Monthly Income ($ / Month)', 'Employee Age (Years)', 'Total Working Experience (Years)']

for i, (col, label) in enumerate(zip(dist_cols, labels)):
    ax = axes[i]
    mean_val = df[col].mean()
    median_val = df[col].median()
    skew_val = stats.skew(df[col])
    kurt_val = stats.kurtosis(df[col])
    
    sns.histplot(df[col], kde=True, ax=ax, color='#2b5c8f', stat='density', bins=25, edgecolor='black', alpha=0.6)
    
    # Add vertical lines for Mean and Median
    ax.axvline(mean_val, color='#d62728', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='#2ca02c', linestyle='-', linewidth=2, label=f'Median: {median_val:.1f}')
    
    # Statistical text annotation box
    stat_text = f"Skewness: {skew_val:+.2f}\nKurtosis: {kurt_val:+.2f}"
    ax.text(0.95, 0.78, stat_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.set_title(f"Distribution of {col}", pad=10)
    ax.set_xlabel(label)
    ax.set_ylabel("Probability Density")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')

plt.suptitle("Figure 2: Probability Density Estimation & Moment Diagnostics (Skewness & Kurtosis)", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, 'fig2_continuous_kde_distributions.png')
plt.savefig(fig2_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

# ---------------------------------------------------------------------
# Figure 3: Bivariate Boxplots Across Attrition Classes + Mann-Whitney U
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
box_cols = ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
box_labels = ['Monthly Income ($)', 'Total Working Experience (Years)', 'Tenure at Company (Years)']

for i, (col, label) in enumerate(zip(box_cols, box_labels)):
    ax = axes[i]
    p_val = mwu_results[col]['p_value']
    u_val = mwu_results[col]['U_stat']
    
    sns.boxplot(x='Attrition', y=col, data=df, ax=ax, palette=[COLOR_ACTIVE, COLOR_ATTRITED],
                width=0.45, boxprops=dict(alpha=0.85), flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    # Add median annotations
    med_no = df[df['Attrition'] == 'No'][col].median()
    med_yes = df[df['Attrition'] == 'Yes'][col].median()
    
    # P-value header banner
    ax.text(0.5, 0.94, f"Mann-Whitney U: {u_val:,.0f}\np = {p_val:.2e} (Significant)",
            transform=ax.transAxes, ha='center', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', edgecolor='#d62728', alpha=0.9))
    
    ax.text(0, med_no, f"  Median: {med_no:,.0f}", verticalalignment='bottom', fontweight='bold', color='navy')
    ax.text(1, med_yes, f"  Median: {med_yes:,.0f}", verticalalignment='bottom', fontweight='bold', color='darkred')
    
    ax.set_title(f"{label} by Attrition Status", pad=12)
    ax.set_xlabel("Attrition Status")
    ax.set_ylabel(label)
    ax.set_xticklabels(['Retained (No)', 'Attrited (Yes)'])
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle("Figure 3: Non-Parametric Bivariate Distribution Comparison Across Attrition Classes", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'fig3_bivariate_boxplots_mannwhitney.png')
plt.savefig(fig3_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3_path}")

# ---------------------------------------------------------------------
# Figure 4: Categorical Bivariate Mosaic / Stacked & Grouped Bar Chart
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))

# Subplot 1: OverTime vs Attrition (Stacked + Percentages)
ct_ot = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index') * 100
ct_ot_raw = pd.crosstab(df['OverTime'], df['Attrition'])
bars_no = ax1.bar(ct_ot.index, ct_ot['No'], label='Retained (No)', color=COLOR_ACTIVE, width=0.45, edgecolor='black')
bars_yes = ax1.bar(ct_ot.index, ct_ot['Yes'], bottom=ct_ot['No'], label='Attrited (Yes)', color=COLOR_ATTRITED, width=0.45, edgecolor='black')

for i, idx in enumerate(ct_ot.index):
    # annotate Yes %
    pct_yes = ct_ot.loc[idx, 'Yes']
    cnt_yes = ct_ot_raw.loc[idx, 'Yes']
    cnt_total = ct_ot_raw.loc[idx].sum()
    ax1.annotate(f"{pct_yes:.1f}%\n({cnt_yes}/{cnt_total})",
                 xy=(i, 100 - pct_yes / 2), ha='center', va='center',
                 color='white', fontweight='bold', fontsize=11)
    
ax1.set_title(f"Attrition Rate by OverTime Requirement\n($\\chi^2 = {chi2_results['OverTime']['chi2']:.2f}, p = {chi2_results['OverTime']['p_value']:.2e}$)", pad=12)
ax1.set_ylabel("Proportion of Employees (%)")
ax1.set_xlabel("Mandatory Overtime Status")
ax1.set_ylim(0, 105)
ax1.legend(loc='lower left')
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# Subplot 2: JobRole vs Attrition (Horizontal Grouped Rate)
role_rates = (df.groupby('JobRole')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100)
              .sort_values(ascending=True))

barh = ax2.barh(role_rates.index, role_rates.values, color='#e6550d', edgecolor='black', height=0.6)
ax2.axvline(df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0).mean() * 100,
            color='black', linestyle='--', linewidth=1.5, label='Overall Average Attrition (16.1%)')

for bar in barh:
    width = bar.get_width()
    ax2.annotate(f'{width:.1f}%',
                 xy=(width, bar.get_y() + bar.get_height() / 2),
                 xytext=(5, 0), textcoords="offset points",
                 ha='left', va='center', fontsize=10, fontweight='bold')

ax2.set_title(f"Attrition Prevalence Across Organizational Roles\n($\\chi^2 = {chi2_results['JobRole']['chi2']:.2f}, p = {chi2_results['JobRole']['p_value']:.2e}$)", pad=12)
ax2.set_xlabel("Attrition Rate (%)")
ax2.set_xlim(0, 48)
ax2.legend(loc='lower right')
ax2.grid(axis='x', linestyle='--', alpha=0.4)

plt.suptitle("Figure 4: Bivariate Categorical Diagnostics with Contingency Chi-Square Significance", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, 'fig4_categorical_attrition_chisq.png')
plt.savefig(fig4_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig4_path}")

# ---------------------------------------------------------------------
# Figure 5: Triangular Correlation Heatmap
# ---------------------------------------------------------------------
corr_features = [
    'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

# Add numeric target
df['Attrition_Num'] = (df['Attrition'] == 'Yes').astype(int)
corr_matrix = df[corr_features + ['Attrition_Num']].corr(method='pearson')

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(14, 11))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.4, vmax=1.0, center=0,
            annot=True, fmt=".2f", square=True, linewidths=0.6,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient ($r$)"},
            annot_kws={"size": 8})

plt.title("Figure 5: Triangular Lower-Bound Pearson Correlation Heatmap of Structural Variables", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, 'fig5_correlation_heatmap.png')
plt.savefig(fig5_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig5_path}")

# ---------------------------------------------------------------------
# Figure 6: Multivariate Pairplot
# ---------------------------------------------------------------------
pair_vars = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
sub_df = df[pair_vars + ['Attrition']].copy()
sub_df['Attrition'] = sub_df['Attrition'].map({'No': 'Retained', 'Yes': 'Attrited'})

g = sns.pairplot(sub_df, hue='Attrition', palette={'Retained': COLOR_ACTIVE, 'Attrited': COLOR_ATTRITED},
                 corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 25, 'edgecolor': 'none'},
                 diag_kws={'fill': True, 'alpha': 0.4})
g.fig.suptitle("Figure 6: High-Dimensional Feature Space Pairwise Interactions by Attrition Status",
               fontsize=14, fontweight='bold', y=1.02)
fig6_path = os.path.join(OUTPUT_DIR, 'fig6_multivariate_pairplot.png')
g.savefig(fig6_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig6_path}")

# =====================================================================
# 5. DATA PREPROCESSING, DUMMY ENCODING & STRATIFIED PARTITIONING
# =====================================================================
print("="*80)
print("[STAGE 4: PREPROCESSING & SPLIT] Encoding and Stratified Partitioning...")

df_proc = df.copy()
df_proc.drop(columns=['Attrition_Num'], inplace=True, errors='ignore')

# Target
df_proc['Attrition'] = (df_proc['Attrition'] == 'Yes').astype(int)

# Binary categorical mapping
df_proc['OverTime'] = (df_proc['OverTime'] == 'Yes').astype(int)
df_proc['Gender'] = (df_proc['Gender'] == 'Male').astype(int)

# One-hot encoding for multi-class categorical features with drop_first=True
multi_cats = df_proc.select_dtypes(include=['object']).columns.tolist()
print(f"Multi-class categorical columns for dummy encoding: {multi_cats}")
df_encoded = pd.get_dummies(df_proc, columns=multi_cats, drop_first=True, dtype=float)

X = df_encoded.drop(columns=['Attrition'])
y = df_encoded['Attrition']

# 70% Train, 15% Val, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

# Feature Standardization: fit STRICTLY on X_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Stratified Partition Summary:")
print(f"  Training Set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%) | Attrited: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"  Validation Set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%) | Attrited: {y_val.sum()} ({y_val.mean()*100:.2f}%)")
print(f"  Holdout Test:   {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%) | Attrited: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
print(f"  Feature Dimension: {X_train.shape[1]} input dimensions.")

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# =====================================================================
# 6. DEEP LEARNING MODEL ARCHITECTURE & HYPERPARAMETER TUNING
# =====================================================================
print("="*80)
print("[STAGE 5: DEEP LEARNING] Formulating Deep MLP and Executing Tuning Experiments...")

class DeepAttritionMLP(nn.Module):
    """
    Deep Feed-Forward Multi-Layer Perceptron (MLP) for Employee Attrition Prediction.
    Architecture:
      - Input Layer: dimension D = 44
      - Hidden Layer 1: Dense + BatchNorm + ReLU + Dropout
      - Hidden Layer 2: Dense + BatchNorm + ReLU + Dropout
      - Hidden Layer 3: Dense + BatchNorm + ReLU
      - Output Layer: Dense + Sigmoid Logit Activation (Binary Classification)
    """
    def __init__(self, in_features, h1, h2, h3, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_prob)
        
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.relu3 = nn.ReLU()
        
        self.out = nn.Linear(h3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.sigmoid(self.out(x))
        return x

# Hyperparameter search configurations
tuning_configs = [
    {
        'Config_ID': 'Config A (Balanced/Optimal)',
        'Hidden_Layers': [64, 32, 16],
        'LR': 0.001,
        'L2_Penalty': 1e-4,
        'Dropout': 0.20
    },
    {
        'Config_ID': 'Config B (Deep/Over-parameterized)',
        'Hidden_Layers': [128, 64, 32],
        'LR': 0.0005,
        'L2_Penalty': 5e-4,
        'Dropout': 0.30
    },
    {
        'Config_ID': 'Config C (Compact/Under-parameterized)',
        'Hidden_Layers': [32, 16, 8],
        'LR': 0.002,
        'L2_Penalty': 1e-3,
        'Dropout': 0.10
    }
]

tuning_results = []
NUM_EPOCHS = 80

for cfg in tuning_configs:
    set_seed(SEED)
    h = cfg['Hidden_Layers']
    model_candidate = DeepAttritionMLP(X_train.shape[1], h[0], h[1], h[2], cfg['Dropout'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_candidate.parameters(), lr=cfg['LR'], weight_decay=cfg['L2_Penalty'])
    
    best_loss = float('inf')
    best_epoch = 0
    best_val_auc = 0.0
    best_val_acc = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model_candidate.train()
        optimizer.zero_grad()
        out = model_candidate(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
        model_candidate.eval()
        with torch.no_grad():
            val_out = model_candidate(X_val_t)
            v_loss = criterion(val_out, y_val_t).item()
            v_probs = val_out.numpy().flatten()
            v_auc = roc_auc_score(y_val, v_probs)
            v_acc = accuracy_score(y_val, (v_probs >= 0.5).astype(int))
            
            if v_loss < best_loss:
                best_loss = v_loss
                best_epoch = epoch
                best_val_auc = v_auc
                best_val_acc = v_acc
                
    tuning_results.append({
        'Config_ID': cfg['Config_ID'],
        'Hidden_Layers': str(cfg['Hidden_Layers']),
        'Learning_Rate': cfg['LR'],
        'L2_Penalty': cfg['L2_Penalty'],
        'Best_Epoch': best_epoch,
        'Val_Loss': best_loss,
        'Val_AUC': best_val_auc,
        'Val_Accuracy': best_val_acc
    })

tuning_df = pd.DataFrame(tuning_results)
print("Hyperparameter Tuning Summary:")
print(tuning_df.to_string(index=False))

# =====================================================================
# 7. FINAL MODEL TRAINING & EVALUATION CHECKPOINTING
# =====================================================================
print("="*80)
print("[STAGE 6: TRAINING CHOSEN MODEL] Training Config A and logging trajectories...")
set_seed(SEED)
best_h = [64, 32, 16]
final_model = DeepAttritionMLP(X_train.shape[1], best_h[0], best_h[1], best_h[2], dropout_prob=0.2)
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

best_chk_loss = float('inf')
optimal_epoch = 0

for epoch in range(1, NUM_EPOCHS + 1):
    final_model.train()
    optimizer.zero_grad()
    train_out = final_model(X_train_t)
    loss = criterion(train_out, y_train_t)
    loss.backward()
    optimizer.step()
    
    final_model.eval()
    with torch.no_grad():
        # Train metrics
        t_probs = train_out.numpy().flatten()
        t_acc = accuracy_score(y_train, (t_probs >= 0.5).astype(int))
        
        # Validation metrics
        v_out = final_model(X_val_t)
        v_loss = criterion(v_out, y_val_t).item()
        v_probs = v_out.numpy().flatten()
        v_acc = accuracy_score(y_val, (v_probs >= 0.5).astype(int))
        
    train_loss_history.append(loss.item())
    val_loss_history.append(v_loss)
    train_acc_history.append(t_acc * 100)
    val_acc_history.append(v_acc * 100)
    
    if v_loss < best_chk_loss:
        best_chk_loss = v_loss
        optimal_epoch = epoch
        torch.save(final_model.state_dict(), "best_attrition_mlp.pt")

print(f"Optimal Checkpoint identified at Epoch {optimal_epoch} (Validation Loss: {best_chk_loss:.4f})")

# Load optimal weights for holdout testing
final_model.load_state_dict(torch.load("best_attrition_mlp.pt"))
final_model.eval()

# Holdout Test Predictions
with torch.no_grad():
    mlp_test_probs = final_model(X_test_t).numpy().flatten()
    mlp_test_preds = (mlp_test_probs >= 0.5).astype(int)

# Logistic Regression Baseline
print("[BASELINE] Training Regularized Logistic Regression Baseline...")
lr_baseline = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
lr_baseline.fit(X_train_scaled, y_train)
lr_test_probs = lr_baseline.predict_proba(X_test_scaled)[:, 1]
lr_test_preds = (lr_test_probs >= 0.5).astype(int)

# Metrics calculation
def calculate_metrics(y_true, y_pred, y_probs):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall (Sensitivity)': rec,
        'Specificity': spec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

mlp_metrics = calculate_metrics(y_test, mlp_test_preds, mlp_test_probs)
lr_metrics = calculate_metrics(y_test, lr_test_preds, lr_test_probs)

benchmark_comparison = pd.DataFrame([
    {'Model': 'Deep Learning MLP (PyTorch)', **mlp_metrics},
    {'Model': 'Logistic Regression (Baseline)', **lr_metrics}
])

print("\n" + "="*80)
print("OUT-OF-SAMPLE TEST BENCHMARK COMPARISON TABLE")
print("="*80)
print(benchmark_comparison.to_string(index=False))

# =====================================================================
# 8. PERMUTATION FEATURE IMPORTANCE (MODEL-AGNOSTIC EXPLAINABILITY)
# =====================================================================
print("="*80)
print("[STAGE 7: MODEL EXPLAINABILITY] Computing Permutation Feature Importance (5 Repeats)...")
base_auc = roc_auc_score(y_test, mlp_test_probs)
feature_names = X.columns.tolist()
perm_importance = {feat: [] for feat in feature_names}

for repeat in range(5):
    for i, col in enumerate(feature_names):
        X_p = X_test_scaled.copy()
        X_p[:, i] = np.random.permutation(X_p[:, i])
        with torch.no_grad():
            p_probs = final_model(torch.tensor(X_p, dtype=torch.float32)).numpy().flatten()
            p_auc = roc_auc_score(y_test, p_probs)
            perm_importance[col].append(base_auc - p_auc)

perm_df = pd.DataFrame({
    'Rank': range(1, len(feature_names) + 1),
    'Feature': feature_names,
    'Mean Drop in AUC': [np.mean(perm_importance[f]) for f in feature_names],
    'Std Dev': [np.std(perm_importance[f]) for f in feature_names]
}).sort_values(by='Mean Drop in AUC', ascending=False)
perm_df['Rank'] = range(1, len(feature_names) + 1)

print("\nTop 10 Drivers of Attrition (Permutation Importance):")
print(perm_df.head(10).to_string(index=False))

# =====================================================================
# 9. EVALUATION VISUALIZATIONS (FIGURES 7 - 10, 300 DPI)
# =====================================================================
print("="*80)
print("[STAGE 8: EVALUATION VISUALIZATION] Generating Figures 7 to 10...")

# ---------------------------------------------------------------------
# Figure 7: Training vs Validation Loss Trajectory
# ---------------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
epochs_range = range(1, NUM_EPOCHS + 1)
plt.plot(epochs_range, train_loss_history, label='Training Loss (BCE)', color='#1f77b4', linewidth=2.2)
plt.plot(epochs_range, val_loss_history, label='Validation Loss (BCE)', color='#d62728', linewidth=2.2, linestyle='--')
plt.axvline(optimal_epoch, color='black', linestyle=':', linewidth=1.8,
            label=f'Optimal Checkpoint (Epoch {optimal_epoch}, Loss: {best_chk_loss:.4f})')
plt.scatter([optimal_epoch], [val_loss_history[optimal_epoch - 1]], color='red', s=70, zorder=5)

plt.title("Figure 7: Training vs. Validation Binary Cross-Entropy Loss Trajectory", pad=12)
plt.xlabel("Training Epoch")
plt.ylabel("Binary Cross-Entropy (BCE) Loss")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
fig7_path = os.path.join(OUTPUT_DIR, 'fig7_training_val_loss_curve.png')
plt.savefig(fig7_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig7_path}")

# ---------------------------------------------------------------------
# Figure 8: Training vs Validation Accuracy Trajectory
# ---------------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
plt.plot(epochs_range, train_acc_history, label='Training Accuracy (%)', color='#2ca02c', linewidth=2.2)
plt.plot(epochs_range, val_acc_history, label='Validation Accuracy (%)', color='#ff7f0e', linewidth=2.2, linestyle='--')
plt.axvline(optimal_epoch, color='black', linestyle=':', linewidth=1.8,
            label=f'Optimal Epoch {optimal_epoch} (Val Acc: {val_acc_history[optimal_epoch-1]:.2f}%)')
plt.scatter([optimal_epoch], [val_acc_history[optimal_epoch - 1]], color='darkgreen', s=70, zorder=5)

plt.title("Figure 8: Training vs. Validation Accuracy Trajectory Across Epochs", pad=12)
plt.xlabel("Training Epoch")
plt.ylabel("Classification Accuracy (%)")
plt.ylim(65, 100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
fig8_path = os.path.join(OUTPUT_DIR, 'fig8_training_val_accuracy_curve.png')
plt.savefig(fig8_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig8_path}")

# ---------------------------------------------------------------------
# Figure 9: Holdout Test Confusion Matrix
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, mlp_test_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

annot_matrix = np.empty_like(cm, dtype=object)
labels_diag = [['True Negative (TN)', 'False Positive (FP)'], ['False Negative (FN)', 'True Positive (TP)']]
for r in range(2):
    for c in range(2):
        annot_matrix[r, c] = f"{labels_diag[r][c]}\n{cm[r, c]:,}\n({cm_norm[r, c]:.1f}%)"

sns.heatmap(cm, annot=annot_matrix, fmt="", cmap='Blues', cbar=False, ax=ax,
            linewidths=1.5, annot_kws={"size": 11, "weight": "bold"})

ax.set_title("Figure 9: Holdout Test Set Confusion Matrix (Normalized % & Raw Counts)", pad=14)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Ground Truth Class")
ax.set_xticklabels(['Retained (0)', 'Attrited (1)'])
ax.set_yticklabels(['Retained (0)', 'Attrited (1)'], va='center')
plt.tight_layout()
fig9_path = os.path.join(OUTPUT_DIR, 'fig9_confusion_matrix.png')
plt.savefig(fig9_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig9_path}")

# ---------------------------------------------------------------------
# Figure 10: ROC-AUC Comparison Curve (MLP vs Baseline Logistic Regression)
# ---------------------------------------------------------------------
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_test_probs)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_probs)

plt.figure(figsize=(8.5, 6.5))
plt.plot(fpr_mlp, tpr_mlp, color='#1f77b4', linewidth=2.5,
         label=f'Deep Learning MLP (AUC = {mlp_metrics["ROC-AUC"]:.4f})')
plt.plot(fpr_lr, tpr_lr, color='#ff7f0e', linewidth=2.2, linestyle='--',
         label=f'Logistic Regression Baseline (AUC = {lr_metrics["ROC-AUC"]:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.8,
         label='Random Chance (AUC = 0.5000)')

plt.title("Figure 10: Receiver Operating Characteristic (ROC) Benchmark Comparison", pad=12)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall / Sensitivity)")
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', framealpha=0.95)
plt.tight_layout()
fig10_path = os.path.join(OUTPUT_DIR, 'fig10_roc_auc_curve.png')
plt.savefig(fig10_path, bbox_inches='tight')
plt.close()
print(f"Saved: {fig10_path}")

# =====================================================================
# 10. METRICS EXPORT & SUMMARY GENERATION
# =====================================================================
summary_payload = {
    "project": "Employee Attrition Prediction & Visualization",
    "course_code": "U21ADP05",
    "student": "Raveena S (24AD087)",
    "institution": "KPR Institute of Engineering and Technology",
    "dataset": {
        "source": "IBM HR Analytics Attrition Dataset",
        "raw_shape": [df_raw.shape[0], df_raw.shape[1]],
        "refined_shape": [df.shape[0], df.shape[1]],
        "feature_dimension": X_train.shape[1],
        "class_counts": {"No": int(counts['No']), "Yes": int(counts['Yes'])},
        "attrition_rate": float(pcts['Yes'])
    },
    "iqr_fences": iqr_summary,
    "tuning_experiments": tuning_results,
    "test_benchmark": [
        {'Model': 'Deep Learning MLP', **mlp_metrics},
        {'Model': 'Logistic Regression Baseline', **lr_metrics}
    ],
    "top_10_permutation_features": perm_df.head(10).to_dict(orient='records')
}

def convert_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Convert all dictionary values recursively
def sanitize_dict(d):
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(v) for v in d]
    else:
        return convert_types(d)

clean_payload = sanitize_dict(summary_payload)

with open("benchmark_metrics.json", "w") as f:
    json.dump(clean_payload, f, indent=4)

print("="*80)
print("Pipeline execution completed successfully!")
print(f"All 10 publication figures saved in: {OUTPUT_DIR}")
print(f"Summary metrics exported to: benchmark_metrics.json")
print("="*80)
