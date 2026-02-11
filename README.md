# AI-Powered Security Incident Risk Prioritization System

An end-to-end machine learning system designed to help Security Operations Centers (SOCs) intelligently prioritize network security alerts using predictive analytics, explainable AI, and agentic workflows.

This project transforms raw network event data into actionable risk scores that help analysts focus on the most critical threats—reducing alert fatigue and improving response efficiency.

---

## Project Overview

Modern SOC teams face overwhelming volumes of security alerts. Most are false positives, yet each must be reviewed manually. This project addresses that challenge by building a machine learning system that:

- Predicts the probability that a network event is a true security incident  
- Prioritizes alerts based on estimated risk  
- Explains model decisions for analyst transparency  
- Integrates AI agents to recommend actions  
- Delivers insights through an interactive dashboard  

The result is a practical decision-support system that bridges advanced analytics with real-world security operations.

---

## Methodology – End-to-End Approach

### Step 1 — Problem Definition and Modeling Objective

- Defined the modeling target as:  
  **Predict the probability that a network event represents a true security incident**
- Framed in SOC terms:  
  - Output = `p(incident = 1 | features)`
  - A configurable threshold determines whether an alert should be escalated or deprioritized
- Established direct business impact by linking predictions to:
  - Analyst workload reduction  
  - Faster response times  
  - Improved risk management  

---

### Step 2 — Data Quality Assessment and Exploratory Data Analysis (EDA)

Performed rigorous data validation:

- Checked for missing values and duplicates  
- Verified data types and feature distributions  
- Analyzed class imbalance and attack category breakdowns  
- Compared training vs test distributions  
- Generated visualizations:
  - Feature histograms  
  - Event distributions  
  - Attack class frequencies  

Key risks such as imbalance and dataset shift were identified early.

---

### Step 3 — Handling Class Imbalance

- Quantified severe imbalance between normal and attack traffic  
- Applied imbalance-aware techniques:
  - Class weighting for logistic regression and SVM  
  - `scale_pos_weight` for tree-based models  
- Focused on appropriate evaluation metrics:
  - PR-AUC  
  - Recall  
  - Cost-sensitive measures  

---

### Step 4 — Train/Test Strategy and Leakage Prevention

- Used the official **UNSW-NB15 predefined train/test split** as the final holdout  
- Ensured strict separation between training and evaluation  
- Prevented information leakage by:
  - Performing cross-validation only on training data  
  - Fitting preprocessing pipelines inside CV folds  

---

### Step 5 — Model Exploration and Selection

Implemented and compared multiple approaches:

- Logistic Regression  
- Linear SVM  
- Nonlinear SVM  
- XGBoost  
- LightGBM  

Findings:

- Nonlinear SVM was computationally expensive and impractical at scale  
- Linear SVM provided strong performance with much better efficiency  
- PCA + nonlinear SVM offered limited gains  
- Final preference shifted toward scalable, interpretable models  

---

### Step 6 — Results Analysis

Models were evaluated using:

- ROC-AUC  
- PR-AUC  
- Brier Score  
- Business cost-based thresholds  

Key conclusions:

- Tree-based models (XGBoost / LightGBM) delivered the best overall performance  
- Detailed tradeoff analysis between:
  - Precision vs recall  
  - Threshold optimization  
  - Operational cost reduction  

---

### Step 7 — Limitations and Future Improvements

Identified real-world deployment challenges:

- Possible distribution shift between training and production  
- Need for probability calibration  
- Sensitivity to changing environments  

Proposed enhancements:

- Regular threshold re-optimization  
- Calibration techniques (Platt scaling / isotonic regression)  
- Drift monitoring  
- Time-based validation in production  

---

### Step 8 — Model Interpretability

To ensure analyst trust and transparency:

- Applied SHAP explainability  
- Identified most influential risk features  
- Generated both:
  - Global model explanations  
  - Local per-alert explanations  

These insights validate model logic and support decision-making.

---

### Step 9 — AI-Augmented Development Process

Development was accelerated using modern AI tools:

- Cursor  
- ChatGPT  
- GitHub Copilot  

These tools were used for:

- Debugging  
- Refactoring  
- Performance optimization  
- Standardizing project structure  

---

### Step 10 — Agentic AI Integration

Built an intelligent SOC Triage Agent that can:

- Select optimal decision thresholds  
- Rank alerts by predicted risk  
- Generate explanations  
- Suggest remediation actions  

Optional LLM components provide:

- Executive SOC summaries  
- Escalation recommendations  
- Context-aware guidance  

---

### Step 11 — Business Delivery Layer

A Streamlit-based interface was developed to:

- Visualize prioritized alerts  
- Display model predictions  
- Present SHAP explanations  
- Enable interactive analyst workflows  

This transforms backend ML into a practical analyst-facing application.

---

## Streamlit link

https://ai-powered-security-incident-risk-prioritization-system-gnhvnh.streamlit.app/

https://ai-powered-security-incident-risk-prioritization-system-egz9lf.streamlit.app/

---

## Outcome

This project evolved beyond a simple classifier into a full SOC decision-support platform that integrates:

- Predictive modeling  
- Threshold optimization  
- Explainability  
- Agentic AI workflows  
- AI-assisted development  
- User-friendly analytics delivery  

It demonstrates how machine learning can be operationalized to meaningfully improve security operations.

---

## Technologies Used

- Python  
- Scikit-learn  
- XGBoost / LightGBM  
- SHAP  
- Streamlit  
- Pandas / NumPy  
- AI coding assistants (Cursor, Copilot, ChatGPT)

---

## Dataset

This project utilizes the **UNSW-NB15** cybersecurity dataset, a widely used benchmark for intrusion detection research.

---

## Future Work

Planned extensions include:

- Real-time streaming ingestion  
- Continuous learning pipelines  
- Active learning from analyst feedback  
- Deployment to enterprise SOC environments  
- Integration with SIEM platforms  

---

## Author

Developed as an applied machine learning and AI systems project focused on practical cybersecurity impact.

