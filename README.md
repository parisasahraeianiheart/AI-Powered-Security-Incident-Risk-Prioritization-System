# AI-Powered-Security-Incident-Risk-Prioritization-System

Project Methodology – End-to-End Approach

Step 1 — Problem Definition and Modeling Objective
•	Clearly defined the modeling target as:
Predict the probability that a network event represents a true security incident.
•	Framed the problem in operational SOC terms:
o	Output = p(\text{incident} = 1 | \text{features})
o	Threshold determines whether an alert should be escalated or deprioritized
•	Established business relevance by linking model decisions to analyst workload and risk management.
 
Step 2 — Data Quality Assessment and Exploratory Data Analysis (EDA)
•	Performed comprehensive data validation:
o	Checked for missing values and duplicates
o	Verified data types and feature distributions
•	Analyzed class imbalance and event distributions
•	Generated visualizations for:
o	Feature histograms
o	Train vs test comparisons
o	Attack category distributions
•	Documented initial insights and potential risks (e.g., imbalance and dataset shift).
 
Step 3 — Handling Class Imbalance
•	Quantified imbalance between normal vs attack traffic
•	Incorporated imbalance-aware modeling techniques:
o	Class weighting for logistic regression and SVM
o	scale_pos_weight for tree-based models
•	Evaluated metrics suited for imbalanced problems (PR-AUC, recall, cost-based metrics).
 
Step 4 — Train/Test Strategy and Leakage Prevention
•	Used the official UNSW-NB15 predefined train/test split as the final holdout
•	Ensured:
o	No information leakage between training and evaluation
o	Cross-validation used only on training data
•	Built preprocessing pipelines that were fitted strictly within CV folds.
 
Step 5 — Model Exploration and Deployment
•	Implemented and compared multiple modeling approaches:
o	Logistic Regression
o	Linear SVM
o	Nonlinear SVM
o	XGBoost
o	LightGBM
•	Initially experimented with nonlinear SVM but found it:
o	Computationally expensive
o	Less practical at enterprise scale
•	Transitioned to linear SVM for efficiency and scalability
•	Explored PCA-based dimensionality reduction + nonlinear SVM, but:
o	Performance gains were limited
o	Decided to retain simpler, interpretable linear models.
 
Step 6 — Results Analysis and Findings
•	Evaluated models using:
o	ROC-AUC
o	PR-AUC
o	Brier score
o	Cost-based thresholds
•	Identified tree-based models (XGBoost/LightGBM) as top performers
•	Conducted detailed comparison across:
o	Threshold-optimized performance
o	Business cost reduction
o	Precision/recall tradeoffs.
 
Step 7 — Observed Limitations and Future Improvements
•	Identified key challenges:
o	Potential distribution shift between train and test
o	Probability calibration needs
o	Threshold sensitivity to changing environments
•	Proposed improvements:
o	Regular threshold re-optimization
o	Calibration techniques (Platt scaling / isotonic)
o	Monitoring for concept drift
o	Time-based validation in production.
 
Step 8 — Model Interpretability
•	Applied SHAP explainability to the final models
•	Identified most influential features driving predictions
•	Generated global and local explanations for SOC analysts
•	Used insights to validate model logic and fairness.
 
Step 9 — AI-Augmented Development Process
•	Leveraged modern AI coding tools throughout the project:
o	Cursor
o	ChatGPT
o	VS Code Copilot
•	Used them to:
o	Debug issues
o	Refactor code
o	Improve performance
o	Standardize structure across Python modules.
 
Step 10 — Agentic AI Integration
•	Built an operational SOC Triage Agent that:
o	Selects optimal thresholds
o	Ranks alerts by risk
o	Generates explanations and suggested actions
•	Added optional LLM-powered summarization for:
o	Executive SOC briefings
o	Action recommendations
o	Escalation criteria.
 
Step 11 — Business Delivery Layer
•	Implemented a Streamlit dashboard to:
o	Visualize model outputs
o	Display prioritized alerts
o	Present explanations interactively
•	Transformed technical backend into a user-friendly analyst tool.
 
Outcome
The project evolved from a traditional ML classifier into a full end-to-end SOC decision system that combines:
•	Predictive modeling
•	Threshold optimization
•	Explainability
•	Agentic workflows
•	AI-assisted development
•	User-facing delivery

<img width="468" height="639" alt="image" src="https://github.com/user-attachments/assets/c6c4b8de-243f-48e4-8694-dc94437daeea" />
