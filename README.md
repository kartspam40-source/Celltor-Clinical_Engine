# 🧬 CellTor v1.0: Fairness-Aware Genomic Clinical Engine

**CellTor** is a Clinical Decision Support System (CDSS) designed to identify and mitigate ancestral bias in genomic variant interpretation. By implementing fairness-aware machine learning, CellTor ensures that pathogenic variants in under-represented populations (specifically African ancestries) are given equitable diagnostic weight.

---

## 🚀 Live Demo
Access the deployed clinical portal here: [Insert Your Streamlit Link Here]

## 🛠️ The Problem: The "Genomic Gap"
Current clinical databases (like ClinVar) are heavily skewed toward European ancestry (~80%). Standard "Ancestry-Blind" AI models trained on this data often fail to identify rare pathogenic variants in minority groups, leading to systemic misdiagnosis.

## ✨ Key Features
* **Batch Dataset Audit:** Upload large genomic files to detect systemic ancestral skew.
* **Real-time RSID Search:** Sub-second lookup for bias-mitigated pathogenicity predictions.
* **Explainable AI (XAI):** Integrated clinical insights explaining the ancestral distribution of specific variants.
* **Fairness Engine:** Utilizes **Inverse-Probability Sample Reweighting (IPSR)** within an XGBoost framework.

## 🧰 Tech Stack
* **Language:** Python 3.9+
* **Framework:** Streamlit (UI/Deployment)
* **ML Library:** XGBoost, Scikit-learn
* **Bioinformatics:** Pysam, 1000 Genomes Project Data (Chr 22)

## 🏃 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [your-repo-link]