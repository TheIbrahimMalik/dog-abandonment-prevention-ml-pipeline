# Machine Learning for Dog Abandonment Prevention: Prediction, Explanation, and Recommendations

_Ibrahim Malik, September 2025_




## 📜 Table of Contents
- [📄 Proposal & Report](#proposal--report)
- [📌 Project Description](#project-description)
- [⚙️ Setup & Installation](#setup--installation)
- [🚀 How to Run](#how-to-run)
- [📂 Repository Structure](#repository-structure)
- [📊 Datasets](#datasets)
- [🛠️ Software & Libraries](#software--libraries)
- [📚 References](#references)
- [🔒 License](#license)



## 📄 Proposal & Report

This project is two-staged: 
1. [Proposal](reports/design_doc.md) - outlines the project idea, design, and proposed solution. 
2. [Report](reports/technical_report.md) - write-up of the implementation including results, analysis, and conclusions.  



## 📌 Project Description

Dog abandonment after adoption is a significant challenge for animal welfare organisations, leading to overcrowded shelters, increased operational strain, and negative outcomes for the animals. This project leverages machine learning to help prevent abandonment by predicting risk, explaining the key contributing factors, and recommending proactive interventions.

Using historical intake and outcome records from the Austin Animal Center, the system performs three core tasks:  

1. **Binary classification:** Predict whether a dog is likely to be abandoned after adoption.  
2. **Multi-class classification:** Identify the most probable reason for abandonment.  
3. **Recommendation modelling (hybrid):** Use predictions for decision support. High-risk dogs trigger **interventions** (e.g., veterinary care, training), while adoption-ready profiles are **matched to owners** who can meet their needs.

The pipeline spans preprocessing, EDA, modelling, and interpretability, with deployment on AWS SageMaker for scalability. By enabling early intervention, it aims to improve adoption outcomes and reduce shelter overcrowding and ultimately improve animal welfare.



## ⚙️ Setup & Installation

### AWS SageMaker Studio
1. **Open SageMaker Studio** and start a **Conda-based kernel** (e.g., Python 3 (Data Science)).
2. **Install dependencies** in the Studio terminal:
    ```bash
    pip install -U numpy pandas scikit-learn matplotlib jupyter
    ```
3. **(Optional) Export your dev environment:**
    ```bash
    pip freeze > requirements-dev.txt
    ```



## 🚀 How to Run

### Run Notebooks
1. Open and run `notebooks/01_data_exploration.ipynb` to explore and clean data.
2. Open and run `notebooks/02_model_training.ipynb` to train baseline and advanced models and for evaluation and interpretation. 



## 📂 Repository Structure

```
├── .gitignore              # Git ignore file
├── LICENSE                 # Project license
├── README.md               # Project README
├── requirements.txt        # Python dependencies
├── data/                   # Raw and processed datasets (samples only in repo)
│   ├── raw/                # Original CSVs (sample_intakes.csv, sample_outcomes.csv)
│   └── processed/          # Cleaned/merged datasets for modeling
├── notebooks/              # Jupyter notebooks for EDA, training, evaluation
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── src/                    # Source code for preprocessing, training, inference
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py   # Will modularise in here post-submission
│   ├── predict.py          # Will modularise in here post-submission
│   ├── processing/         # Scripts for SageMaker Processing Jobs
│   └── training/           # Scripts for SageMaker Training Jobs
├── models/                 # Saved model artifacts
├── figures/                # Generated plots and visualisations
└── reports/                   # Proposal, report, and design reports
    ├── proposal.md
    └── report.md
```



## 📊 Datasets

1.  **Dataset Name:** Austin Animal Center Intakes (10/01/2013 to 05/05/2025)
    * **Source:** [City of Austin, Texas - data.austintexas.gov](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes-10-01-2013-to-05-05-2/wter-evkm/about_data)
    * **License:** Public Domain

2.  **Dataset Name:** Austin Animal Center Outcomes (10/01/2013 to 05/05/2025)
    * **Source:** [City of Austin, Texas - data.austintexas.gov](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes-10-01-2013-to-05-05-/9t4d-g238/about_data)
    * **License:** Public Domain



## 🛠️ Software & Libraries

- Python 3.10  
- NumPy 1.25  
- pandas 2.1  
- scikit-learn 1.3  
- matplotlib 3.7  
- jupyter 1.0



## 📚 References

City of Austin (n.d.a) Austin Animal Center Intakes (10/01/2013 to 05/05/2025) [dataset]. Available at: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes-10-01-2013-to-05-05-2/wter-evkm/about_data (Accessed: 9 August 2025).

City of Austin (n.d.b) Austin Animal Center Outcomes (10/01/2013 to 05/05/2025) [dataset]. Available at: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes-10-01-2013-to-05-05-/9t4d-g238/about_data (Accessed: 8 August 2025).



## Attribution

Originally completed as part of the Udacity AWS Machine Learning Engineer Nanodegree Capstone (2025).



## 🔒 License

[MIT LICENSE](LICENSE)



-----------