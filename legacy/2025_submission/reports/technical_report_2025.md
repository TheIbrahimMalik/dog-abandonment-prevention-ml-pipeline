# Implementation Report: Machine Learning for Dog Abandonment Prevention: Prediction, Explanation, and Recommendations

_Ibrahim Malik, September 2025_  




## Contents

- [I. Definition](#i-definition)
- [II. Analysis](#ii-analysis)
- [III. Methodology](#iii-methodology)
- [IV. Results](#iv-results)
- [V. Conclusion](#v-conclusion)
- [VI. References](#vi-references)
- [VII. Further Reading](#vii-further-reading)



## I. Definition


### Project Overview

Dog abandonment is a severe global issue. Dogs are domesticated animals hence require humans to look after them. Abandonment causes mental and physical pain to them, shelter overcrowding and funding strain, and increases the number of dogs roaming wild on the streets, some of which are infectious due to a lack of vaccinations and treatment.

Solving this, specifically dog abandonment after adoption, will alleviate their suffering, which is the motivation for this project (ML for social good), in addition to my love of animals.

The data source I am using for this project is two datasets from the Austin Animal Center, specifically their Austin Animals Center Intakes dataset (City of Austin, n.d.a) and their Austin Animals Center Outcomes dataset (City of Austin, n.d.b). These two datasets both range from 10/01/2013 to 05/05/2025 and each have 174K rows and 12 columns. These datasets are suitable for this project because they provide the intake information (when an animal arrives at the shelter) and the outcomes data (when an animal leaves the shelter). 

This data provides the necessary information required to use ML to explore questions such as predicting dog abandonment risk, explaining the reason for abandonment, and recommending personalised intervention strategies.


### Problem Statement

Since I am trying to solve the issue of dog abandonment as detailed in the project overview, I want to achieve this via this **three core tasks ML-pipeline**: 

1. **Binary classification:** Predict whether a dog is likely to be abandoned after adoption. Helps shelters identify vulnerable cases early.  
2. **Multi-class classification:** Identify the most probable reason for abandonment. Helps tailor the intervention type.
3. **Recommendation modelling (hybrid):** Turns prediction and explanation results into actionable guidance. High-risk dogs are provided **personalised interventions** (such as medical care, training support), whilst adoption-ready dogs are **matched to suitable owners** (such as experienced vs. family-friendly adopter) based on their needs.

```
Intakes + Outcomes Data
        ↓
   Preprocessing
        ↓
┌──────────────┐
│ Binary Model │ → risk score
└──────────────┘
        ↓
┌──────────────┐
│ Reason Model │ → reason prediction
└──────────────┘
        ↓
┌─────────────────────────────┐
│ Hybrid Recommendation Layer │ → intervention OR adoption matching
└─────────────────────────────┘  
```

These predictions help shelters make more informed decisions and not exclusion. By knowing which dogs are most at risk and why, shelters can take proactive measures, whether its personalised interventions or matching to suitable adopters.

These three tasks are implemented through the following **ML workflow:**

1. **Preparation:** Collection, cleaning, integration of datasets, exploratory analysis, and data splitting.
2. **Modelling:** Training and refinement (model selection and tuning).
3. **Evaluation:** Testing with metrics, validating results, and interpretation.


### Metrics

**Binary Classification Task (Abandonment Risk):**

- **PR-AUC:** Performance under class imbalance. Captures ability to detect rare but critical abandonment cases.
- **F1-score:** Balances precision and recall, avoiding both false alarms and missed at-risk dogs.
- **Precision/Recall:** Precision ensures flagged dogs are truly at risk. Recall ensures vulnerable dogs aren’t overlooked.
- **Why not ROC-AUC?** ROC can look overly optimistic under imbalance, whereas PR-AUC reflects the real challenge.

**Multi-Class Classification Task (Reason for Abandonment):**

- **Accuracy:** Overall correctness of predictions, but biased toward common reasons.
- **Macro-F1:** Balances performance across all classes, ensuring rarer reasons (such as medical, housing etc...) are not ignored.
- **Why not Micro-F1?** Overweights common classes, masking poor performance on rare but important ones.

**Task 3 (Recommendations):** 
- Evaluated indirectly through Tasks 1 and 2, since reliable recommendations (intervention or matching) depend on accurate predictions from the two risk and reason tasks. 
- Real-world evaluation would involve monitoring adoption outcomes and intervention effectiveness after deployment.



## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration

I use two input files or datasets for this project:  

1. The Austin Animal Center **Intakes** dataset (...).  
- Size: 23.7MB  
- Date range: 10/01/2013 to 05/05/2025  
- Key fields: Animal ID, Name, DateTime, MonthYear, Found Location, Intake Type, Intake Condition, Animal Type, Sex upon Intake, Age upon Intake, Breed, Color.  
- Counts: Intakes shape: (173812, 12)
- Missing: Just 60.2% of intake_datetime.
- Dog only count: Dog intakes: (94608, 12)
- Class balance: Just under 65,000 are strays, just over 20,000 are owner surrender, just under 10,000 are public assist, about 1,500 are abandoned, and a very small amount maybe a few hundred max are euthanasia requested.

2. The Austin Animal Center **Outcomes** dataset (...).  
- Size: 20.3MB  
- Date range: 10/01/2013 to 05/05/2025  
- Key fields: Animal ID, Date of Birth, Name, DateTime, MonthYear, Outcome Type, Outcome Subtype, Animal Type, Sex upon Outcome, Age upon Outcome, Breed, Color.
- Counts: Outcomes shape: (173775, 12)
- Missing: 100% of outcome_datetime (or at least not in a recognisible format to be causing this inferred missingness), 54.16% of outcome_subtype, 0.03% of outcome_type, and 0.01% of age_upon_outcome.
- Dog only count: dog outcomes: (94505, 12)
- Class balance: Just under 50,000 are adopted, 22,000 are returned to owner, just over 20,000 are transferred, 2,000 are euthanised, 1,000 are RTO-adopted, a few hundred died and a small amount were sadly disposed.
- Top 5 dog breeds in outcomes: 1. pit bull mix, 2. labrador retriever mix, 3. chihuahua shorthair mix, 4. german shephard mix, 5. pit bull.

Tables 1 and 2 below show samples of the Intakes and Outcomes datasets. They show the types of features available and the common issues such as missing values (`nan`, `NaT`), inconsistent text formatting (such as misspelled locations), and categorical variables that will need preprocessing.  

**Table 1. Sample rows from the Intakes dataset**  
| animal_id | name | intake_datetime     | intake_monthyear | found_location         | intake_type | intake_condition | animal_type | sex_upon_intake | age_upon_intake | breed                        | color        |
|-----------|------|---------------------|------------------|------------------------|-------------|------------------|-------------|-----------------|-----------------|------------------------------|--------------|
| A521520   | Nina | 2013-10-01 07:51:00 | 2013-10-01       | Norht Ec in Austin (TX) | Stray       | Normal           | Dog         | Spayed Female   | 7 years         | Border Terrier/Border Collie | White/Tan    |
| A664235   | nan  | 2013-10-01 08:33:00 | 2013-10-01       | Abia in Austin (TX)     | Stray       | Normal           | Cat         | Unknown         | 1 week          | Domestic Shorthair Mix       | Orange/White |
| A664236   | nan  | 2013-10-01 08:33:00 | 2013-10-01       | Abia in Austin (TX)     | Stray       | Normal           | Cat         | Unknown         | 1 week          | Domestic Shorthair Mix       | Orange/White |  

**Table 2. Sample rows from the Outcomes dataset**  
| animal_id | date_of_birth | name | outcome_datetime | outcome_monthyear | outcome_type | outcome_subtype | animal_type | sex_upon_outcome | age_upon_outcome | breed      | color        |
|-----------|---------------|------|------------------|-------------------|--------------|-----------------|-------------|------------------|------------------|------------|--------------|
| A668305   | 2012-01-12    | nan  | NaT              | 2013-12-01        | Transfer     | Partner         | Other       | Unknown          | 1 year           | Turtle Mix | Brown/Yellow |
| A673335   | 2012-02-22    | nan  | NaT              | 2014-02-01        | Euthanasia   | Suffering       | Other       | Unknown          | 2 years          | Raccoon    | Black/Gray   |
| A675999   | 2013-03-04    | nan  | NaT              | 2014-04-01        | Transfer     | Partner         | Other       | Unknown          | 1 year           | Turtle Mix | Green        |



### Exploratory Visualisation

**Fig 1. Intake Type Distribution**  
![Intake Type Distribution](../figures/dogs_intake_type_bar.png)

Fig 1. shows that the majority of dogs intake type are from strays with over 70,000 followed by owner surrendered at around 20,000 and then public assist at under 10,000 with abandoned and euthanasia requested very small. This is useful as the intake type is likely linked in some way to the adoption type so understanding this as well as class imbalances is useful for the construction of our models.

**Fig 2. Intake Owner Surrender Rate by Age**  
![Intake Owner Surrender Rate by Age](../figures/owner_surrender_rate_by_agebin.png)

Fig 2. shows are fairly equal distribution in terms of surrender rate by age. However the older years particularly 3-7 years and 7+ years seem to have the highest chance of owner surrender but not significantly more than the younger years. Interestingly the 1-3 years has the lowest owner surrender rate being even less than puppies or dogs under 1 year. This is probably explained by the fact that younger dogs need more effort to look after them and sadly unsuitable people get dogs of which they don't take the responsibility for.

**Fig 3. Outcome Type Distribution**  
![Outcome Type Distribution](../figures/dogs_outcome_type_bar.png)

Fig 3. shows that the majority of outcome types is adoption with just over 50,000 followed by return to owner and transfer both being just over 20,000. Euthanasia, rto-adopt, died, and disposed are also there but all seemed to be less than 1,000.

**Fig 4. Top 15 Breeds Outcomes**  
![Top 15 Breeds Outcomes](../figures/dogs_top_breeds_bar.png)

Fig 4. shows that the majority of dog breeds based on outcomes are mixes namely pit bull mix, labrador retriever mix, chihuahua shorthair mix, and german shephard mix, following by a non-mixed pit bull.

**Fig 5. Days from Intake to Nearest Outcome**  
![Days from Intake to Nearest Outcome](../figures/days_to_outcome_hist.png)

Fig 5. shows that most of the time dogs have an outcome from an intake mostly within the first 25 days followed by 50 days and as time goes by the amount of dogs from intake and outcome decreases significantly.

### Algorithms and Techniques

As previously mentioned, the data type/modality we are dealing with in this project is in structured tabular categorical csv form (data structued in a table with rows and columns representing records and fields/attributes). Therefore there are particular algorithms and techniques most suitable for tabular data that we can explore and subsequently discard those algorithms and techniques which are not suitable for this data format.

I go into detail about my merged preprocessed tabular modelling dataset in the "Data Preprocessing" section in "III. Methodology". However the algorithms and techniques most appropriate/suitable for our modelling table with intake-side features and labels will be discussed and justified here.

For our tabular dataset especially considering its not overly large classical ML as opposed to deep learning would be more effective. We also need to balance interpretability with performance under class imbalance. All of the algorithms have hyperparameters which we can modify during training to fine the best combination pairs.

**Binary Classification (Abandonment Risk)**:

- Logistic regression: this linear model is the simplest and acts as a good easy simple model. Its very interpretable thus we can work out what features have the most predictive power.

- Decision tree: goes a step further than logistic regression as decision trees can handle non-linear relationships meaning more complex patterns and relationships can be captured. Its also inherit via its model design works well with categorical data and is very interpretable.

- Random forests: a ensemble (combines multiple models) of decision trees to produce an overall better single new model. Trees for RFs are built in parallel and it has all the same benefits as decision trees but just usually better.

- XGBoost: also an ensemble but this time its the tree building is done sequentially. This ensemble is very powerful as when the model is developed it aims to correct errors of previous trees resulting in a very powerful model.

- Multi-layer perceptron (MLP): use a simple neural network architecture to see how well it performs. Its less interpretable than the classical ML algorithms shown before but its very good on large and complex datasets. Its worth trying to see if it is comparable to the previous algorithms.


**Multi-Class Classification (Reason for Abandonment)**:

- Logistic regression: good easy simple model to try although its likely to struggle capturing the complexity of the multi-class data.

- Decision trees / random forests: same explanation from before except also very good at dealing with (both performance and interpretability) in multi-class problems.

- XGBoost: same as before, one of the state-of-the-art models for tabular data so very powerful and effective. Said to performs well even in class imbalance aka rare cases.

- Naive Bayes: probabilistic model that is known to work well on categorical data and is also simple and fast.

**Recommendation Layer (Hybrid Decision Support)**:

This will not use its own ML model but will rather use the results from the first two tasks and use a subsequent rule or post-processing logic based algorithms to generate appropriate actionable strategies.

**Techniques Applied**:

- **Feature encode** categorical variables (age, sex, breed...) using one-hot encoding so that my dataset is ML usable.

- Apply **class imbalance handling** such as class weightings or resampling strategies (SMOTE etc...) for rare classes so the models are not biased for the majority classes and ensure overall accurate and representative results.

- Use appropriate **evaluation metrics** such as PR-AUC and F1 for binary classification and macro-f1 for multi-class classification to ensure performance across both common and rare outcome reasons. 

### Benchmark

The following are appropriate baseline benchmarks we use for both taks (binary and multi-class), but just applied to different label sets appropriate for each task. These benchmarks are simple, intuitive, and naive strategies which we can compare to our actual models to see if those models are useful and an improvement to our benchmarks or not.

**Majority class predictor** (stronger naive bar): 
- Always predicts the most frequent class such as not abandoned for binary classification and adoption for multi-classification.
- In sklearn: DummyClassifier(strategy="most_frequent").
- Expected performance:
  - Binary task: Since ~75–80% of dogs are not abandoned, the accuracy would be ≈ 0.75–0.80. However, PR-AUC would be ≈ 0.00 because the model never predicts abandonment.
  - Multi-class task: Since ~65% of dogs enter as strays, the accuracy would be ≈ 0.65, but the Macro-F1 would drop to ≈ 0.20 because all other classes are ignored.

**Stratified random predictor** (weaker naive bar): 
- Assign classes at random but weighted according to their frequency distribution in the training set.
- In sklearn: DummyClassifier(strategy="stratified").
- Expected performance:
  - Binary task: Accuracy ≈ 0.50, PR-AUC ≈ 0.50 (equivalent to random guessing).
  - Multi-class task: Accuracy ≈ 0.20 and Macro-F1 ≈ 0.20–0.25, reflecting chance-level performance across ~5 categories.

Our models must score higher than these benchmark metric thresholds ensuring that they are useful.



## III. Methodology

### Data Preprocessing

In order for my Austin Animal Center data to be ready for ML modelling of my tasks I described, I performed the following preprocessing steps in 01_data_preparation.ipynb:

1. **Merged datasets**: 
    - I joined the intakes and outcomes dataset on animal_id with the nearest matching datetime.
    - The ID overlap from this was high ~99.4% which suggests a reliable alignment result.

2. **Filtering**: 
    - I kept only of the datatype dog entries and removed all other animals from the data since our project is based on only on dog abandonment prevention for the time being.
    - This resulted in 94,608 dog intakes and 94,505 dog outcomes which significantly reduced the dataset rows from thw intakes which was before 173,812 and 173,775 for outcomes.

3. **Handled missing values**: 
    - I dropped rows when permitted (such as when the amount of rows was insignificant to the overall dataset) which had missing data such as missing outcomes (outcome type).
    - I imputed or retained rows where feasible:
      -  For instance outcome_datetime for some reason (perhaps data type issues) was 100% missing so I used outcome_monthyear instead.
      - outcome_subtype was missing in 54% of rows but this feature was retained by encoding missing values with "Uknown".

4. **Encoding**: 
    - Applied one-hot encoding on categorial variables such as intake type, breed, colour, outcome type, and sex which made our data usable for ML modelling.

5. **Feature engineering**:
    - Created age bins: <1 year, 1-3 year, 3-7 years, 7+ years. 
    - This gave me descriptive groupings of young, teen, middle age, and old dogs.

6. **Train/test split**:
    - Used a chronological split (time-series ordered) instead of a random split to avoid data leakage.
    - Train had 75,686 rows and test had 18,922 applied to both binary and multi-class tasks (80/20 split).
    - This ensures the models generalise well when evaluated for future unseen data.

### Implementation

Although my original proposal stated that I will predict abandonment risk after adoption and then classify the most likely reason for said abandonment, the constraints of the Austin Animal Center dataset led me to implementing a slightly different implementation although I discuss how to solve the original proposal in the reflection and improvement section in the conclusion.

**Task 1. Binary Classification** (adoption vs non-adoption):
  - Target variable (dependent variable): derived from outcome_type.
  - Positive class or 1 equates to adopted and likewise all other outcomes (transfer, euthanasia, return to owner etc...) is a negative class.
  - Slightly changes scope from purely abandonment risk after adoption to adoption vs non-adoption outcome (what factors lead to adoption vs other outcomes).
  - Naive benchmarks: majority-class and stratified predictors (scikit-learn).
  - Baseline model: logistic regression (scikit-learn).
  - Baseline Logistic Regression model raised convergence warnings (likely due to lack of feature scaling), but results were still usable for comparison.

**Task 2. Multi-class Classification** (intake type):
  - Target variable (dependent variable) was derived from the intake_type.
  - Classes: stray, owner surrender, public assist, abandoned, euthanasia requested.
  - Changed scope from proposal of reason for abandonment to intake type classification (what reasons led to the dog entering the shelter in the first place).
  - Naive benchmarks: Majority-class and stratified predictors (scikit-learn).
  - Baseline model: Decision Tree (scikit-learn).
  - Complications: Severe class imbalance (rare classes like Abandoned and Euthanasia Requested) constrained performance and led to frequent misclassifications.

### Refinement

**Task 1. Binary Classification** (Adoption vs non-adoption):
  - Refined model: XGBClassifier (XGBoost).
  - Hyperparameter tuning: RandomizedSearchCV on n_estimators, max_depth, subsample, colsample_bytree, learning_rate, min_child_weight.
  - Best configuration: n_estimators=400, max_depth=6, subsample=0.8, colsample_bytree=0.7, learning_rate=0.05.
  - Class imbalance: handled with scale_pos_weight≈9.46.
  - Cross validation gave average precision (AP/PR-AUC) of 0.141 which is (+-0.019) so only ever so slightly better than logistic regression.

**Task 2. Multi-class Classification** (intake type):
  - Refined model: RandomForest (scikit-learn). This model was chosen because its an ensemble and so should be more powerful resulting in a reduction in variance and improved generalisation.
  - Hyperparameter tuning: explored depth and number of estimators however there seemed to be no best performance or hyperparameter configuration.
  - Improvement over Decision Tree was limited, reflecting data imbalance and sparsity. Techniques like class weightings or oversampling could improve the model performance in the future.



## IV. Results

### Model Evaluation and Validation

**Task 1. Binary Classification** 
  - Final best model chosen was the XGBClassifier (XGBoost) with the optimal found tuned hyperparameters of n_estimators=400, max_depth=6, subsample=0.8, colsample_bytree=0.7, learning_rate=0.05.
  - Evaluation metric used was the average precision (AP/PR-AUC) as it measures performance under class imbalance.
  - Results (best to worst):
    1. Logistic regression (baseline): AP~0.099 and F1~0.149.
    2. XGBoost (refined): AP~0.097 and best F1~0.161.
    3. Majority class (baseline): AP~0.057.
  - Validation: timeseriessplit cross-validation gave AP~0.141 which is a small increase from the baseline.

**Task 2. Multi-class Classification** (intake type):
  - Final model chosen was Decision tree (scikit-learn) as the best performing model as it slightly outperformed the RandomForest ensemble.
  - Evaluation metrics used were accuracy (overall performance) and macro-f1 (performance across imbalanced classes).
  - Results (best to worst):
    1. Decision tree (baseline): accuracy~0.478, macro-f1~0.387.
    2. Randomforest (refined): accuracy~0.468, macro-f1~0.316.
    3. Stratified predictor (baseline): accuracy~0.342, macro-f1~0.225.
  - Validation: model predicted the majority classes such as stray and ownder surrender well but struggled on rare classes such as abandoned and euthanasia. This resulted in lower macro-f1 scores.
 
 For both tasks and associated best model, increases in training time or model depth (validation, sensitivity analysis, and robustness) don't seem to improve model performance suggesting the limitations are with the data and specifically class imbalance and limited features than say the model itself or model instability.

### Justification

**Final results compared to benchmarks**:
- **Binary task**: Both logistic regression and XGBoost had improved AP scores compared to the majority baseline naive strategy score 0.099 compared to 0.057 (57.6% better). This improvement suggests that the models learned some predictive patterns from the intake features.
- **Multi-class task**: Decision tree had an improved macro-f1 score compared to the baseline stratified predictor with score of 0.387 compared to 0.225 (58% improvement). The simpler decision tree was better at dealing with the class imbalances than the randomforest.

**What these results mean**:
- Whilst the models showed an improvement compared to the naive baseline strategies which shows an initial promising result (proof of concept), the scores achieve i.e. 0.099 AP for binary classification and 0.387 macro-f1 are far too low for actual deployment. 
- We would want scores of at least 0.5 for binary AP (between random and perfection) and around 0.7 for macro-f1 multi-class (good score and also performs well with the rarer classes).
- People in the animal rescue team will generally have a pretty good understanding of dog risks and reasons by their specialist knowledge and experience so our job is to verify this but also help catch the outliers or dogs which are harder to predict and in order to achieve this we need to get very high metric scores.



## V. Conclusion

### Free-Form Visualization

**Fig 6. Confusion Matrix for Binary Classification (baseline logistic regression)**  
![Confusion Matrix for Binary Classification (baseline logistic regression](../figures/cm_binary_bestthr.png)

This visualisation shows a confusion matrix for the binary adoption vs non-adoption task.

- We can see that the majority of non-adoption (15,296) were correctly identified, however only 313 true positives were correctly caight (dogs corrected predicted at adopted).
- There were 764 false negatives (dogs incorrectly predicted as not adopted) and 2,549 false positives (dogs incorrectly predicted as adopted).

What this means:
- The model is biased to predicting the majority class (class imbalance) namely non-adoption and has poor recall for the positive class (adopted).
- Threshold sensitivity: it has an optical decision threshold of 0.58 balancing precision and recall which is decent however it still is unable to or misdetects many adoption cases. 
- This is very important because the at risk dogs need to be correctly identified and dogs mis-identified or missed means we haven't done our job and it could hinder our dog abandonment prevention efforts.

### Reflection

I have successfully built two ML models that complete the two defined tasks:
1. Adoption vs non-adoption (binary classification)
2. Intake type (multi-class classification)

However, as I noted during implementation this is slightly different to that of my initial proposal. The proposal was about predicting post-adoption abandonment risk and the subsequent reasons for said abadonment. However our dataset did not have the necessary labels for this type of modelling so I used adoption outcomes and intake types instead as sort of proxies or the closest to that.

I think the core determinator of project success I have learnt whilst doing this project is that its the quality of data and data preprocessing that seems to be the hardest and most important thing. Good clean and informative data seems to be the core of ML.

Project challenges:
  - Severe class imbalance which affected minority class prediction and subsequently class-imbalance adjusted performance evaluations.
  - Feature limitations as only intake and outcome data was available and no adopter demographics or longer term data. Additional useful features even in the existing dataset would have been very useful.

### Improvement

- Have better labels so we can actually link the after adoption results with abadonment risk.
- Perform better imbalance data handling techniques such as the application of class weightings, SMOTE oversampling.
- We could improve our dataset by working out with data we already have other data that it ought to have and add it. We can also enrich our dataset by merging other useful and meaningful datasets that link nicely with what we have.
- Extend our decision support wrapper into a proper recommendation system by perhaps using collaborative filtering and linking Chatbot (via APIs) and LLMs to create a great detailed yet insightful recommendations.



## VI. References

Austin Animal Center. (2013–2025). *Intakes and Outcomes Data*. [https://data.austintexas.gov](https://data.austintexas.gov)

City of Austin (n.d.a) Austin Animal Center Intakes (10/01/2013 to 05/05/2025) [dataset]. Available at: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes-10-01-2013-to-05-05-2/wter-evkm/about_data (Accessed: 9 August 2025).

City of Austin (n.d.b) Austin Animal Center Outcomes (10/01/2013 to 05/05/2025) [dataset]. Available at: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes-10-01-2013-to-05-05-/9t4d-g238/about_data (Accessed: 8 August 2025).

Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS. [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)

Scikit-learn Developers. (2024). *Scikit-learn: Machine Learning in Python*. [https://scikit-learn.org](https://scikit-learn.org)