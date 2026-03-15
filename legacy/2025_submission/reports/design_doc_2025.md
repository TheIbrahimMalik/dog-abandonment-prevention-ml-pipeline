# Design document: Machine Learning for Dog Abandonment Prevention: Prediction, Explanation, and Recommendations

_Ibrahim Malik, September 2025_  




## Contents

- [I. Domain Background](#i-domain-background)
- [II. Problem Statement](#ii-problem-statement)
- [III. Datasets and Inputs](#iii-datasets-and-inputs)
- [IV. Solution Statement](#iv-solution-statement)
- [V. Benchmark Model](#v-benchmark-model)
- [VI. Evaluation Metrics](#vi-evaluation-metrics)
- [VII. Project Design](#vii-project-design)
- [VIII. References](#viii-references)



## I. Domain Background

Animal welfare is not at the standard it should be. Many animals are sadly suffering, but all animals deserve a good long life. There are usually two types of animals namely wild and domesticated. Wild animal welfare is important, however for this project I am focussing on domesticated animals which are animals bred to live alongside humans. Examples of these include dogs, cats, and rabbits. I will be focussing on dogs although the methodology used could be expanded in the future to include other domesticated animals and perhaps even wild animals with some modification. Since dogs are domesticated, they need humans to look after them for their survival and wellbeing. 

Dog abandonment is a big problem in the UK i.e., "In total 20,999 abandonment reports were made to the charity’s (RSPCA) emergency line in 2023 and according to latest 2024 figures (available up until the end of October) 19,067 have been reported this year (2023) – which, if the trend continues, will be almost 23,000 reports." (RSPCA Suffolk Central, 2023) and worldwide. This is due to various factors, such as overbreeding, unsuitable owners looking after dogs or certain dog breeds, and financial issues. This abandonment leads to significant multifactor issues such as harm to these dogs' physical and mental wellbeing, as well as pressure on dog shelters and even the streets such as aggressive and potential infectious untreated dogs. Solving this dog abandonment problem will improve dog wellbeing and quality of life, and less crowding and resource constraints on animal shelters.

People have proposed measures to help address this such as "regulation of online selling, public education and awareness, mandatory owner licensing, support for animal charities, and contributions towards rehoming centres" (Lambley, 2024). "Many countries have opted to implement sterilization campaigns (controlling reproduction), programs to eradicate abandoned dogs, and educational initiatives, but neither eradication campaigns nor sterilization programs have proven effective, potential rescue dog owners obliged to sign an adoption contract where they agree to sterilize the pet, provide identification, and give all the required vaccines" (Ortega-Pacheco, Jiménez-Coello and Segura-Correa, 2021) could be an effective solution but nevertheless this dog abandonment and wellbeing problem is still a severe ongoing issue.

By leveraging ML, this project aims to provide a data-driven approach that can tackle this problem whilst supplementing existing solution efforts, resulting in a better and more scientific dog welfare management strategy by predicting high-risk abandonment cases, explaining the likely causes, and recommending the best course of action for these cases.



## II. Problem Statement

Animal welfare organisations currently lack a proactive, data-driven system to effectively predict, explain, and mitigate the risk of dog abandonment. Existing solutions are reactive (meaning they respond only after a dog is returned or abandoned), resulting in high rates of re-abandonment. By enabling earlier interventions, shelters can reduce return rates, allocate resources more effectively, and improve adoption matching.

1. First Problem: Can we predict whether a dog will be abandoned after adoption?

2. Second Problem: If a dog is likely to be abandoned, can we identify the most probable reason (e.g., owner issues, behavioural problems, medical needs)?

3. Third Problem: Based on risk and reason predictions, can we generate recommendations for optimal adoption matches or interventions?

These three questions/problems directly solve our problem statement we mentioned above.

These problems are:

- Quantifiable: Through a variety of metrics i.e., with the outcome data, we can label abandoned vs. non-abandoned dogs vs. re-surrender rates. Our trained models will also output a probability score.
- Measurable: With standard evaluation metrics like AUC and F1-score to compare model performance.
- Replicable: Using publicly available shelter data.



## III. Datasets and Inputs

I wanted to use data from the UK so that the solution I created was most accurate for the UK, however I could not find suitable and large enough data for the UK. Therefore I had to look for data elsewhere and came across the Austin Animal Center Outcomes (10/01/2013 to 05/05/2025) dataset (City of Austin, n.d.d). As of August 2025 this dataset has 174K rows, 12 columns, with each row being one outcome per animal per encounter. "Outcomes represent the status of animals as they leave the Animal Center" (City of Austin, n.d.d). 

The 12 column names or features for the outcomes dataset are: 
1. Animal ID, 
2. Date of Birth, 
3. Name, 
4. DateTime, 
5. MonthYear, 
6. Outcome Type, 
7. Outcome Subtype, 
8. Animal Type, 
9. Sex upon Outcome, 
10. Age upon Outcome, 
11. Breed, 
12. Color. 

They are all according to the website of datatype text. 

I also came across its associated Austin Animal Center Intakes (10/01/2013 to 05/05/2025) dataset (City of Austin, n.d.c). As of August 2025 this dataset also has 174K rows, 12 columns, with each row being one outcome per animal per encounter. "Intakes represent the status of animals as they arrive at the Animal Center" (City of Austin, n.d.c).

The 12 column names or features for the intakes dataset are: 

1. Animal ID
2. Name
3. DateTime
4. MonthYear
5. Found Location
6. Intake Type
7. Intake Condition
8. Animal Type
9. Sex upon Intake
10. Age upon Intake
11. Breed
12. Color

They are all according to the website of datatype text apart from DateTime and MonthYear which are both Floating Timestamp datatype.

So essentially we have two datasets:
- Intakes: when an animal arrives at the shelter.
- Outcomes: when an animal leaves the shelter.

By comparing the two related datasets we can see that they both share the features or column names Animal ID, Name, DateTime, MonthYear, Animal Type, Breed, and Color. I assume although they share the feature names DateTime and MonthYear these two are different for each dataset with the outcomes being when the animal gets discharged and for intakes when the animal first reaches the Austin Animal Center but we can explore this and check further later on. 

The features or columns names that are different are Date of Birth which is there for outcomes but strangely not intakes. They both have type, condition, sex, and age, but they are different in each dataset where for outcomes they are specifically each of those with outcomes and similarly the same with intakes.

I will need both the intake conditions and the outcome type for the same animal and event to solve the problems. I will merge on Animal ID and perhaps using a matching time reference for nearest DateTime between intake and outcome for each animal and use this merged dataset as the main training dataset. Note that some animals may have multiple intake/outcome events.

When I downloaded the outcomes dataset (csv file), I found out via file properties that its size is 20.3 MB, subsequently I found that the intakes dataset is 23.7 MB.

"Austin is proudly the largest and longest running No Kill city in the nation (USA)" (Austin Pets Alive!, n.d.). "Most in the no-kill movement define a no-kill shelter, a no-kill city, a no-kill community or a no-kill nation as a place where all healthy and treatable animals are saved and where only unhealthy & untreatable animals are euthanized" (Maddie’s Fund, n.d.). I've also read from some places that a no-kill place is where/considered when the place reaches a "90% live animal outcome rate" (City of Austin, n.d.e). The Austin Animal Center has reached this rate since March 2010 and saved "approximately 20,000 animals entering the shelter each year following the guidelines laid forth in the City of Austin's No-Kill implementation plan, approved in March of 2010 by the Austin City Council" (City of Austin, n.d.e).

Helping to improve the rates of these existing shelters and strategies from this project will result in more momentum for no kill policies to spread elsewhere in the world as its shown to work successfully. This will result in more animals (in this case dogs) not losing their lives and living good long lives.

This large dataset is suitable for a range of ML approaches, from traditional algorithms to deep learning. I can use the 12 features to work out how some features impact the likelihood of another feature (target variable). I can also do feature engineering to create more features and improve this process. This dataset can be obtained by downloading it all as a CSV file or obtained via an API endpoint.



## IV. Solution Statement

My initial solution idea is first to clean the dataset and perform some feature engineering. Then I can create predictive models using various traditional and deep learning ML algorithms to work out the three specific problems to be solved in the form of a three-part ML pipeline namely: 
 1. The risk a particular dog has for abandonment (binary classification predictive model).
 2. The reason why for this potential abandonment (multi-class classification explainability component).
 3. A suggester or recommender system about how best to look after this abandoned dog such as who would be the best or most suitable person to adopt said dog (rule-based logic or clustering). 

1 and 2 seem to be supervised classification problems and 3 is a recommendation / matching system likely rule-based or unsupervised.
 
As an optional, post-submission extension, I will explore creating a LLM/Copilot extension using the outputs of the previous three stages to act as an assistant/adviser and summariser for the overall solution.



## V. Benchmark Model

The benchmark model (simplest model you can reasonably use) will be:

- Logistic Regression will provide a baseline AUC target for binary classification with simplicity, interpretability for tabular datasets.
- Multinomial Naive Bayes or Decision Tree will provide macro-F1 benchmarks for multi-class classification and provide interpretable decision boundaries.

These are interpretable, well-understood baseline models (usually from traditional ML but can also be a non-ML heuristic) that will be compared against more complex models like XGBoost or MLP (deep neural networks). This will determine if the extra complexity is worth it or not.



## VI. Evaluation Metrics

A metric measures some aspect of model performance or in another words how good the model is.

Binary Task (Abandonment Risk):

- PR-AUC over AUC-ROC: evaluates model performance across all classification thresholds and is needed for imbalanced datasets.
- F1-score
- Precision/Recall

Multi-Class Task (Reason):

- Accuracy
- Macro-F1 score (due to likely class imbalance): gives equal weight to each class regardless of its frequency.

These metrics will quantify classification performance, particularly in imbalanced scenarios typical in real-world data.

By imbalanced I mean that say if for instance 80% of the dogs in the dataset are not abandoned, and 20% are then the model could predict "not abandoned" for all cases and be deemed a highly accurate model during evaluation, but is not actually the case. Essentially, if one class has a much higher frequency than others we have an imbalance which could result in inaccurate results and so that's why our use of suitable evaluation metrics is so important to deal with this potential bias. 



## VII. Project Design

The solution will follow this high-level workflow:

1. Data Loading + Preprocessing

- Merge Outcomes and Intakes on animal_id and event date to derive labels
- Remove non-dog entries
- Handle missing values
- Encode categorical variables
- Feature engineering (age bins, neuter status, intake condition groups)

2. Exploratory Data Analysis (EDA)
- Visualize outcome distributions, breed effects, seasonal trends

3. Modeling

- Binary classification: Abandonment risk
- Multi-class classification: Abandonment reason
- Baseline models: Logistic Regression, Naive Bayes
- Advanced models: XGBoost, MLP

4. Evaluation and Interpretation

- Use cross-validation, confusion matrix, ROC, classification reports including using appropriate metrics for imbalanced datasets (PR-AUC, F1-score, macro-F1)

5. Deployment

- Deploy best performing model to Amazon SageMaker endpoint
- Optionally build an inference script to make predictions for new data

6. Optional / Post-submission: LLM Assistant

- Use GPT-style model to summarise predictions
- Offer recommendations or explanations in natural language

I will use SageMaker Processing jobs for the ETL (extract, transform, and load) process and feature engineering as everything will be kept organised in one managed environment that is also easier to scale with its associated tools. Model training and tuning will run on SageMaker's built-in containers and I will apply hyperparameter optimisation. I will store the best performing models in SageMaker Model Registry so they can be deployed as endpoints for testing and live-deployment.

To make the model as accurate as possible, the dataset will be split chronologically with the training being done on the first three quarters or so of the dataset in years and the testing on the final part of the dataset in years. This prevents data leakage as the model will not learn from future data.

We will deal with imbalanced classes during model training by using class weightings and could also oversample minority classes.



## VIII. References

Austin Pets Alive! (n.d.) Impact. Available at: https://www.austinpetsalive.org/impact (Accessed: 10 August 2025).

City of Austin (n.d.a) About Austin Animal Center [webpage]. Available at: https://www.austintexas.gov/department/about-austin-animal-center (Accessed: 9 August 2025).

City of Austin (n.d.b) Austin Animal Center [website]. Available at: https://www.austintexas.gov/austin-animal-center (Accessed: 9 August 2025).

City of Austin (n.d.c) Austin Animal Center Intakes (10/01/2013 to 05/05/2025) [dataset]. Available at: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes-10-01-2013-to-05-05-2/wter-evkm/about_data (Accessed: 9 August 2025).

City of Austin (n.d.d) Austin Animal Center Outcomes (10/01/2013 to 05/05/2025) [dataset]. Available at: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes-10-01-2013-to-05-05-/9t4d-g238/about_data (Accessed: 8 August 2025).

City of Austin (n.d.e) No Kill Plan [webpage]. Available at: https://www.austintexas.gov/page/no-kill-plan (Accessed: 9 August 2025).

Lambley, K. (2024) ‘A dog is for life, not just for Christmas: UK’s number of unwanted dogs hits crisis point’, Yorkshire Bylines, 21 December. Available at: https://yorkshirebylines.co.uk/opinion/a-dog-is-for-life-not-just-for-christmas-uks-number-of-unwanted-dogs-hits-crisis-point/ (Accessed: 8 August 2025).

Maddie’s Fund (n.d.) Defining No Kill: Editorial. Available at: https://www.maddiesfund.org/defining-no-kill-editorial.htm (Accessed: 10 August 2025).

Ortega-Pacheco, A., Jiménez-Coello, M. and Segura-Correa, J.C. (2021) ‘Abandonment of dogs in Latin America: Strategies and ideas’, Animals, 11(11), p. 3125. doi:10.3390/ani11113125. Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC8613775/ (Accessed: 8 August 2025).

RSPCA (2023a) Cost of living. Available at: https://www.rspca.org.uk/adviceandwelfare/costofliving (Accessed: 8 August 2025).

RSPCA (2023b) Pet food bank scheme. Available at: https://www.rspca.org.uk/adviceandwelfare/costofliving/foodbank (Accessed: 8 August 2025).

RSPCA (2023c) Lost dog advice. Available at: https://www.rspca.org.uk/adviceandwelfare/pets/lost/dog (Accessed: 8 August 2025).

RSPCA (2023d) Latest facts. Available at: https://www.rspca.org.uk/whatwedo/latest/facts (Accessed: 8 August 2025).

RSPCA (2023e) Kindness Index 2023. Available at: https://www.rspca.org.uk/whatwedo/latest/kindnessindex/2023/pet (Accessed: 8 August 2025).

RSPCA Suffolk Central (2023) RSPCA report: Shocking increase in animal abandonment in Suffolk. Available at: https://rspca-suffolkcentral.org.uk/rspca-report-shocking-increase-in-animal-abandonment-in-suffolk/ (Accessed: 8 August 2025).

UK Parliament (2025) Pet welfare and abuse: Government response 2024–25. Environment, Food and Rural Affairs Committee. Available at: https://committees.parliament.uk/publications/46060/documents/229283/default/ (Accessed: 8 August 2025).



-----------