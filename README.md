# ML Project

An university wants you to build two different ML systems using the data they provide. 


## Part 1: Failure Prediction 

- The Goal: Build a model to predict a student's grade. 

The Use Case: The school will look at students with a predicted grade below 50 and offer them extra tutoring. 

- The Task: This is a classic regression (predicting a number) or potentially a classification (pass/fail) problem. You need to train a model on past student data. 

 

 

## Part 2: Automatic Correction (OCR) 

The Goal: Build a system to recognize handwritten characters and digits. 

The Use Case: To automatically grade handwritten exam answers. 

The Task: This is an image classification problem. You are given a dataset of character and digit images to train a model on. 

 

 

For Advanced AI Labs Students (4eiai40): 
You have an extra 5-page report requirement. You must analyze the project from three specific perspectives using the datasets: 

Security: How could a student game the system by lying on the input data to get undeserved tutoring? 

Bias: If the model uses sensitive variables (e.g., gender, background), does it create unfair bias? How can you measure and fix it? 

 

Explainability: How can you explain why a student was flagged for tutoring (which features mattered most)? How can you show a teacher which parts of an image the OCR model uses to recognize a letter? 

 

 

Checklist : 

1. Report Structure (Evaluation Grid) 
Your report needs clear sections that mirror the steps in the grid. The key sections are: 

EDA (Exploratory Data Analysis): Lots of graphs! Show patterns, trends, and insights, not just basic stats. 

Feature Selection & Creation: Justify why you chose certain features. If you create new features (e.g., average of past scores), explain why they are useful. 

Data Pre-processing: 

Missing Values/Outliers: Show you handled them with a "robust and nuanced approach." 

Categorical Encoding: Use proper methods (like one-hot encoding), don't just assign numbers arbitrarily. 

Scaling: Scale your numerical features and choose a method appropriate for your model. 

Modeling (The most heavily weighted part - 25%): 

Data Splitting: Use a proper train/validation/test split or cross-validation. Don't just do a single train/test split. 

Model Training: You must test at least 3 models. To get an excellent score, you need more than 3 models, including at least one deep learning model (like a neural network). 

Performance Comparison: Use multiple metrics (e.g., MAE, RMSE for regression; Accuracy, Precision, Recall for classification) and compare them in a structured way. 

Hyperparameter Tuning: Do this properly with cross-validation. 

Over/Underfitting: Explicitly discuss this. Did your model overfit? How do you know? What did you do to fix it? 

2. OCR Specifics (25% of grade) 

Dataset Preparation: This is crucial. Show your preprocessing steps (e.g., binarization, resizing, denoising, segmentation). 

Model Choice: You need a good justification for your model. For an excellent score, it must be an "excellent choice with strong justification." Convolutional Neural Networks (CNNs) are the standard here. 

Evaluation: The tip explicitly says: use a confusion matrix to see which characters/digits the model confuses. 

3. Presentation & Analysis (20% of grade) 

Critical Analysis: Don't just describe what you did. Discuss the limitations of your approach. Why might it not work perfectly? What are the trade-offs? 

Baseline: The tip explicitly tells you to use a baseline model (like predicting the average grade for everyone) to prove your model is actually better than a naive guess. 

4. The "Must-Do's" from the Tips 

Read the Evaluation Grid: The description says this multiple times. Use it as your checklist before submitting. 

Graphs: Put detailed graphs in the annex if you run out of space in the main 10 pages. 

5. For Advanced AI Labs Students (The 5-Page Supplement) 
You have three distinct tasks. Your report must address: 

Security (Adversarial Attack): How to minimally change inputs to get a "predicted grade < 50"? 

Bias (Fairness): Identify a sensitive variable, measure if the model is biased, and try to mitigate it. 

Explainability (XAI): 

For tabular data (failure prediction): Use techniques like SHAP or LIME to explain feature importance for a single prediction. 

For image data (OCR): Use techniques like Saliency Maps or Grad-CAM to show which pixels were important. 

 

 













## Venv setup : 

Use the Kaggle Jupyter Server, you can use the provided VSCode Compatible URL and use it as a as a remote server url.


```bash
python3 -m venv .venvKaggleProject2026
source .venvKaggleProject2026/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name MLProject2026 --display-name "Python (MLProject2026)"

# Launch the Jupyter Server
jupyter notebook
```



