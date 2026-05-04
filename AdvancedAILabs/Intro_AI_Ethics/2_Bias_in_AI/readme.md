# Identifying Bias in AI:



## Big Idea

- ML can help people, but it can also cause harm, especially when it reinforces or amplifies unfair outcomes for certain groups.
- In this tutorial, “bias” means unwanted negative consequences in ML systems.


## The 6 Types of Bias

### 1. Historical bias
The world the data came from was already unfair, so the model learns those unfair patterns.

### 2. Representation bias
Some groups are missing or underrepresented in the training data, so the model works poorly for them.

### 3. Measurement bias
The data is less accurate for some groups, often because of flawed proxy variables or inconsistent measurements.

### 4. Aggregation bias
Different groups are lumped together when they should be modeled separately, causing the model to fit the majority group better than others.

### 5. Evaluation bias
The benchmark or test data does not reflect the real population the model will serve, so performance looks better than it really is.

### 6. Deployment bias
The model is used in a way that is different from the way it was designed or tested, which can create harm in practice.


## Main Takeaways

Bias can happen at every stage of the ML pipeline: data collection, training, evaluation, and deployment.
A model can suffer from more than one type of bias at the same time.
Spotting bias requires looking beyond accuracy and asking who the model works for, who it fails, and why.


## Exercise


- The notebook trains a simple text classifier to decide whether a comment is toxic or not toxic.
- The goal is not just to build a model, but to notice how bias can show up in real data and affect predictions.
- You’re using a real dataset from the Jigsaw Unintended Bias in Toxicity Classification competition.


## 1.Load the data

- The notebook reads comment text and labels from a CSV file.
- It splits the data into training and test sets.
- It converts text into word-count features using CountVectorizer.

## 2.Train a simple model

A LogisticRegression model is trained on the text features.
The model is evaluated on test data and gets around 93% accuracy.
That sounds good, but the exercise shows why accuracy alone can hide bias.
Test the model on example comments

You try short comments like "I love apples" and "Apples are stupid".
This helps you see that the model is learning from word patterns, not real understanding.
It may misclassify comments based on specific words or associations.
Inspect the most toxic words

The notebook shows the words with the highest toxic coefficients.
This helps you see which words the model strongly associates with toxicity.
Some words may be surprising or problematic, especially if they are identity-related.
Probe for bias with identity terms

You compare comments like:
"I have a christian friend"
"I have a muslim friend"
"I have a white friend"
"I have a black friend"
This checks whether the model treats some identity words more negatively than others.
Answer conceptual questions about bias

The final questions ask you to identify which type of bias is happening in different scenarios.
These are not coding tasks; they’re meant to test whether you understand the bias concepts from the tutorial.





