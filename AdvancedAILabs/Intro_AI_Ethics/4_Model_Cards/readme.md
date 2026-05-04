# Model Cards

## Introduction

A model card is a short document that provides key information about a machine learning model. Model cards increase transparency by communicating information about trained models to broad audiences.

In this tutorial, you will learn which audiences to write a model card for and which sections a model card should contain. Then, in the exercise, you will apply what you have learned to a couple of real-world scenarios.

## What model cards are

Model cards, introduced in a 2019 paper, are one way for teams to communicate key information about their AI systems to a broad audience. This information generally includes:

- intended uses for the model
- how the model works
- how the model performs in different situations

You can think of model cards as similar to the nutritional labels on packaged foods.

## Who model cards are for

A model card should balance being easy to understand with communicating important technical information. When writing one, think about the audience: the groups of people most likely to read it.

For example, a model card for an AI system that helps medical professionals interpret x-rays is likely to be read by medical professionals, scientists, patients, researchers, policymakers, and developers of similar AI systems. The model card may therefore assume some knowledge of health care and AI systems.

## What a model card should contain

Per the original paper, a model card should include nine sections. Different organizations may add, remove, or rearrange sections depending on their needs.

Before proceeding, review the example model cards from the original paper:

- Model Card - Smiling Detection in Images
- Model Card - Toxicity in Text

### 1. Model Details

Include background information such as the developer and model version.

### 2. Intended Use

- What use cases are in scope?
- Who are the intended users?
- What use cases are out of scope?

### 3. Factors

Explain which factors affect the model’s impact. For example, the smiling detection model’s results vary by demographic factors like age, gender, or ethnicity, environmental factors like lighting or rain, and instrumentation like camera type.

### 4. Metrics

Describe the metrics used to measure performance and why they were chosen.

- For classification systems, consider error types such as false positive rate, false negative rate, false discovery rate, and false omission rate.
- For score-based analyses, consider reporting performance across groups.

### 5. Evaluation Data

- Which datasets were used to evaluate performance?
- Why were those datasets chosen?
- Are they representative of typical, anticipated, or challenging cases?

### 6. Training Data

Describe which data the model was trained on.

### 7. Quantitative Analyses

Report how the model performed on the chosen metrics and break down performance by important factors and their intersections. For example, performance may be shown by age, gender, and then by both together.

### 8. Ethical Considerations

Describe ethical concerns related to the model, such as sensitive data used during training, implications for human life, health, or safety, how risk was mitigated, and what harms may still be present in use.

### 9. Caveats and Recommendations

Add anything important that was not covered elsewhere in the model card.

## Using model cards in practice

Detailed model cards can be challenging because organizations may not want to reveal proprietary data, internal processes, or trade secrets. In those cases, teams should think about how model cards can still be useful and empowering without including sensitive information.

Some teams use other formats, such as FactSheets, to collect and log ML model information.

## Exercise

Use what you learned to decide how to apply model cards in real-world scenarios.
