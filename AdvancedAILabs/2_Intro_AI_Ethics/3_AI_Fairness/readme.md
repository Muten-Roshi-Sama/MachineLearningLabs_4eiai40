# AI Fairness tutorial

## Main idea

The tutorial asks a key question: what does “fair” mean for an ML model? There is no single universal answer. Different fairness definitions can conflict with each other, so you must choose the one that best fits the real problem.

## What the tutorial is about

You’ll learn four common fairness criteria for ML models that make decisions about people. The example used is a model that approves or denies credit card applications. Then, in the exercise notebook, you’ll train a few models and compare how fair they are.

## The 4 fairness criteria

- **Demographic parity / statistical parity**: The model is fair if the selected group has the same demographic composition as the applicant pool.
	- Example: if 50% of applicants are women, then 50% of approved people should be women.
	- This focuses on representation in the selected group, not whether the decisions are correct.
- **Equal opportunity**: The model is fair if the true positive rate is the same across groups.
	- In simple terms: among the people who should be approved, each group should be approved at the same rate.
	- This is useful when you care about not missing qualified people from any group.
- **Equal accuracy**: The model is fair if the overall accuracy is the same across groups.
	- That means the model is equally good at making correct decisions for each group.
	- This balances both false approvals and false denials.
- **Group unaware / fairness through unawareness**: The model is trained without using group membership features like gender or race.
	- The idea is that if the model “doesn’t know” the group, it can’t discriminate.
	- But this can fail if other features act as proxies for the group, like zip code for race.

## Why the tutorial uses confusion matrices

A confusion matrix shows how many predictions were correct or incorrect. By making one confusion matrix per group, you can check whether the model behaves differently for different groups. This helps measure fairness using real numbers instead of intuition alone.

## Exercise overview

It trains a simple model to approve or deny credit card applications. Then it checks whether the model treats two groups fairly. The dataset is synthetic, so it’s made for learning rather than real-world use.

### What data it uses

- `Income`
- `Num_Children`
- `Own_Car`
- `Own_Housing`
- `Group` — the protected group column, used to compare fairness between Group A and Group B
- `Target` — whether the applicant should be approved or denied

### Step 1: Load and split the data

The notebook loads the CSV file. It separates features (`X`) from the target (`y`). It splits the data into training and test sets. This lets the notebook train a model on one part and evaluate it on unseen data.

### Step 2: Train the baseline model

It uses a `DecisionTreeClassifier` with `max_depth=3`. The model is trained on the full feature set, including `Group`. Then it predicts on the test set. A helper function prints:

- total approvals
- approvals per group
- overall accuracy
- accuracy per group
- true positive rate per group

### What fairness metrics it checks

- **Demographic parity**: Are both groups represented equally among approved applicants?
	- Example: if applicants are 50/50 across groups, are approvals also 50/50?
- **Equal accuracy**: Is the model equally correct for both groups?
	- This compares overall prediction accuracy by group.
- **Equal opportunity**: Do both groups get the same true positive rate?
	- In other words, among people who should be approved, are both groups approved equally often?

### Step 3: Visualize the baseline model

The notebook prints the decision tree. This shows the exact rules the model uses. One important thing: the tree uses `Group` directly, so the model may make different decisions just because someone is in Group A or Group B.

### Why this matters

If the same applicant gets a different outcome only because of group membership, that’s a sign of unfairness. The notebook asks you to notice this by imagining the same person in a different group.

### Step 4: Train a “group unaware” model

The notebook removes `Group` from the input features. It retrains the decision tree. This is called fairness through unawareness.

### Important lesson

Removing `Group` does not guarantee fairness. Other features like income, car ownership, or home ownership may still act as proxies for group membership. So the model can still behave differently across groups even without the group column.

### Step 5: Compare fairness after removing `Group`

The notebook compares the new model to the baseline. You check whether:

- demographic parity improved
- equal accuracy improved
- equal opportunity improved

Often, one fairness metric gets better while another gets worse.

### Step 6: Try group thresholds

The notebook then adjusts thresholds differently for Group A and Group B. This is a way to force the approval rates to be more balanced. It helps illustrate that fairness often involves tradeoffs.

## Main lesson of the exercise

Fairness is not just one number. A model can look good overall but still treat groups differently. Different fairness definitions can conflict:

- improving demographic parity may hurt accuracy
- improving accuracy may hurt equal opportunity
- removing sensitive features may still not fix bias

## What you should take away

Always compare performance across groups, not just overall accuracy. Think carefully about which fairness definition matters most for the real use case. Fairness decisions are context-dependent and often require tradeoffs.


