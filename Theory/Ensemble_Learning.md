# Ensemble Learning

> Notes by Arzaan 

---

## What is Ensemble Learning?

Instead of training one model and relying on it, ensemble learning combines multiple models so their collective output is better than any single model alone.

**Core intuition:**
> One person's opinion = guess. Many diverse experts' opinions = wisdom of the crowd.

**Why it works — bias-variance tradeoff:**
- A single model either has high variance (overfits) or high bias (underfits)
- Combining models averages out individual errors → lower overall error

---

## Three Types of Ensemble Learning

```
                    Ensemble Learning
                   /        |         \
              Bagging    Boosting    Stacking
           (parallel)  (sequential) (meta-learner)
```

---

## 1. Bagging (Bootstrap Aggregating)

**Idea:** Train multiple models **in parallel** on different random subsets of the data, then aggregate their predictions.

**How it works:**
```
Full data
  → bootstrap sample₁ → Model₁ → prediction₁
  → bootstrap sample₂ → Model₂ → prediction₂    →  vote / average
  → bootstrap sample₃ → Model₃ → prediction₃
```

- **Bootstrap sampling** = sampling with replacement (same row can appear multiple times)
- **Aggregation** = majority vote (classification) or average (regression)
- Models are trained **independently** — no model knows what the other did

**Reduces:** Variance (fixes overfitting)

**Example:** Random Forest
- Many Decision Trees, each trained on a random subset of data AND random subset of features
- Final prediction = majority vote of all trees

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

**Drawback:**
- Cannot fix a biased model — if the base learner is bad, averaging bad models is still bad
- More compute than a single model

---

## 2. Boosting

**Idea:** Train models **sequentially** — each new model focuses on correcting the mistakes of the previous one.

**How it works:**
```
Full data
  → weak model₁ → finds errors
  → weak model₂ → focuses on those errors → finds remaining errors
  → weak model₃ → focuses on those errors
  → weighted combination of all models → final prediction
```

- Models are **NOT independent** — each model depends on the errors of the last
- Misclassified samples get higher weights so next model pays more attention to them
- Final prediction = weighted sum (better models get higher weight)

**Reduces:** Bias (fixes underfitting)

**Examples:**

**AdaBoost** (Adaptive Boosting)
- Updates sample weights after each round
- Misclassified samples get higher weight

**XGBoost / Gradient Boosting**
- Each new model fits the **residual errors** (gradient of loss) of the previous model
- Much faster and more powerful than AdaBoost

```python
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

**Drawback:**
- Sequential → cannot parallelize easily
- Prone to overfitting if too many estimators or too high learning rate
- Sensitive to noisy data and outliers (keeps focusing on them)

---

## 3. Stacking (Stacked Generalization)

**Idea:** Train multiple **different** base models, then train a **meta-learner** on top of their predictions.

**How it works:**
```
Full data
  → KNN        → prediction_knn   ─┐
  → SVM        → prediction_svm   ─┤→ meta-learner (e.g. Logistic Regression) → final output
  → Naive Bayes→ prediction_nb    ─┘
```

- Base models can be **completely different algorithms**
- Their predictions become the **new feature set** for the meta-learner
- Meta-learner learns how to best combine them

**Reduces:** Both variance and bias

```python
from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

base_models = [KNeighborsClassifier(), SVC(probability=True)]
meta_model  = LogisticRegression()

stack = StackingClassifier(classifiers=base_models, meta_classifier=meta_model)
stack.fit(X_train, y_train)
print(stack.score(X_test, y_test))
```

**Drawback:**
- Most complex to implement and tune
- Risk of data leakage if base models and meta-learner are trained on the same data
  - Fix: use cross-val predictions for meta-learner training (out-of-fold predictions)
- High compute cost

---

## Quick Comparison Table

| Feature | Bagging | Boosting | Stacking |
|---|---|---|---|
| Training order | Parallel | Sequential | Parallel (base) + 1 more |
| Base models | Same algorithm | Same algorithm | Different algorithms |
| Focuses on | Variance reduction | Bias reduction | Both |
| Combination | Vote / average | Weighted sum | Meta-learner |
| Risk | Slow on huge data | Overfitting | Data leakage |
| Example | Random Forest | XGBoost, AdaBoost | mlxtend StackingClassifier |

---

## What to Say in an Interview

> "Ensemble learning combines multiple models to get better performance than any single model.
> There are three main types.
> Bagging trains models in parallel on random data subsets and averages their output —
> it reduces variance. Random Forest is the classic example.
> Boosting trains models sequentially where each one corrects the previous one's errors —
> it reduces bias. XGBoost is the go-to here.
> Stacking trains different base models and feeds their predictions into a meta-learner —
> it can reduce both bias and variance but is the most complex.
> In practice I'd start with Random Forest as a baseline, then try XGBoost for better performance,
> and only use stacking if I need to squeeze out the last few percentage points and have time to tune it carefully."

---

## One-Line Mental Model

> Bagging = many same models in parallel → average out noise
> Boosting = many same models in sequence → fix each other's mistakes
> Stacking = many different models → let a smarter model decide who to trust

---
