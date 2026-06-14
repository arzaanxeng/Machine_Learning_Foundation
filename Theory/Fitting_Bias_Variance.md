# Overfitting, Underfitting, Bias and Variance

> Notes by Arzaan 

---

## What is a Model Actually Doing?

When you train a model, it tries to learn a pattern from training data
so it can predict correctly on new unseen data.

The goal is to learn the TRUE pattern — not too simple, not too complex.

---

## Underfitting

Model is TOO SIMPLE. Fails to learn even the basic pattern.

Real life analogy:
Studying for an exam by only reading chapter headings.
You fail on both practice questions AND the real exam.

```
Data points:        o  o     o  o
                         o o
True pattern:       a curved line

Underfitted model:  _______________ (flat line, too simple, misses everything)
```

- Does badly on BOTH training data and test data
- High Bias = model has a wrong/too simple assumption about data

When does it happen:
- Model is too simple (Linear Regression on curved data)
- Too few features
- Trained for too few iterations

---

## Overfitting

Model is TOO COMPLEX. Memorizes training data including noise.

Real life analogy:
Memorizing every practice paper question word by word.
Score 100% on practice paper. Fail on real exam (questions slightly different).

```
Data points:        o  o     o  o
                         o o
True pattern:       a smooth curved line

Overfitted model:   a crazy wiggly line passing through EVERY point
```

- Does very well on training data, badly on test data
- High Variance = model is too sensitive, picks up noise as real pattern

When does it happen:
- Model is too complex (Decision Tree with no depth limit)
- Too many features, too little data
- Trained for too many iterations

---

## Key Clarification (Common Confusion)

Your educator says  "100% accuracy = overfit"   → talking about TRAINING data
I say "overfit = low accuracy"                  → talking about TEST data

BOTH are correct. Same model, different datasets.

```
Overfit model:
  Training accuracy  →  100%   (memorized everything)
  Test accuracy      →  60%    (fails on new data)
```

Always ask: "accuracy on WHAT?" — training data or test data?

---

## Bias and Variance

### Bias
How wrong the model's assumption is about the data.
Model has already decided "I will fit a straight line no matter what."
That pre-decided stubbornness = bias.

High bias → too simple → underfitting

### Variance
How much the model changes when you change the training data.
Train on slightly different data → model completely changes → high variance.

High variance → too sensitive → overfitting

---

## The Archer Analogy

Imagine shooting arrows at a target. Center = correct answer.

```
                HIGH BIAS              LOW BIAS
                (wrong aim)            (correct aim)

LOW VARIANCE    x  x                      x
(consistent)     x x        vs           x x
                (all wrong              (all correct
                 same spot)              same spot)
                = UNDERFIT              = PERFECT ✓

HIGH VARIANCE   x    x               x       x
(scattered)   x        x           x    x     x
               (scattered           (scattered around
                all wrong)           center)
               = WORST CASE         = OVERFIT
```

---

## Bias-Variance Tradeoff

```
Error
  |
  |\
  | \          Total Error (U shaped)
  |  \        /
  |   \      /
  |    \    /
  |     \  /
  |      \/   ← sweet spot
  |
  |_____________________
        Model Complexity →

As complexity increases:
  Bias decreases
  Variance increases
  Total error = Bias + Variance → U shape
  Bottom of U = best model
```

---

## Summary Table

| Situation        | Bias | Variance | Train Acc | Test Acc | Problem    |
|------------------|------|----------|-----------|----------|------------|
| Too simple model | High | Low      | Low       | Low      | Underfit   |
| Too complex model| Low  | High     | 100%      | Low      | Overfit    |
| Just right       | Low  | Low      | High      | High     | Perfect ✓  |

---

## How to Detect

| Training Accuracy | Test Accuracy | Problem      |
|-------------------|---------------|--------------|
| Low               | Low           | Underfitting |
| High              | Low           | Overfitting  |
| High              | High (close)  | Just right ✓ |

---

## How to Fix

Underfitting fixes:
- Use a more complex model
- Add more features
- Train longer
- Remove regularization

Overfitting fixes:
- Use a simpler model
- Get more training data
- Use regularization (L1, L2)
- Dropout (in neural networks)
- Use ensemble methods (Bagging, Boosting)

---

## Connection to Ensemble Learning

| Method   | What it does                          | Why it works                          |
|----------|---------------------------------------|---------------------------------------|
| Bagging  | Averages many high variance models    | Variance cancels out, bias stays same |
| Boosting | Chains many high bias models together | Each step reduces bias                |

Bagging example:
  One Decision Tree = low bias but high variance (overfits)
  Average 100 trees = variance drops dramatically
  That is Random Forest

Boosting example:
  One shallow tree = high bias (underfits)
  Add another to fix its mistakes, and another
  Bias keeps reducing with each step
  That is XGBoost

---

## One Line to Remember

Bias   = consistently wrong (wrong assumption, too simple)
Variance = inconsistently right (too sensitive, memorizes noise)
Goal   = low bias AND low variance
```
