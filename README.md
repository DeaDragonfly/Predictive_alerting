# Predictive Alerting for Queue-Based Systems

This project implements a predictive alerting model that forecasts whether an incident will occur within the next **H time steps**, based on the previous **W steps** of system metrics.

The solution follows a **sliding window formulation**, uses a **tree-based machine learning model**, and includes a **rule-based baseline** for comparison.

---

## Problem Formulation

We model a system that processes incoming requests with limited capacity (e.g., API server, task queue, or message broker).

At each time step, we observe:
- `queue_length`
- `processing_time`
- `arrival_rate`
- `timeout_rate`

The goal is:

> Predict whether an **incident will occur within the next H steps**, based on the last W observations.

This is a **binary classification problem**:
- `0` → no incident in the near future
- `1` → incident will occur soon

---

## Dataset

We generate a synthetic time series simulating realistic queue dynamics:

- Incoming load fluctuates over time and includes **overload events**
- Processing slows down as the queue grows
- System capacity degrades under load
- Timeout rate increases when system performance worsens

An **incident** is defined as:
- high queue length + elevated timeout rate  
  OR  
- extremely large queue

This creates labeled incident intervals.

---

## Sliding Window Representation

Instead of feeding raw time series, we use a **window-based feature extraction**:

- Window size: `W = 30`
- Prediction horizon: `H = 10`

Each window (`30 × 4`) is transformed into a feature vector using:
- last value
- mean, std, min, max
- trend (difference between first and last)
- short-term differences (`diff1`, `diff2`)

This results in a tabular dataset suitable for classical ML models.

---

## Model

We use:

- `HistGradientBoostingClassifier` (scikit-learn)
- Class balancing via sample weights
- 200 boosting iterations

This model was chosen because:
- it handles tabular data well
- captures nonlinear interactions
- requires minimal preprocessing

---

## Evaluation Setup

We split data chronologically:
- **Train**: first 70%
- **Validation**: next 15%
- **Test**: final 15%

This avoids data leakage and reflects real-world deployment.

### Threshold Selection

The model outputs probabilities.  
We select the classification threshold using **F1-score on the validation set**.

---

## Metrics

We report:

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Confusion matrix

These metrics reflect:
- alert quality (precision)
- missed incidents (recall)
- overall balance (F1)

---

## Baseline

A simple rule-based baseline is implemented:
```bash
incident = (queue_length > threshold_q) AND (timeout_rate > threshold_to)
```

This simulates traditional threshold-based alerting systems.

---

## Results (example)

| Model                  | Precision | Recall | F1   |
|----------------------|----------|--------|------|
| Rule-based baseline  | 0.97     | 0.76   | 0.85 |
| ML model             | 1.00     | 0.90   | 0.95 |

Key observations:
- ML model significantly improves **recall**
- No increase in false positives
- Better overall balance (higher F1)

---

## Interpretation

The ML model outperforms rule-based alerting because it:
- combines multiple signals (queue, processing, timeouts)
- captures trends and dynamics over time
- detects early warning patterns

---

## Limitations

This is a **synthetic dataset**, which introduces important limitations:

- Relationships between metrics and incidents are relatively clean
- The signal is strong and partially deterministic
- Very high metrics (e.g. ROC-AUC ≈ 1.0) indicate the problem is easier than in reality

In real systems, we would expect:
- noisy measurements
- delayed effects
- incomplete observability
- concept drift

---

## Possible Improvements

- Add noise, delays, or missing data to simulation
- Use raw sequence models (e.g., LSTM, Transformer)
- Incorporate seasonality and external signals
- Optimize threshold for business cost (not just F1)
- Online retraining / drift detection

---

## How to Run

```bash
python main.py
```

---

## Key Takeaways

- Sliding-window formulation is effective for time-series classification
- Simple tree-based models can outperform heuristic rules
- Proper evaluation (time split + threshold tuning) is critical
- Synthetic experiments are useful but must be interpreted carefully
