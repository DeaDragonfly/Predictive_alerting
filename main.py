import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (precision_score,recall_score,f1_score,average_precision_score,roc_auc_score,confusion_matrix)


def generate_queue_data(n_steps=6000, seed=42):
    rng = np.random.default_rng(seed)

    #metrics
    queue_length = np.zeros(n_steps)
    processing_time = np.zeros(n_steps)
    arrival_rate = np.zeros(n_steps)
    timeout_rate = np.zeros(n_steps)
    incident = np.zeros(n_steps, dtype=int)

    #basic parameters
    base_capacity = 20.0
    queue_length[0] = 8

    overload_signal = np.zeros(n_steps)

    n_events = 20
    starts = rng.choice(np.arange(100, n_steps - 150), size=n_events, replace=False) #generate random start moments of overload
    starts.sort()

    #generate forms of overloads
    for start in starts:
        ramp_len = rng.integers(15, 35) #how many steps does the load increase
        peak_len = rng.integers(15, 30) #how long does the peak last
        decay_len = rng.integers(20, 40) #how long has the decline been going on
        strength = rng.uniform(4, 8) #how strong is the overload

        #the boundaries of the event phases are calculated:
        ramp_end = min(start + ramp_len, n_steps) 
        peak_end = min(ramp_end + peak_len, n_steps)
        decay_end = min(peak_end + decay_len, n_steps)

        #an overload profile is built:
        overload_signal[start:ramp_end] += np.linspace(0, strength, ramp_end - start)
        overload_signal[ramp_end:peak_end] += strength
        overload_signal[peak_end:decay_end] += np.linspace(strength, 0, decay_end - peak_end)

    #for each moment in time all metrics are calculated
    for t in range(n_steps): 
        prev_q = queue_length[t - 1] if t > 0 else queue_length[0]

        #incoming load
        arrival = 16 + 1.5 * np.sin(t / 60) + rng.normal(0, 1.0) + overload_signal[t]
        arrival = max(arrival, 0.1) #ensures that the load does not become negative
        arrival_rate[t] = arrival

        #processing time grows with queue, but moderately
        proc_time = 1.0 + 0.005 * prev_q + rng.normal(0, 0.03)
        proc_time = np.clip(proc_time, 0.7, 2.5) #limits the values ​​from below and above
        processing_time[t] = proc_time

        #capacity degrades mildly instead of collapsing
        service_capacity = base_capacity - 2.0 * (proc_time - 1.0) + rng.normal(0, 0.5)
        service_capacity = max(service_capacity, 8.0)

        #queue updating
        new_q = max(prev_q + arrival - service_capacity, 0)

        #recovery term: large queues get partially drained
        if prev_q > 25:
            new_q = max(new_q - 0.15 * (prev_q - 25), 0)

        queue_length[t] = new_q

        #timeout rate
        timeout = 0.001 * max(new_q - 20, 0) + 0.02 * max(proc_time - 1.2, 0) + rng.normal(0, 0.002)
        timeout_rate[t] = np.clip(timeout, 0, 1)

        #incident rule
        if (new_q > 30 and timeout_rate[t] > 0.02) or (new_q > 40):
            incident[t] = 1

    return pd.DataFrame({
        "queue_length": queue_length,
        "processing_time": processing_time,
        "arrival_rate": arrival_rate,
        "timeout_rate": timeout_rate,
        "incident": incident })

#the function creates a target variable for predictive alerting
def build_future_target(incident_series, horizon):
    incident = incident_series.to_numpy()
    y = np.zeros(len(incident), dtype=int)
    #for each moment in time t: look ahead H steps. if there is at least one incident, set it to 1; otherwise, set it to 0.
    for t in range(len(incident)):
        future_end = min(len(incident), t + 1 + horizon)
        future = incident[t + 1:future_end]
        y[t] = 1 if future.sum() > 0 else 0

    return y

#the function takes one history window of length W and turns it into a set of numerical features
def extract_window_features(window_df):
    feats = {}
    #for each metric, features are calculated
    for col in window_df.columns:
        values = window_df[col].to_numpy()

        feats[f"{col}_last"] = values[-1]
        feats[f"{col}_mean"] = values.mean()
        feats[f"{col}_std"] = values.std()
        feats[f"{col}_min"] = values.min()
        feats[f"{col}_max"] = values.max()
        feats[f"{col}_trend"] = values[-1] - values[0]

        #short-term dynamics
        if len(values) >= 3:
            feats[f"{col}_diff1"] = values[-1] - values[-2] #what happened in the last step
            feats[f"{col}_diff2"] = values[-2] - values[-3] #the step before
        else: #if the window is too short
            feats[f"{col}_diff1"] = 0.0
            feats[f"{col}_diff2"] = 0.0

    return feats

#main dataset for training
def make_dataset(df, feature_cols, window, horizon):
    target = build_future_target(df["incident"], horizon)
    X_rows = [] #feature strings
    Y_rows = [] #corresponding labels
    time_idx = [] #the time to which they belong
    for t in range(window - 1, len(df) - horizon): #passage through all possible windows
        window_df = df.iloc[t - window + 1:t + 1][feature_cols] #selecting the current window
        #features are extracted from the window, a target label is attached to them, a point in time is stored
        X_rows.append(extract_window_features(window_df))
        Y_rows.append(target[t])
        time_idx.append(t)
    #Transformation into final objects
    x = pd.DataFrame(X_rows)
    y = np.array(Y_rows)
    time_idx = np.array(time_idx)

    return x, y, time_idx

def train_model(X_train, Y_train):
    weights = compute_sample_weight(class_weight="balanced", y=Y_train) #positive examples are given more weight
    #creating a model:
    model = HistGradientBoostingClassifier(
        max_depth=5, #how complex can trees be
        learning_rate=0.05,
        max_iter=200, #number of boosting iterations
        random_state=42
    )
    model.fit(X_train, Y_train, sample_weight=weights) #the model learns to predict a binary label based on features
    return model

#the model does not immediately produce 0/1, but a probability
def find_best_threshold(y_true, y_prob): 
    thresholds = np.linspace(0.1, 0.9, 17) #enumeration of thresholds (thresholds from 0.1 to 0.9 are checked)

    #starting values ​​for finding the best f1
    best_thr = 0.5
    best_f1 = -1
    #for each threshold: the probabilities are converted to 0/1, F1 is calculated, if it has improved, the threshold is maintained
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1

#calculates quality metrics
def evaluate(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int) #turning probabilities into classes
    #checking for one-class case
    if len(np.unique(y_true)) < 2:
        roc_auc = None
    else:
        roc_auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }

#the function divides the data by time
def time_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)

    #length check, to ensure that there are the same number of features and labels
    assert len(X) == len(y), "X and y must have same length"

    #calculating boundaries
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X.iloc[:train_end].copy()
    y_train = y[:train_end]

    X_val = X.iloc[train_end:val_end].copy()
    y_val = y[train_end:val_end]

    X_test = X.iloc[val_end:].copy()
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

#baseline without ML
def rule_based_predict(X, threshold_q=28, threshold_to=0.015):
        return ((X["queue_length_last"] > threshold_q) & 
            (X["timeout_rate_last"] > threshold_to)).astype(int)

def evaluate_binary_predictions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist()}

def print_metrics(title, metrics):
    print(title)
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print("  confusion_matrix:")
            print(f"    {value[0]}")
            print(f"    {value[1]}")
        elif value is None:
            print(f"  {key}: None")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()

def print_class_distribution(y, name):
    positives = int(y.sum())
    total = len(y)
    negatives = total - positives
    ratio = positives / total if total > 0 else 0

    print(f"{name}")
    print(f"  total samples : {total}")
    print(f"  positive class: {positives}")
    print(f"  negative class: {negatives}")
    print(f"  positive ratio: {ratio:.2%}")
    print()

def main():
    #hyperparameters
    WINDOW = 30
    HORIZON = 10
    #metrics that are used as input to the model
    FEATURE_COLS = [
        "queue_length",
        "processing_time",
        "arrival_rate",
        "timeout_rate"
    ]

    df = generate_queue_data(n_steps=6000, seed=42)
    #printing general statistics
    print("=" * 60)
    print("RAW DATA SUMMARY")
    print("=" * 60)
    print(f"Raw incident count : {int(df['incident'].sum())}")
    print(f"Raw incident ratio : {df['incident'].mean():.2%}")
    print()
    #a training set is constructed from raw time series: X are the features from the windows, y are the target labels
    X, y, _ = make_dataset(
        df=df,
        feature_cols=FEATURE_COLS,
        window=WINDOW,
        horizon=HORIZON
    )

    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X, y)

    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print_class_distribution(y_train, "Train")
    print_class_distribution(y_val, "Validation")
    print_class_distribution(y_test, "Test")

    model = train_model(X_train, y_train)
    #selection of threshold by validation
    val_prob = model.predict_proba(X_val)[:, 1]
    best_thr, best_val_f1 = find_best_threshold(y_val, val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    ml_results = evaluate(y_test, test_prob, best_thr)

    print(f"Validation F1: {best_val_f1:.4f}")
    #the metrics of a simple rule are calculated per test. this provides a comparison: rule-based and ML
    rule_pred = rule_based_predict(X_test)
    rule_results = evaluate_binary_predictions(y_test, rule_pred)
    print("=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    print(f"Best validation threshold: {best_thr:.2f}")
    print(f"Validation F1          : {best_val_f1:.4f}")
    print()

    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print_metrics("Rule-based baseline", rule_results)
    print_metrics("Gradient Boosting model", ml_results)

if __name__ == "__main__":
    main()
