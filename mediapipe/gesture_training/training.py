"""Train and evaluate a classifier using session-separated landmark samples."""

from collections import Counter
from datetime import datetime, timezone
import json

from .config import FEATURE_VERSION, gesture_names
from .features import features


def train(args):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit

    labels = gesture_names()
    records = []
    with args.data.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in labels or not row.get("session"):
                raise ValueError(f"Invalid label/session on line {line_number}")
            features(row["landmarks"])
            records.append(row)
    counts = Counter(row["label"] for row in records)
    sessions = {
        label: {row["session"] for row in records if row["label"] == label}
        for label in labels
    }
    for label in labels:
        if counts[label] < 20 or len(sessions[label]) < 2:
            raise ValueError(
                f"{label}: need >=20 samples across >=2 separate capture runs; "
                f"found {counts[label]} samples / {len(sessions[label])} sessions"
            )
    x = np.asarray([features(row["landmarks"]) for row in records])
    y = np.asarray([row["label"] for row in records])
    groups = np.asarray([row["session"] for row in records])
    # Never randomly split adjacent frames from a single capture session.
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.25, random_state=42)
    selected = None
    for training, testing in splitter.split(x, y, groups):
        if all(
            (y[training] == label).sum() >= 10
            and (y[testing] == label).sum() >= 5
            for label in labels
        ):
            selected = training, testing
            break
    if selected is None:
        raise ValueError(
            "No session-separated split covers all five classes sufficiently. "
            "Record every gesture in more independent capture runs."
        )
    training, testing = selected
    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(x[training], y[training])
    predicted = model.predict(x[testing])
    report = classification_report(
        y[testing], predicted, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y[testing], predicted, labels=labels)
    print("Held-out sessions (not used for fitting):")
    print(report)
    print("Confusion matrix, rows=true / columns=predicted:", labels)
    print(matrix)
    args.classifier.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "labels": labels, "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_sessions": sorted(set(groups[training])),
        "test_sessions": sorted(set(groups[testing])),
    }, args.classifier)
    args.classifier.with_suffix(".metrics.json").write_text(json.dumps({
        "labels": labels, "report": report, "confusion_matrix": matrix.tolist(),
        "train_samples": len(training), "test_samples": len(testing),
        "train_sessions": sorted(set(groups[training])),
        "test_sessions": sorted(set(groups[testing])),
    }, indent=2))
    print(f"Saved {args.classifier} (standalone classifier, not a .task file)")
