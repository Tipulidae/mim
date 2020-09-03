import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


def parse_json(file):
    with open(file) as fp:
        data = [json.loads(line) for line in fp.readlines()]

    other_features = ['age', 'tnts']
    ordinal_features = [
        "prev5y-AMI",
        "prev5y-COPD",
        "prev5y-Diabetes",
        "prev5y-Heartfail",
        "prev5y-Hypertens",
        "prev5y-Renal",
        "prev5y-PAD",
        "prev5y-UA",
        "prev5y-CABG",
        "prev5y-PCI",
        "Charlson-AcuteMyocardialInfarction",
        "Charlson-CongestiveHeartFailure",
        "Charlson-PeripheralVascularDisease",
        "Charlson-CerebralVascularAccident",
        "Charlson-Dementia",
        "Charlson-PulmonaryDisease",
        "Charlson-ConnectiveTissueDisorder",
        "Charlson-PepticUlcer",
        "Charlson-LiverDisease",
        "Charlson-Diabetes",
        "Charlson-DiabetesComplications",
        "Charlson-Parapelgia",
        "Charlson-RenalDisease",
        "Charlson-Cancer",
        "Charlson-MetstaticCancer",
        "Charlson-SevereLiverDisease",
        "Charlson-HIV",
        "gender",
        "label-index-mi"
    ]
    df = pd.DataFrame.from_records(
        data,
        columns=other_features + ordinal_features)
    tnt_features = extract_tnt_features(df.tnts)

    df[ordinal_features] = OrdinalEncoder().fit_transform(df[ordinal_features])
    df = pd.concat([df, tnt_features], axis=1)
    df = df.rename(columns={'label-index-mi': 'y'})
    X = df.drop(columns=['y', 'tnts'])
    y = df.y

    return X, y


def extract_tnt_features(tnts):
    tnts = pd.DataFrame.from_records(tnts, columns=['tnt1', 'tnt2'])
    tnt1 = pd.DataFrame.from_records(tnts.tnt1, columns=['t', 'tnt'])
    tnt2 = pd.DataFrame.from_records(tnts.tnt2, columns=['t', 'tnt'])
    dt = (pd.to_datetime(tnt2.t) - pd.to_datetime(
        tnt1.t)).dt.total_seconds() // 60
    return pd.DataFrame(
        zip(tnt1.tnt, tnt2.tnt, (tnt2.tnt - tnt1.tnt) / dt),
        columns=['tnt', 'tnt_repeat', 'tnt_diff']
    )


def classify(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(X_train, y_train)
    y_proba_rf = rf.predict_proba(X_test)
    print(f"Random Forest AUC ROC: {roc_auc_score(y_test, y_proba_rf[:, 1])}")

    gb = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=2,
        min_samples_leaf=7,
        subsample=0.5
    )
    gb.fit(X_train, y_train)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    print(
        f"Gradient Boosting AUC ROC: {roc_auc_score(y_test, y_proba_gb)}")
