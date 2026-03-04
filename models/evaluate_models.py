import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    results = {
        "Model": model_name,

        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Train Precision": precision_score(y_train, train_pred),
        "Train Recall": recall_score(y_train, train_pred),
        "Train F1": f1_score(y_train, train_pred),

        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Test Precision": precision_score(y_test, test_pred),
        "Test Recall": recall_score(y_test, test_pred),
        "Test F1": f1_score(y_test, test_pred)
    }

    return results


def compare_models(models, X_train, X_test, y_train, y_test):

    rows = []

    for name, model in models.items():

        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)

        rows.append(result)

    comparison_df = pd.DataFrame(rows)

    return comparison_df
