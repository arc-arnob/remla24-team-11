
from keras.models import load_model
import numpy as np
from dvclive import Live
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, roc_auc_score, f1_score

def evaluate(model, data, labels, live):

    """
        Dump all evaluation metrics and plots for given datasets.
    """

    y_pred = (model.predict(data, batch_size=1000)).astype(float)
    print(y_pred)

    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = labels.reshape(-1,1)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)

    accuracy = accuracy_score(y_test,y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred_binary)

    if not live.summary:
        live.summary = {"accuracy": {}, "roc_auc": {}, "f1": {}}
    live.summary["accuracy"] = accuracy
    live.summary["roc_auc"] = roc_auc
    live.summary["f1"] = f1

    live.log_sklearn_plot(
        "confusion_matrix",
        y_test,
        y_pred_binary,
        name=f"cm",
    )

    live.log_sklearn_plot(
        "precision_recall",
        y_test,
        y_pred,
        name=f"prc",
    )

if __name__ == '__main__':
    EVAL_PATH = "eval"


    x_test = np.load('data/processed/x_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    model = load_model("models/model.keras")

    with Live(EVAL_PATH) as live:
        evaluate(model, x_test, y_test, live)