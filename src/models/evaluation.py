
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

def evaluate():

    """
        Dump all evaluation metrics and plots for given datasets.
    """

    x_test = np.load('data/processed/x_test.npy')
    y_test = np.load('data/processed/y_test.npy')

    reconstructed_model = load_model("models/model.keras")
    y_pred = reconstructed_model.predict(x_test, batch_size=1000)
    print(y_pred)

    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test=y_test.reshape(-1,1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)
    accuracy = accuracy_score(y_test,y_pred_binary)
    
    # Save the evaluation metrics to files
    with open('eval/classification_report.txt', 'w') as f:
        f.write(report)

    np.savetxt('eval/confusion_matrix.csv', confusion_mat, delimiter=',')

    with open('eval/accuracy.txt', 'w') as f:
        f.write(str(accuracy))

if __name__ == '__main__':
    evaluate()