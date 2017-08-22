import sys
from sklearn.metrics import accuracy_score, confusion_matrix

if len(sys.argv) <= 1:
    print("Usage:", sys.argv[0], " <file.txt>")
    exit(0)

y_true = []
y_pred = []
with open(sys.argv[1]) as f:
    for line in f:
        if line.startswith("#"):
            continue
        [fname, actual, pred] = line.split(' ')
        y_true.append(int(actual))
        y_pred.append(int(pred))

print("accuracy: ", accuracy_score(y_true, y_pred))
print("confusion matrix: ")
print(confusion_matrix(y_true, y_pred))
