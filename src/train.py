from joblib import dump
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from loadData import load_data


def main(repo_path):
    train_csv_path = repo_path / "data/prepared/train.csv"
    train_data, labels = load_data(train_csv_path)

    sgd = SGDClassifier(random_state=91, max_iter=500)
    trained_model = sgd.fit(train_data, labels)

    # rf = RandomForestClassifier(
    #     random_state=91, max_depth=5, n_estimators=10, max_features=1
    # )
    # trained_model = rf.fit(train_data, labels)

    # svc = SVC(random_state=91, gamma=0.001)
    # trained_model = svc.fit(train_data, labels)

    dump(trained_model, repo_path / "model/model.joblib")


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
