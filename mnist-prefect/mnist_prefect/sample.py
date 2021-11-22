import prefect
from joblib import dump
from prefect import task, Flow, Parameter
from prefect.engine.results import LocalResult
from sklearn import metrics, svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@task(target="{task_name}.pickle")
def load_digits_data():
    return load_digits()


@task
def preprocesing_task(digits_data, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        digits_data.images.reshape((len(digits_data.images), -1)),
        digits_data.target,
        test_size=0.5,
        shuffle=True,
        random_state=random_state
    )

    dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    return dataset


@task
def training_task(dataset):
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]

    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)

    return clf


@task
def prediction_task(dataset, clf):
    dataset['y_pred'] = clf.predict(dataset['X_test'])
    return dataset


@task
def classification_report_task(predicted_dataset):
    y_test = predicted_dataset["y_test"]
    predicted = predicted_dataset["y_pred"]

    return f"{ metrics.classification_report(y_test, predicted) }\n"


@task
def save_model_task(model):
    dump(model, 'output/model.joblib')


@task
def save_classification_report_task(classification_report):
    with open("output/classification_report.txt", "wt") as fout:
        fout.write(classification_report)


def mnist_experiment_flow():
    with Flow("hello-flow", result=LocalResult(dir="resources")) as flow:
        random_state = Parameter("random_state", default=42)

        digits_data = load_digits_data()
        dataset = preprocesing_task(digits_data, random_state)
        clf = training_task(dataset)
        save_model_task(clf)
        predicted_dataset = prediction_task(dataset, clf)
        classification_report = classification_report_task(predicted_dataset)
        save_classification_report_task(classification_report)

    return flow


if __name__ == "__main__":
    flow = mnist_experiment_flow()
    flow.run(random_state=42)
