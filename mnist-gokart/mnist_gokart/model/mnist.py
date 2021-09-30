from logging import getLogger

import luigi

from sklearn import metrics, svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mnist_gokart.utils.template import GokartTask

logger = getLogger(__name__)


class GetMNISTDatasetTask(GokartTask):
    def run(self):
        digits = load_digits()
        self.dump(digits)


class PreprocessingTask(GokartTask):
    random_state = luigi.IntParameter(default=42)

    def requires(self):
        return GetMNISTDatasetTask()

    def run(self):
        digits = self.load()

        X_train, X_test, y_train, y_test = train_test_split(
            digits.images.reshape((len(digits.images), -1)),
            digits.target,
            test_size=0.5,
            shuffle=True,
            random_state=self.random_state
        )

        dataset = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

        self.dump(dataset)


class TrainingTask(GokartTask):
    def requires(self):
        return PreprocessingTask()

    def run(self):
        dataset = self.load()

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]

        clf = svm.SVC(gamma=0.001)
        clf.fit(X_train, y_train)

        self.dump(clf)


class PredictionTask(GokartTask):
    def requires(self):
        return {
            "dataset": PreprocessingTask(),
            "model": TrainingTask()
        }

    def run(self):
        dataset = self.load('dataset')
        clf = self.load('model')

        dataset['y_pred'] = clf.predict(dataset['X_test'])

        self.dump(dataset)


class ClassificationReportTask(GokartTask):
    def requires(self):
        return PredictionTask()

    def output(self):
        return self.make_target('classification_report.txt')

    def run(self):
        dataset = self.load()

        y_test = dataset["y_test"]
        predicted = dataset["y_pred"]

        output = f"{ metrics.classification_report(y_test, predicted) }\n"

        self.dump(output)
