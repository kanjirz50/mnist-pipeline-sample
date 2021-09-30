import joblib
import luigi
from luigi.util import requires
from sklearn import svm

from .preprocessing import DataPreprocessingTask


@requires(DataPreprocessingTask)
class TrainingTask(luigi.Task):
    def output(self):
        return luigi.LocalTarget(
            "data/mnist_model.pickle",
            format=luigi.format.Nop
        )

    def run(self):
        with self.input().open("rb") as fin:
            dataset = joblib.load(fin)

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]

        clf = svm.SVC(gamma=0.001)
        clf.fit(X_train, y_train)

        with self.output().open("wb") as fout:
            joblib.dump(clf, fout)
