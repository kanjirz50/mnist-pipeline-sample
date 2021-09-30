import joblib
import luigi
from luigi.util import requires
from sklearn.model_selection import train_test_split
from .data_fetcher import GetMNISTDatasetTask


@requires(GetMNISTDatasetTask)
class DataPreprocessingTask(luigi.Task):
    random_state = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(
            "data/mnist_preprocessed.pickle",
            format=luigi.format.Nop
        )

    def run(self):
        with self.input().open("rb") as fin:
            digits = joblib.load(fin)

        X_train, X_test, y_train, y_test = train_test_split(
            digits.images.reshape((len(digits.images), -1)),
            digits.target,
            test_size=0.5,
            shuffle=True,
            random_state=self.random_state
        )

        with self.output().open("wb") as fout:
            dataset = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test
            }
            joblib.dump(dataset, fout)
