import joblib
import luigi
from sklearn.datasets import load_digits


class GetMNISTDatasetTask(luigi.Task):
    def output(self):
        return luigi.LocalTarget(
            "data/mnist.pickle",
            format=luigi.format.Nop
        )

    def run(self):
        digits = load_digits()

        with self.output().open("wb") as fout:
           joblib.dump(digits, fout)
