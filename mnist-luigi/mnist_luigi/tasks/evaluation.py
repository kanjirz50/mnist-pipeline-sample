import joblib
import luigi
from luigi.util import requires
from sklearn import metrics

from .training import TrainingTask
from .preprocessing import DataPreprocessingTask


@requires(DataPreprocessingTask, TrainingTask)
class PredictionTask(luigi.Task):
    def output(self):
        return luigi.LocalTarget(
            "data/mnist_predicted.pickle",
            format=luigi.format.Nop
        )

    def run(self):
        preprocessed_input, trained_input = self.input()

        with preprocessed_input.open("rb") as fin:
            dataset = joblib.load(fin)
    
        with trained_input.open("rb") as fin:
            clf = joblib.load(fin)

        dataset["y_pred"] = clf.predict(dataset["X_test"])

        with self.output().open("wb") as fout:
            joblib.dump(dataset, fout)


@requires(PredictionTask)
class ClassificationReportTask(luigi.Task):
    def output(self):
        return luigi.LocalTarget(
            "data/classification_report.txt"
        )

    def run(self):
        with self.input().open("rb") as fin:
            dataset = joblib.load(fin)
    
        y_test = dataset["y_test"]
        predicted = dataset["y_pred"]

        output = f"{ metrics.classification_report(y_test, predicted) }\n"

        with self.output().open("wt") as fout:
            fout.write(output)
