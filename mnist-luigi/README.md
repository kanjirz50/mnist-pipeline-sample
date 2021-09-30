# mnist-luigi

```sh
poetry install

luigi --module mnist_luigi.tasks.evaluation ClassificationReportTask --local-scheduler

cat data/classification_report.txt
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        83
           1       0.98      0.99      0.98        93
           2       1.00      1.00      1.00        85
           3       0.99      0.99      0.99        98
           4       1.00      1.00      1.00       100
           5       0.98      0.98      0.98        81
           6       0.99      1.00      0.99        88
           7       0.99      0.99      0.99        89
           8       0.98      0.99      0.98        81
           9       0.98      0.95      0.96       101

    accuracy                           0.99       899
   macro avg       0.99      0.99      0.99       899
weighted avg       0.99      0.99      0.99       899
```