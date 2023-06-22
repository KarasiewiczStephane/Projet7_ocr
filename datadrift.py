import pandas as pd

from sklearn import datasets

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_preset import DataQualityTestPreset

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

data_trim_sample = pd.read_csv("data_trim_sample.csv",index_col=[0])
compression_opts = dict(method='zip', archive_name='data_trim.csv')
data_trim = pd.read_csv("data_trim.zip",index_col=[0], compression=compression_opts)

data_trim.to_csv('data_trim.zip', index=True, compression=compression_opts)

data_drift_report = Report(metrics=[
    DataDriftPreset(),
])
#Replace reference by data by all and current by samples
data_drift_report.run(current_data=data_trim_sample, reference_data=data_trim, column_mapping=None)
data_drift_report
data_drift_report.save_html("data_drift_report.html")