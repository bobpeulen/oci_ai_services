# OCI AI Servies
Demos and example notebooks in using the different OCI AI Services, running from OCI Data Science

## Examples
- AI Anomaly Detection (OCI_anomaly_detection.ipynb, demo-test-data.csv, demo-training-data.csv).
Small example of loading a .csv file, authentication, and invoking a pre-built OCI Anomaly Detection model. First, create an Anomaly Detection model using the Oracle console or build one using the REST APIs. Use the test .csv to invoke the model. Note. In the test .csv are no anomalies yet. You can manually change some of the values to mimick anomalies.

- AI Language (twitter_feed_sentiment.ipynb).
Twitter batch processing of multiple accounts, looping through user and reply_to information, key phrases, and sentiment analysis using AI Language. Results pushed to autonomous database

- AI Document Understanding (The 2 pdf files as input, credit scoring.ipynb)
Analyzing PDF files using OCI Data Science and OCI Document Understanding.
