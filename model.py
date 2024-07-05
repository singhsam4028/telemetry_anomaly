import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn2pmml import PMMLPipeline, sklearn2pmml
import boto3

# Load the dataset
file_path = 'iot_telemetry_data.csv'  # Change this to the path of your CSV file
data = pd.read_csv(file_path)

# Preprocess the data (Convert relevant columns to appropriate types)
data['ts'] = pd.to_datetime(data['ts'], unit='s')
data['light'] = data['light'].astype(bool)
data['motion'] = data['motion'].astype(bool)

# Select relevant features
features = ['temp', 'co', 'humidity', 'lpg', 'smoke']

aws_access_key_id = "my_key"
aws_secret_access_key = "my_access_key"
aws_region = 'us-east-1'
bucket_name = 'telemetry-model'  # Change this to your S3 bucket name

for feature in features:
    model = IsolationForest(contamination=0.01)  # Adjust contamination rate as needed
    pipeline = PMMLPipeline([("estimator", model)])
    pipeline.fit(data[[feature]])

    # Serialize the model to PMML
    pmml_filename = f'{feature}_isolation_forest_model.pmml'
    sklearn2pmml(pipeline, pmml_filename)

    # Upload the PMML model to S3
    s3 = boto3.client('s3',
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      region_name=aws_region)
    s3.upload_file(pmml_filename, bucket_name, pmml_filename)

    print(f'Model for {feature} saved to S3 bucket {bucket_name} as {pmml_filename}')
