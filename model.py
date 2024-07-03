import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
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

# Train an Isolation Forest model for each feature and serialize them
for feature in features:
    model = IsolationForest(contamination=0.01)  # Adjust contamination rate as needed
    model.fit(data[[feature]])

    # Serialize the model
    model_filename = f'{feature}_isolation_forest_model.pkl'
    joblib.dump(model, model_filename)

    aws_access_key_id = "my_key"
    aws_secret_access_key = "my_access_key"
    aws_region = 'us-east-1'
    # Upload the serialized model to S3
    s3 = boto3.client('s3',
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      region_name=aws_region)
    bucket_name = 'telemetry-model'  # Change this to your S3 bucket name
    s3.upload_file(model_filename, bucket_name, model_filename)

    print(f'Model for {feature} saved to S3 bucket {bucket_name} as {model_filename}')