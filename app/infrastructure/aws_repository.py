import boto3
import joblib

from app.infrastructure.abstract_repository import AbstractRepository

AWS_ACCESS_KEY_ID = "AKIATV2IMV6APNDKCZFF"
AWS_SECRET_ACCES_KEY = "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"

s3_client = boto3.client('s3', region_name="us-west-2",
                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCES_KEY)
location = {'LocationConstraint': 'us-west-2'}

bucket_name = "bbg-bucket"


class AWSRepository(AbstractRepository):
    def load_model_lr(self):
        s3_client.download_file(bucket_name, "lr.joblib", "lr.joblib")
        loaded_model = joblib.load("lr.joblib")
        return loaded_model
