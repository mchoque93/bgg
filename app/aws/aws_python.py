import logging
import boto3
from botocore.exceptions import ClientError
if __name__ == '__main__':

    AWS_ACCESS_KEY_ID = "AKIATV2IMV6APNDKCZFF"
    AWS_SECRET_ACCES_KEY = "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"

    s3_client = boto3.client('s3', region_name="us-west-2",
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCES_KEY)
    location={'LocationConstraint': 'us-west-2'}

    ##CREAR BUCKET
    #s3_client.create_bucket(Bucket= 'bbg-bucket',
    #                        CreateBucketConfiguration=location)

    response = s3_client.list_buckets()
    for bucket in response['Buckets']:
        print(bucket['Name'])

    file_name ="/home/marta/PycharmProjects/Boardgames/lr.joblib"
    bucket_name= "bbg-bucket"
    object_name = "lr.joblib"
    response = s3_client.upload_file(file_name, bucket_name, object_name)

    ##DESCARGAR ARCHIVO
    #s3_client.download_file(bucket_name, "lr.joblib", "lr.joblib")

