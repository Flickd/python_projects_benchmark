import os

import boto3


def upload_to_s3(file_path, bucket_name, object_name=None):
    """Upload a file to an S3 bucket.

    Args:
        file_path (str): The local path of the file to be uploaded.
        bucket_name (str): The name of the S3 bucket where the file will be stored.
        object_name (str, optional): The name under which the file will be stored in S3. If not provided,
                                     it defaults to the base name of the file.

    Returns:
        None

    Raises:
        Exception: If an error occurs during the upload process.

    Notes:
        Ensure that AWS credentials are configured properly to allow access to the specified bucket.
    """
    if object_name is None:
        object_name = os.path.basename(file_path)

    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"File '{file_path}' uploaded to '{bucket_name}/{object_name}'")
    except Exception as e:
        print(f"Error uploading file: {e}")


if __name__ == "__main__":
    file_to_upload = "your_file.txt"
    bucket = "your-bucket-name"
    upload_to_s3(file_to_upload, bucket)
