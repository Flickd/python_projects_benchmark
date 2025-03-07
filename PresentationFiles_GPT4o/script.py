import os

import boto3


def upload_to_s3(file_path, bucket_name, object_name=None):
    """
    Upload a file to an S3 bucket.

    Parameters:
    file_path (str): The local path to the file to be uploaded.
    bucket_name (str): The name of the S3 bucket where the file will be stored.
    object_name (str, optional): The name of the object in the S3 bucket.
                                 If not specified, the file's basename will be used.

    Returns:
    None
    """
    if object_name is None:
        # Use the file's basename as the object name if not provided
        object_name = os.path.basename(file_path)
    s3_client = boto3.client("s3")
    try:
        # Attempt to upload the file to the specified S3 bucket
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"File '{file_path}' uploaded to '{bucket_name}/{object_name}'")
    except Exception as e:
        # Print an error message if the upload fails
        print(f"Error uploading file: {e}")


if __name__ == "__main__":
    # Example usage: specify the file to upload and the target S3 bucket
    file_to_upload = "your_file.txt"
    bucket = "your-bucket-name"
    upload_to_s3(file_to_upload, bucket)
