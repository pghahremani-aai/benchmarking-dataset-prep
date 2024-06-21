    """
    Description:
        This script generates TSV files for CI segment benchmarking in the ASR analysis pipeline.
        It fetches the labels from the HappyScribe transcriptions and copies the audio files to a new location in GCS.
        The TSV files are then uploaded to GCS for further processing.
        You need to first run python copy_HappyScribed_transcript_to_original_segmented_tables.py to create the new tables with column HappyScribed_transcript.

    Returns:
        _type_: _description_
    """
import csv
import json
import os
from google.cloud import bigquery, storage

# Initialize clients for BigQuery and Google Cloud Storage
bq_client = bigquery.Client()
storage_client = storage.Client()
gcs_upload_path = 'gs://aai-us-central1/datasets/asr_analysis/english_test_benchmarks/'

def upload_to_gcs(local_file_path, destination_blob_path):
    """Uploads a file to the bucket at the specified GCS path."""
    bucket_name = destination_blob_path.split('/')[2]
    destination_blob_name = '/'.join(destination_blob_path.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"File {local_file_path} uploaded to {destination_blob_path}.")

def fetch_label_from_gcs(gcs_uri):
    """Fetches and constructs the label from a JSON file in GCS."""
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    json_data_text = blob.download_as_text(encoding='utf-8-sig')
    json_data = json.loads(json_data_text)
    return ' '.join(word['text'] for entry in json_data for word in entry['words']).strip()

def copy_file_in_gcs(source_bucket_name, source_blob_name, destination_bucket_name, destination_blob_name):
    """Copies a file within GCS from one bucket to another."""
    source_bucket = storage_client.bucket(source_bucket_name)
    source_blob = source_bucket.blob(source_blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)
    source_bucket.copy_blob(source_blob, destination_bucket, new_name=destination_blob_name)
    print(f"Copied from {source_bucket_name}/{source_blob_name} to {destination_bucket_name}/{destination_blob_name}")

#segments = ['email', 'phone_number', 'address', 'alphanumeric', 'website']
segments = ['address']
base_dataset = 'assemblyai-nlp.youtube_scrape_2024_04_18'

# Process each segment
for segment in segments:
    query = f"""
    SELECT id, transcoded_filepath, HappyScribed_transcript
    FROM `{base_dataset}.filtered_{segment}_10hrs_youtube_usm_scraped_dataset_w_Roberta_tags_w_HappyScribe_transcript`
    """
    results = bq_client.query(query).result()
    processed_ids = set()  # Set to store processed IDs

    # Prepare the TSV file
    tsv_filename = f"{segment}_10hrs_youtube_scraped_2024_04_18_dataset_HappyScribe_transcript_final.tsv"
    with open(tsv_filename, 'w', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow(['Index', 'Paths', 'Labels', 'Dataset'])

        for index, row in enumerate(results):
            if row['id'] in processed_ids:
                continue  # Skip if ID has already been processed

            file_name = os.path.basename(row['transcoded_filepath'])
            relative_path = f"{segment}_10hrs_youtube_scraped_2024_04_18_dataset_HappyScribe_transcript_final/{file_name}"

            # Copy the .wav file to the new location
            source_gcs_uri = row['transcoded_filepath']
            destination_gcs_uri = f"gs://aai-us-central1/datasets/asr_analysis/datasets/{relative_path}"
            source_bucket_name, source_blob_name = source_gcs_uri.replace("gs://", "").split("/", 1)
            destination_bucket_name, destination_blob_name = destination_gcs_uri.replace("gs://", "").split("/", 1)
            copy_file_in_gcs(source_bucket_name, source_blob_name, destination_bucket_name, destination_blob_name)

            # Fetch the label from the JSON file in GCS
            if row['HappyScribed_transcript']:
                label = fetch_label_from_gcs(row['HappyScribed_transcript'])
                writer.writerow([index, relative_path, label, f"{segment}_10hrs_youtube_scraped_2024_04_18_dataset_HappyScribe_transcript_final"])
                processed_ids.add(row['id'])  # Add ID to the set of processed IDs


    # Upload TSV file to GCS
    destination_blob_path = f"{gcs_upload_path}{tsv_filename}"
    upload_to_gcs(tsv_filename, destination_blob_path)

print("TSV files generated and files copied for all segments.")