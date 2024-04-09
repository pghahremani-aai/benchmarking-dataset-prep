"""
This script automates the creation of a curated dataset from an existing BigQuery table, 
which is primarily focused on audio segment analysis with specific attributes such as proper noun density. 
The workflow of the script is as follows:

1. **Filtering**: The script filters the original dataset based on audio duration and the content of 
the human transcript text. It prioritizes entries with higher proper noun density and limits the 
total duration to a predefined threshold. 
The length of the audio segments is constrained to be between 2 minutes and 20 minutes, 
and the data is sorted according to the density of proper nouns.
This step expects the original table to have fields for 'duration', 
'human_transcript_text', and a structured field 'categories.proper_nouns.density' for proper noun density.

2. **Table Creation**: It then creates a new BigQuery table to store these filtered results, 
ensuring that the data is more focused and manageable for further analysis.

3. **TSV Generation and File Uploading**: A TSV file is generated from the new table, 
containing essential details of the filtered records. The script uploads this TSV file 
and the corresponding audio files, referenced in the 'raw_filepath' field of the table, 
to a specified Google Cloud Storage (GCS) location.

4. **Statistics Logging**: Lastly, the script calculates and logs statistical information 
about the new dataset, including mean, variance, standard deviation, and percentiles for 
the density of proper nouns and the audio duration. This provides insights into the dataset’s 
characteristics.

By executing these steps, the script ensures the extraction and organization of relevant 
audio segments based on defined linguistic and duration parameters, aiding in the efficient 
handling of data for downstream processing or analytical tasks. We can use the output tsv file in 
the next steps of the pipeline for evaluation purposes.

"""
from google.cloud import storage, bigquery
import csv
import os
import datetime

def copy_file_in_gcs(source_path, destination_path, destination_bucket_name, source_bucket_name=None):
    storage_client = storage.Client()

    # If source_path contains 'gs://', extract the bucket name and blob path from it
    if source_path.startswith('gs://'):
        parts = source_path[5:].split('/', 1)  # Skip 'gs://' and split
        source_bucket_name = parts[0]
        source_blob_path = parts[1]
    else:
        if not source_bucket_name:
            raise ValueError("source_bucket_name must be provided if source_path does not contain 'gs://'.")
        source_blob_path = source_path

    # Get the source and destination buckets
    source_bucket = storage_client.bucket(source_bucket_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    # Get the source blob
    source_blob = source_bucket.blob(source_blob_path)

    # Copy the blob from the source bucket to the destination bucket
    source_bucket.copy_blob(source_blob, destination_bucket, new_name=destination_path)

def upload_to_gcs(local_path, destination_path, bucket_name):
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(destination_path)
    blob.upload_from_filename(local_path)

def create_filtered_table(source_table_id, filtered_table_name, project_id):
    bq_client = bigquery.Client()
    filter_query = f"""
    WITH SortedFiltered AS (
    SELECT 
        *,
        SUM(duration) OVER (ORDER BY categories.proper_nouns.density DESC) AS running_total
    FROM 
        `{source_table_id}`
    WHERE 
        duration BETWEEN 120 AND 1200 
        AND NOT human_transcript_text LIKE '%[Content_Types].xml%'
        AND NOT REGEXP_CONTAINS(human_transcript_text, r'\\b[A-Z]{2,}\\b')  -- Exclude words in all caps
        AND NOT REGEXP_CONTAINS(human_transcript_text, r'(\\b[A-Z]{2,}\\s+){2,}')  -- Exclude segments with multiple consecutive all-caps words
        AND REGEXP_CONTAINS(human_transcript_text, r'([.!?]\\s+[A-Z])|([.!?]$)')  -- Check for proper sentence structure
        AND NOT REGEXP_CONTAINS(human_transcript_text, r'♪')  -- Exclude lines with musical notes, likely from machine transcription of music
    ),
    LimitedByTime AS (
        SELECT 
            *
        FROM 
            SortedFiltered
        WHERE 
            running_total <= 36000
    )
    SELECT 
        *
    FROM 
        LimitedByTime
    ORDER BY 
        categories.proper_nouns.density DESC
    """

    # Construct the full table name with the timestamp
    destination_table_id = f"{project_id}.{source_table_id.split('.')[1]}.{filtered_table_name}"
    job_config = bigquery.QueryJobConfig(destination=destination_table_id)
    job = bq_client.query(filter_query, job_config=job_config)
    job.result()
    return destination_table_id

def generate_tsv_from_table(query, local_output_file, destination_folder, dataset_name, bucket_name):
    bq_client = bigquery.Client()
    results = bq_client.query(query)

    with open(local_output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['', 'paths', 'labels', 'dataset'])

        for index, row in enumerate(results):
            source_blob_path = row.raw_filepath
            filename = source_blob_path.split('/')[-1]
            destination_blob_path = f"{destination_folder}/{filename}"

            copy_file_in_gcs(source_blob_path, destination_blob_path, bucket_name)
            writer.writerow([index, destination_blob_path, row.human_transcript_text, dataset_name])

def log_statistics(table_id, log_file="statistics.log"):
    bq_client = bigquery.Client()
    stats_query = f"""
    SELECT
      AVG(categories.proper_nouns.density) AS mean_proper_nouns_density,
      VAR_SAMP(categories.proper_nouns.density) AS variance_proper_nouns_density,
      STDDEV_SAMP(categories.proper_nouns.density) AS stddev_proper_nouns_density,
      APPROX_QUANTILES(categories.proper_nouns.density, 100)[OFFSET(50)] AS P50_proper_nouns_density,
      APPROX_QUANTILES(categories.proper_nouns.density, 100)[OFFSET(90)] AS P90_proper_nouns_density,
      APPROX_QUANTILES(categories.proper_nouns.density, 100)[OFFSET(99)] AS P99_proper_nouns_density,
      AVG(duration) AS mean_duration,
      VAR_SAMP(duration) AS variance_duration,
      STDDEV_SAMP(duration) AS stddev_duration,
      APPROX_QUANTILES(duration, 100)[OFFSET(50)] AS P50_duration,
      APPROX_QUANTILES(duration, 100)[OFFSET(90)] AS P90_duration,
      APPROX_QUANTILES(duration, 100)[OFFSET(99)] AS P99_duration
    FROM
      `{table_id}`
    """

    result = bq_client.query(stats_query).result()
    with open(log_file, "w") as file:
        for row in result:
            for key, value in row.items():
                # Check if the value is a float and format it
                if isinstance(value, float):
                    file.write(f"{key}: {value:.3f}\n")
                else:
                    file.write(f"{key}: {value}\n")

def main():
    dataset_name = "proper_nouns_2024_10hrs_youtube_usm_scraped_dataset"
    source_table_id = "assemblyai-nlp.youtube_usm_scraped_dataset.2024-03-scrape-human-transcripts-distinct-w-new-categories-full"
    filtered_table_name = "filtered_proper_nouns_10hrs_youtube_usm_scraped_dataset"
    bucket_name = "aai-us-central1"
    destination_folder = f"datasets/asr_analysis/datasets/{dataset_name}"
    
    # Log file for dataset statistics
    local_path = os.path.dirname(os.path.abspath(__file__))
    local_dataset_dir = f"{local_path}/datasets/proper_nouns"
    if not os.path.exists(local_dataset_dir):
        os.makedirs(local_dataset_dir)
    dataset_log_file = f"{local_dataset_dir}/{dataset_name}_statistics.log"
    local_output_file = f"{local_dataset_dir}/{dataset_name}.tsv"

    gcs_output_path = f"datasets/asr_analysis/english_test_benchmarks/{dataset_name}_v2.tsv"
    project_id = 'assemblyai-nlp' # Replace with your project ID

    # Create a new filtered table
    # Generate the current timestamp
    add_timestamp = True # Set to False if you don't want to add a timestamp to the table name
    if add_timestamp:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filtered_table_name = f"{filtered_table_name}_{timestamp}"
    filtered_table_id = create_filtered_table(source_table_id, filtered_table_name, project_id)

    # Generate TSV file from the filtered table and copy files to GCS
    print(f"Generating TSV file from the filtered table: {filtered_table_id}")
    query = f"SELECT raw_filepath, human_transcript_text FROM `{filtered_table_id}`"
    generate_tsv_from_table(query, local_output_file, destination_folder, dataset_name, bucket_name)

    # Log statistics of the filtered table
    print(f"Logging statistics for the filtered table: {filtered_table_id}")
    log_statistics(filtered_table_id, log_file=dataset_log_file)

    # Upload the generated TSV file to GCS and clean up the local file
    print(f"Uploading the TSV file to GCS: {gcs_output_path}")
    upload_to_gcs(local_output_file, gcs_output_path, bucket_name)
    os.remove(local_output_file)

if __name__ == "__main__":
    main()