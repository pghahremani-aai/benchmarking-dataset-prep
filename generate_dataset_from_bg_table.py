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
import subprocess
from id_encoder import encode_id, decode_id

def upload_to_gcs(local_path, destination_path, bucket_name):
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(destination_path)
    blob.upload_from_filename(local_path)

def download_from_gcs(source_blob_path, download_dir='/tmp'):
    """Download a file from GCS to a local directory."""
    client = storage.Client()
    bucket_name, blob_name = source_blob_path.replace('gs://', '').split('/', 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_path = os.path.join(download_dir, blob_name.split('/')[-1])
    blob.download_to_filename(local_path)
    return local_path

def convert_audio(source_path, target_format):
    """Convert audio file to a desired format using FFmpeg, handling filenames starting with '-'. """
    target_path = os.path.splitext(source_path)[0] + '.' + target_format
    # Use './' to correctly handle files beginning with a hyphen
    command = ['ffmpeg', '-i', source_path, target_path]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120)  # 2-minute timeout
        return target_path
    except subprocess.TimeoutExpired:
        print(f"Conversion timeout for file: {source_path}")
        return None
    except subprocess.CalledProcessError:
        print(f"Error during conversion for file: {source_path}")
        return None

def gcs_file_exists(bucket_name, path):
    """Check if a file exists in a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    return blob.exists()

def detect_and_convert(source_blob_path, bucket_name, destination_folder, new_filename=None, formats={'m4a': 'wav', 'webm': 'flac'}, download_dir='/tmp'):
    """Detect the file format and convert if needed, checking if the converted file already exists."""
    extension = source_blob_path.split('.')[-1].lower()
    target_format = formats.get(extension)
    if target_format:
        if new_filename:
            target_path = new_filename + '.' + target_format
        else:
            target_path = os.path.splitext(os.path.basename(source_blob_path))[0] + '.' + target_format
        full_gcs_path = f"{destination_folder}/{target_path}"
        
        # Check if the converted file already exists in GCS
        if gcs_file_exists(bucket_name, full_gcs_path):
            print(f"Converted file already exists in GCS: {full_gcs_path}")
            return f"gs://{bucket_name}/{full_gcs_path}"
        else:
            # Proceed with download and conversion
            local_path = download_from_gcs(source_blob_path, download_dir)
            converted_path = convert_audio(local_path, target_format)
            os.remove(local_path)  # Clean up original file
            return converted_path
    return None

def check_table_exists(table_id):
    bq_client = bigquery.Client()
    try:
        bq_client.get_table(table_id)
        return True  # Table exists
    except Exception as e:
        return False  # Table does not exist or an error occurred

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

def create_filtered_table(source_table_id, filtered_table_name, project_id, category_field, delete_existing=False):
    bq_client = bigquery.Client()
    create_new_table = False
    destination_table_id = f"{project_id}.{source_table_id.split('.')[1]}.{filtered_table_name}"
    
    # Check if the destination table already exists
    if not delete_existing:
        if check_table_exists(destination_table_id):
            print(f"Table {destination_table_id} already exists. Skipping table creation.")
            return destination_table_id
        else:
            print(f"Table {destination_table_id} does not exist. Creating a new table...")
            create_new_table = True
    else:
        if check_table_exists(destination_table_id):
            print(f"Table {destination_table_id} already exists. Deleting and recreating...")
            bq_client.delete_table(destination_table_id)  # Delete the existing table
        else:
            print(f"Table {destination_table_id} does not exist. Creating a new table...")
        create_new_table = True
        
    if create_new_table:    
        # filter_query = f"""
        # WITH SortedFiltered AS (
        # SELECT 
        #     *,
        #     SUM(duration) OVER (ORDER BY categories.{category_field}.density DESC) AS running_total
        # FROM 
        #     `{source_table_id}`
        # WHERE 
        #     duration BETWEEN 120 AND 1200 
        #     AND NOT human_transcript_text LIKE '%[Content_Types].xml%'
        #     AND NOT REGEXP_CONTAINS(human_transcript_text, r'\\b[A-Z]{2,}\\b')  -- Exclude words in all caps
        #     AND NOT REGEXP_CONTAINS(human_transcript_text, r'(\\b[A-Z]{2,}\\s+){2,}')  -- Exclude segments with multiple consecutive all-caps words
        #     AND REGEXP_CONTAINS(human_transcript_text, r'([.!?]\\s+[A-Z])|([.!?]$)')  -- Check for proper sentence structure
        #     AND NOT REGEXP_CONTAINS(human_transcript_text, r'♪')  -- Exclude lines with musical notes, likely from machine transcription of music
        # ),
        # LimitedByTime AS (
        #     SELECT 
        #         *
        #     FROM 
        #         SortedFiltered
        #     WHERE 
        #         running_total <= 36000
        # )
        # SELECT 
        #     *
        # FROM 
        #     LimitedByTime
        # WHERE
        #     categories.{category_field}.count > 0  -- Example condition for count threshold
        # ORDER BY 
        #     categories.{category_field}.density DESC
        # """
        filter_query = f"""
        WITH SortedFiltered AS (
        SELECT 
            *,
            SUM(duration) OVER (ORDER BY categories.{category_field}.density DESC) AS running_total
        FROM 
            `{source_table_id}`
        WHERE 
            duration BETWEEN 60 AND 1200 
            AND NOT human_transcript_text LIKE '%[Content_Types].xml%'
            AND NOT REGEXP_CONTAINS(human_transcript_text, r'\\b[A-Z]{2,}\\b')  -- Exclude words in all caps
            AND NOT REGEXP_CONTAINS(human_transcript_text, r'(\\b[A-Z]{2,}\\s+){2,}')  -- Exclude segments with multiple consecutive all-caps words
            AND REGEXP_CONTAINS(human_transcript_text, r'([.!?]\\s+[A-Z])|([.!?]$)')  -- Check for proper sentence structure
            AND NOT REGEXP_CONTAINS(human_transcript_text, r'♪')  -- Exclude lines with musical notes, likely from machine transcription of music
            AND LENGTH(human_transcript_text) >= 10  -- Set a minimum threshold for transcript length
        ),
        LimitedByTime AS (
            SELECT 
                *,
                running_total AS total_duration_selected_files  -- Alias running_total as total_duration_selected_files
            FROM 
                SortedFiltered
            WHERE 
                running_total <= 36000
        )
        SELECT 
            *
        FROM 
            LimitedByTime
        WHERE
            categories.{category_field}.count > 0  -- Example condition for count threshold
        ORDER BY 
            categories.{category_field}.density DESC

        """

        # Run the filter query and create the destination table
        job_config = bigquery.QueryJobConfig(destination=destination_table_id)
        job = bq_client.query(filter_query, job_config=job_config)
        job.result()  # Wait for the job to complete
    
    return destination_table_id

def generate_tsv_from_table(query, local_output_file, destination_folder, dataset_name, bucket_name):
    bq_client = bigquery.Client()
    results = bq_client.query(query)

    with open(local_output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['', 'paths', 'labels', 'dataset'])
        cnt = 0
        for index, row in enumerate(results):
            source_blob_path = row.raw_filepath
            import pdb; pdb.set_trace()
            # Extract the original filename without extension
            orig_filename = os.path.splitext(os.path.basename(source_blob_path))[0]
            # Encode the filename
            encoded_orig_filename = encode_id(orig_filename)
            converted_path = detect_and_convert(source_blob_path, bucket_name, destination_folder, new_filename=encoded_orig_filename)
            
            cnt+=1
            print(f"The file is converted in path {converted_path}, cnt {cnt}.")
            if converted_path:
                if not converted_path.startswith("gs://"):
                    # Upload to GCS if the path isn't already a GCS URL
                    uploaded_path = destination_folder + '/' + os.path.basename(converted_path)
                    upload_to_gcs(converted_path, uploaded_path, bucket_name)
                    print(f"The file is uploaded into the path {uploaded_path}")
                    destination_blob_path = f"gs://{bucket_name}/{uploaded_path}"
                    os.remove(converted_path)  # Clean up converted file
                else:
                    destination_blob_path = converted_path

                writer.writerow([index, '/'.join(destination_blob_path.split('/')[-2:]), row.human_transcript_text, dataset_name])
            else:
                print(f"Skipping file due to issues in conversion or upload: {source_blob_path}")          

def compute_statistics(table_id, category_field, log_file="statistics.log"):
    bq_client = bigquery.Client()

    # Define the statistics query with the dynamic category field name
    stats_query = f"""
        SELECT
            SUM(categories.{category_field}.count) AS total_{category_field}_count,
            AVG(categories.{category_field}.count) AS mean_{category_field}_count,
            AVG(categories.{category_field}.density) AS mean_{category_field}_density,
            VAR_SAMP(categories.{category_field}.density) AS variance_{category_field}_density,
            STDDEV_SAMP(categories.{category_field}.density) AS stddev_{category_field}_density,
            APPROX_QUANTILES(categories.{category_field}.density, 100)[OFFSET(50)] AS P50_{category_field}_density,
            APPROX_QUANTILES(categories.{category_field}.density, 100)[OFFSET(90)] AS P90_{category_field}_density,
            APPROX_QUANTILES(categories.{category_field}.density, 100)[OFFSET(99)] AS P99_{category_field}_density,
            SUM(duration) AS total_duration,
            AVG(duration) AS mean_duration,
            VAR_SAMP(duration) AS variance_duration,
            STDDEV_SAMP(duration) AS stddev_duration,
            APPROX_QUANTILES(duration, 100)[OFFSET(50)] AS P50_duration,
            APPROX_QUANTILES(duration, 100)[OFFSET(90)] AS P90_duration,
            APPROX_QUANTILES(duration, 100)[OFFSET(99)] AS P99_duration
        FROM
            `{table_id}`
    """

    # Execute the stats query
    result_stats = bq_client.query(stats_query).result()
    with open(log_file, "w") as file:
        for row in result_stats:
            for key, value in row.items():
                # Check if the value is a float and format it
                if isinstance(value, float):
                    file.write(f"{key}: {value:.3f}\n")
                else:
                    file.write(f"{key}: {value}\n")

def aggregate_evidence(table_id, category_field, log_file="aggregated_evidence.log"):
    bq_client = bigquery.Client()

    # Define the aggregate evidence query
    aggregate_query = f"""
        WITH FlattenedEvidence AS (
            SELECT
                id,
                evidence
            FROM
                `{table_id}`,
                UNNEST(categories.{category_field}.evidence) AS evidence
        )
        SELECT
            id,
            ARRAY_AGG(evidence) AS aggregated_evidence
        FROM
            FlattenedEvidence
        GROUP BY
            id
    """

    # Execute the aggregate query
    result_aggregate = bq_client.query(aggregate_query).result()
    with open(log_file, "w") as file:
        for row in result_aggregate:
            for key, value in row.items():
                # Check if the value is a float and format it
                if isinstance(value, float):
                    file.write(f"{key}: {value:.3f}\n")
                else:
                    file.write(f"{key}: {value}\n")

def main(category):
    # List of supported categories
    supported_categories = ['proper_noun', 'email', 'address', 'alphanumeric', 'phone_number', 'website']

    # Check if the provided category is supported
    if category not in supported_categories:
        raise ValueError(f"Unsupported category '{category}'. Please choose from: {', '.join(supported_categories)}")

    # Define the dataset name based on the category
    dataset_name = f"{category}_2024_10hrs_youtube_usm_scraped_dataset_v3"

    # Other parameters
    source_table_id = "assemblyai-nlp.youtube_usm_scraped_dataset.2024-03-scrape-human-transcripts-selected-data-join-distinct-w-Roberta-tags-full"
    filtered_table_name = f"filtered_{category}_10hrs_youtube_usm_scraped_dataset_w_Roberta_tags_v2"
    bucket_name = "aai-us-central1"
    destination_folder = f"datasets/asr_analysis/datasets/{dataset_name}"
    
    # Log file for dataset statistics
    local_path = os.path.dirname(os.path.abspath(__file__))
    local_dataset_dir = f"{local_path}/datasets/{category}"
    if not os.path.exists(local_dataset_dir):
        os.makedirs(local_dataset_dir)
    local_stats_log_file = f"{local_dataset_dir}/{dataset_name}_statistics.log"
    local_output_file = f"{local_dataset_dir}/{dataset_name}.tsv"

    gcs_output_path = f"datasets/asr_analysis/english_test_benchmarks/{dataset_name}.tsv"
    project_id = 'assemblyai-nlp' # Replace with your project ID

    # Create a new filtered table
    # Generate the current timestamp
    add_timestamp = False # Set to False if you don't want to add a timestamp to the table name
    if add_timestamp:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filtered_table_name = f"{filtered_table_name}_{timestamp}"
    #filtered_table_id = f"{project_id}.{source_table_id.split('.')[1]}.{filtered_table_name}"
    filtered_table_id = create_filtered_table(source_table_id, filtered_table_name, project_id, category, delete_existing=False)

    # Generate TSV file from the filtered table and copy files to GCS
    print(f"Generating TSV file from the filtered table: {filtered_table_id}")
    query = f"SELECT raw_filepath, human_transcript_text FROM `{filtered_table_id}`"
    generate_tsv_from_table(query, local_output_file, destination_folder, dataset_name, bucket_name)

    # Log statistics of the filtered table
    print(f"Logging statistics for the filtered table: {filtered_table_id}")
    compute_statistics(filtered_table_id, category_field=category, log_file=local_stats_log_file)
    if category != 'proper_nouns':
        aggregate_evidence(filtered_table_id, category_field=category, log_file=f"{local_dataset_dir}/{dataset_name}_aggregated_evidence.log")

    # Upload the generated TSV file to GCS and clean up the local file
    print(f"Uploading the TSV file to GCS: {gcs_output_path}")
    upload_to_gcs(local_output_file, gcs_output_path, bucket_name)
    os.remove(local_output_file)

if __name__ == "__main__":
    for category in ['proper_noun', 'email', 'address', 'alphanumeric', 'phone_number', 'website']:
    #for category in ['alphanumeric', 'phone_number', 'website']:
    #for category in ['proper_noun']:
        print(f"Processing category: {category}")
        main(category)  # Process each category