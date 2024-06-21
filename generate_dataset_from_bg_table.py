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
'transcript_field', and a structured field 'categories.proper_nouns.density' for proper noun density.

2. **Table Creation**: It then creates a new BigQuery table to store these filtered results, 
ensuring that the data is more focused and manageable for further analysis.

3. **TSV Generation and File Uploading**: A TSV file is generated from the new table, 
containing essential details of the filtered records. The script uploads this TSV file 
and the corresponding audio files, referenced in the filepath_field such as 'raw_filepath' field of the table, 
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
from id_encoder import encode_id

def get_table_schema(table_id):
    client = bigquery.Client()
    table = client.get_table(table_id)  # Make an API request.
    print(f"Schema for table {table_id}:")
    for field in table.schema:
        print(f"Field name: {field.name}, Field type: {field.field_type}")

def file_exists_in_gcs(bucket_name, file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return blob.exists()

def upload_to_gcs(local_path, destination_path, bucket_name):
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(destination_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{destination_path}"

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

from google.cloud import bigquery

from google.cloud import bigquery

def create_filtered_table(source_table_id, filtered_table_name, project_id, category_field, 
                          delete_existing=False,
                          transcript_field='human_transcript_text',
                          total_duration_threshold_in_sec=36000,
                          category_stat_field='density',
                          filepath_field='transcoded_filepath'):
    assert filepath_field in ['raw_filepath', 'transcoded_filepath']
    assert category_stat_field in ['density', 'count']
    category_stat_field = f"categories.{category_field}.{category_stat_field}"
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
        if category_field == "proper_noun":
            filter_query = f"""
            WITH SortedFiltered AS (
                SELECT 
                    id,
                    {filepath_field},
                    {transcript_field},
                    duration,
                    categories.{category_field}.density AS density,
                    categories.{category_field}.count AS {category_field}_count,
                    categories.{category_field}.exists AS {category_field}_exists,
                    categories.{category_field}.evidence AS {category_field}_evidence,
                    SUM(duration) OVER (ORDER BY {category_stat_field} DESC) AS running_total
                FROM 
                    `{source_table_id}`
                WHERE 
                    duration BETWEEN 60 AND 1200 
                    AND NOT {transcript_field} LIKE '%[Content_Types].xml%'
                    AND NOT REGEXP_CONTAINS({transcript_field}, r'\\b[A-Z]{2,}\\b')  -- Exclude words in all caps
                    AND NOT REGEXP_CONTAINS({transcript_field}, r'(\\b[A-Z]{2,}\\s+){2,}')  -- Exclude segments with multiple consecutive all-caps words
                    AND REGEXP_CONTAINS({transcript_field}, r'([.!?]\\s+[A-Z])|([.!?]$)')  -- Check for proper sentence structure
                    AND NOT REGEXP_CONTAINS({transcript_field}, r'♪')  -- Exclude lines with musical notes, likely from machine transcription of music
                    AND LENGTH({transcript_field}) >= 10  -- Set a minimum threshold for transcript length
            ),
            LimitedByTime AS (
                SELECT 
                    *,
                    running_total AS total_duration_selected_files  -- Alias running_total as total_duration_selected_files
                FROM 
                    SortedFiltered
                WHERE 
                    running_total <= {total_duration_threshold_in_sec}
            ),
            FilteredWithCount AS (
                SELECT 
                    *,
                    (SELECT COUNT(*) FROM LimitedByTime WHERE {category_field}_count > 0 AND {category_field}_count IS NOT NULL) AS total_count
                FROM 
                    LimitedByTime
                WHERE 
                    {category_field}_count > 0 
                    AND {category_field}_count IS NOT NULL  -- Example condition for count threshold
            )
            SELECT 
                orig.*,
                f.density AS selected_density,
                f.{category_field}_count AS selected_{category_field}_count,
                f.{category_field}_exists AS selected_{category_field}_exists,
                f.{category_field}_evidence AS selected_{category_field}_evidence,
                f.total_duration_selected_files,
                f.total_count,
                f.running_total AS final_running_total
            FROM 
                `{source_table_id}` orig
            LEFT JOIN 
                FilteredWithCount f
            ON 
                orig.id = f.id
            WHERE 
                f.final_running_total <= {total_duration_threshold_in_sec}  -- Final limit to total duration
            ORDER BY 
                f.density DESC
            """
        else:
            filter_query = f"""
            -- Step 1: Create Partitioned Temporary Tables
            WITH PartitionedData AS (
                SELECT 
                    id,
                    {filepath_field},
                    {transcript_field},
                    duration,
                    categories.{category_field}.density AS density,
                    categories.{category_field}.count AS {category_field}_count,
                    categories.{category_field}.exists AS {category_field}_exists,
                    categories.{category_field}.evidence AS {category_field}_evidence,
                    MOD(FARM_FINGERPRINT(id), 20) AS partition_id  -- Create 20 partitions
                FROM 
                    `{source_table_id}`
                WHERE 
                    duration BETWEEN 0 AND 1200 
                    AND NOT REGEXP_CONTAINS({transcript_field}, r'♪')
                    AND LENGTH({transcript_field}) >= 10
                    AND {filepath_field} IS NOT NULL
                    AND categories.{category_field}.count IS NOT NULL
            ),

            -- Step 2: Process Each Partition Separately
            RunningTotalPartitioned AS (
                SELECT 
                    *,
                    SUM(duration) OVER (PARTITION BY partition_id ORDER BY {category_field}_count DESC) AS partition_running_total
                FROM 
                    PartitionedData
            ),

            LimitedByTimePartitioned AS (
                SELECT 
                    *,
                    partition_running_total AS total_duration_selected_files
                FROM 
                    RunningTotalPartitioned
                WHERE 
                    partition_running_total <= 100000  -- Limit to a larger amount to ensure we capture enough data across partitions
            ),

            -- Step 3: Combine the Results
            CombinedData AS (
                SELECT 
                    id,
                    {filepath_field},
                    {transcript_field},
                    duration,
                    density,
                    {category_field}_count AS {category_field}_count,
                    {category_field}_exists AS {category_field}_exists,
                    {category_field}_evidence AS {category_field}_evidence,
                    total_duration_selected_files,
                    COUNT(*) OVER () AS total_count
                FROM 
                    LimitedByTimePartitioned
                WHERE 
                    {category_field}_count > 0
            ),

            -- Step 4: Final Selection and Ordering
            FinalResult AS (
                SELECT 
                    id,
                    {filepath_field},
                    {transcript_field},
                    duration,
                    density,
                    {category_field}_count,
                    {category_field}_exists,
                    {category_field}_evidence,
                    total_duration_selected_files,
                    total_count,
                    SUM(duration) OVER (ORDER BY {category_field}_count DESC) AS final_running_total
                FROM 
                    CombinedData
            )

            SELECT 
                orig.*,
                f.density AS selected_density,
                f.{category_field}_count AS selected_{category_field}_count,
                f.{category_field}_exists AS selected_{category_field}_exists,
                f.{category_field}_evidence AS selected_{category_field}_evidence,
                f.total_duration_selected_files,
                f.total_count,
                f.final_running_total
            FROM 
                `{source_table_id}` orig
            LEFT JOIN 
                FinalResult f
            ON 
                orig.id = f.id
            WHERE 
                f.final_running_total <= {total_duration_threshold_in_sec}  -- Final limit to total duration
            ORDER BY 
                f.density DESC;
            """

        # Run the filter query and create the destination table
        job_config = bigquery.QueryJobConfig(destination=destination_table_id)
        job = bq_client.query(filter_query, job_config=job_config)
        job.result()  # Wait for the job to complete

        # Cleanup Temporary Tables
        cleanup_query = f"""
        DROP TABLE IF EXISTS `{project_id}.{source_table_id.split('.')[1]}.temp_filtered_{category_field}_partitioned`;
        DROP TABLE IF EXISTS `{project_id}.{source_table_id.split('.')[1]}.temp_running_total_{category_field}_partitioned`;
        DROP TABLE IF EXISTS `{project_id}.{source_table_id.split('.')[1]}.temp_limited_by_time_{category_field}_partitioned`;
        DROP TABLE IF EXISTS `{project_id}.{source_table_id.split('.')[1]}.temp_combined_{category_field}`;
        """
        bq_client.query(cleanup_query).result()

    return destination_table_id

def generate_tsv_from_table(query, local_output_file, destination_folder, 
                            dataset_name, bucket_name, transcode_audio=False,
                            transcript_field='human_transcript_text',
                            filepath_field='transcoded_filepath'):
    bq_client = bigquery.Client()
    results = bq_client.query(query)

    with open(local_output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Index', 'Paths', 'Labels', 'Dataset'])
        for index, row in enumerate(results):
            source_blob_path = row[filepath_field]
            orig_filename = os.path.splitext(os.path.basename(source_blob_path))[0]
            encoded_orig_filename = encode_id(orig_filename)
            extension = os.path.splitext(source_blob_path)[-1].lower()

            if (extension not in ['.wav', '.flac'] or transcode_audio):
                # Transcode if not WAV or FLAC or if transcoding is explicitly requested
                converted_path = detect_and_convert(source_blob_path, bucket_name, destination_folder, new_filename=encoded_orig_filename)
            else:
                # Copy with new name if it's already in a suitable format
                new_destination_path = f"{destination_folder}/{encoded_orig_filename}{extension}"
                copy_file_in_gcs(source_blob_path, new_destination_path, bucket_name)
                converted_path = f"gs://{bucket_name}/{new_destination_path}"

            if converted_path:
                if not converted_path.startswith("gs://"):
                    # Upload to GCS if the path isn't already a GCS URL
                    uploaded_path = destination_folder + '/' + os.path.basename(converted_path)
                    upload_to_gcs(converted_path, uploaded_path, bucket_name)
                    destination_blob_path = f"gs://{bucket_name}/{uploaded_path}"
                    os.remove(converted_path)  # Clean up converted file
                else:
                    destination_blob_path = converted_path

                writer.writerow([index, destination_blob_path, row[transcript_field], dataset_name])
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

def generate_tsv_from_directories(transcript_dir, audio_dir, local_output_file, destination_folder, dataset_name, bucket_name):
    """
    Generates a TSV file from transcript from vendors such as HappyScribe and audio files, uploading missing audio files to GCS.

    Args:
        transcript_dir (str): Directory containing transcript files, one corresponding to each audio file. e.g., HappyScribe transcripts.
        The transcript files should have the same name as the audio files with '.txt' extension.
        audio_dir (str): Directory containing corresponding audio files.
        local_output_file (str): Path to save the output TSV file.
        bucket_name (str): Name of the GCS bucket where audio files are stored or will be uploaded.
        dataset_name (str): Name of the dataset, used to structure paths in the bucket.

    Each row in the generated TSV file includes:
        - Index: The index of the transcript in the directory listing.
        - Paths: The GCS path of the audio file.
        - Labels: The transcript text.
        - Dataset: The name of the dataset.
    """

    transcript_files = os.listdir(transcript_dir)
    with open(local_output_file, 'w', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow(['Index', 'Paths', 'Labels', 'Dataset'])
        for index, filename in enumerate(transcript_files):
            if filename.endswith('.txt'):
                transcript_path = os.path.join(transcript_dir, filename)
                # Remove the '.txt' and check the remaining part ends with .wav or .flac
                audio_filename = filename[:-4]  # Strips off '.txt'
                audio_path = os.path.join(audio_dir, audio_filename)
                destination_path = f"{destination_folder}/{audio_filename}"
                
                # Check if file exists in GCS
                if file_exists_in_gcs(bucket_name, destination_path):
                    gcs_path = f"{dataset_name}/{audio_filename}"
                else:
                    # Upload the audio file to GCS if it's missing
                    print(f"Uploading missing audio file to GCS: {audio_path}")
                    if os.path.exists(audio_path) and (audio_path.endswith('.wav') or audio_path.endswith('.flac')):
                        upload_to_gcs(audio_path, destination_path, bucket_name)
                        gcs_path = f"{dataset_name}/{audio_filename}"
                    else:
                        print(f"Skipping file due to missing audio: {audio_path}")
                        continue

                with open(transcript_path, 'r') as t_file:
                    transcript = t_file.read().strip()
                    writer.writerow([index, gcs_path, transcript, dataset_name])

def main(category):
    # List of supported categories
    supported_categories = ['proper_noun', 'email', 'address', 'alphanumeric', 'phone_number', 'website']

    # Check if the provided category is supported
    if category not in supported_categories:
        raise ValueError(f"Unsupported category '{category}'. Please choose from: {', '.join(supported_categories)}")

    # Define the dataset name based on the category
    #dataset_name = f"{category}_10hrs_youtube_scraped_2024_04_18_dataset_HappyScribe_transcript"
    dataset_name = f"{category}_10hrs_youtube_scraped_2024_04_18_dataset_Google_transcript"

    # Other parameters
    source_table_id = "assemblyai-nlp.youtube_scrape_2024_04_18.filtered_30_to_1200_sec_en_metadata_1_wav_full_w_Roberta_tags_full"
    filtered_table_name = f"filtered_{category}_10hrs_youtube_usm_scraped_dataset_w_Roberta_tags"
    bucket_name = "aai-us-central1"
    destination_folder = f"datasets/asr_analysis/datasets/{dataset_name}"
    transcript_field='google_transcript_text'
    filepath_field='transcoded_filepath'
    category_stat_field='count'

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
    add_timestamp = False # Set to False if you don't want to add a timestamp to the table name
    if add_timestamp:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filtered_table_name = f"{filtered_table_name}_{timestamp}"
    
    filtered_table_id = create_filtered_table(source_table_id, filtered_table_name, 
                                              project_id, category, delete_existing=True,
                                              transcript_field=transcript_field,
                                              total_duration_threshold_in_sec=72000,
                                              category_stat_field=category_stat_field,
                                              filepath_field=filepath_field)
    # Print the schema of the filtered table
    get_table_schema(filtered_table_id)

    # Generate TSV file from the filtered table and copy files to GCS
    transcript_dir=None
    audio_dir=None
    local_output_file = f"{local_dataset_dir}/{dataset_name}_w_happyScribe_transcript.tsv"
    if transcript_dir and audio_dir:
        generate_tsv_from_directories(transcript_dir, audio_dir, local_output_file, destination_folder, dataset_name, bucket_name)
    else:
        query = f"SELECT {filepath_field}, {transcript_field} FROM `{filtered_table_id}`"
        generate_tsv_from_table(query, local_output_file, destination_folder,
                                dataset_name, bucket_name,
                                transcript_field=transcript_field,
                                filepath_field=filepath_field)

    # Log statistics of the filtered table
    log_stats = False  # Set to True if you want to log statistics
    if log_stats:
        compute_statistics(filtered_table_id, category_field=category, log_file=local_stats_log_file)
        if category != 'proper_nouns':
            aggregate_evidence(filtered_table_id, category_field=category, log_file=f"{local_dataset_dir}/{dataset_name}_aggregated_evidence.log")

    # Upload the generated TSV file to GCS and clean up the local file
    upload_to_gcs(local_output_file, gcs_output_path, bucket_name)
    os.remove(local_output_file)

if __name__ == "__main__":
    #for category in ['proper_noun', 'email', 'address', 'alphanumeric', 'phone_number', 'website']:
    #for category in ['email', 'address','alphanumeric', 'phone_number', 'website']:
    for category in ['address','phone_number', 'website']:
        print(f"Processing category: {category}")
        main(category)  # Process each category