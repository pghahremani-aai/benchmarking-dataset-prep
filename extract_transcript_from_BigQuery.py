from google.cloud import bigquery
from collections import Counter
import json, logging
import numpy as np
import re
import spacy
import os
from transformers import pipeline
import re
"""
    This script extracts and filters data from a BigQuery table based on certain criteria. 
    It also computes various metrics and statistics on the extracted data.
    The filtered data is then written to a new table in BigQuery.

    You need to install the following packages:
    pip install google-cloud-bigquery
    pip install spacy
    python -m spacy download en_core_web_sm

    or you can use myenvironment.yml file to create a conda environment
    conda env create -f myenvironment.yml
    conda activate myenvironment

    or you can use myrequirements.txt file to install the required packages
    pip install -r myrequirements.txt
    
    You need to set up the Google Cloud SDK and authenticate with your Google Cloud account
    https://cloud.google.com/sdk/docs/install
    https://cloud.google.com/sdk/docs/authorizing
    https://cloud.google.com/bigquery/docs/reference/libraries
    https://cloud.google.com/bigquery/docs/quickstarts/quickstart-client-libraries
    https://cloud.google.com/bigquery/docs/quickstarts/quickstart-client-libraries#bigquery-simple-app-quickstart-python
    https://cloud.google.com/bigquery/docs/quickstarts/quickstart-client-libraries#bigquery-simple-app-quickstart-python
    or 
    source myenv/bin/activate    
"""
# Function to initialize the NLP model
def get_nlp_model(use_transformer=False):
    if use_transformer:
        return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", framework="pt")
    else:
        return spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def write_merged_data_to_new_table(client, dataset_id, source_table_id, selected_data, destination_dataset_id, destination_table_id):
    # Assume selected_data is a list of dictionaries with new metrics
    # First, upload selected_data to a temporary BigQuery table
    temp_table_id = f'{destination_dataset_id}.selected_data'
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, autodetect=True)
    job = client.load_table_from_json(selected_data, f'{client.project}.{temp_table_id}', job_config=job_config)
    job.result()  # Wait for the job to complete

    # Now, create the query to merge the temporary table with the original table
    # You should specify column names explicitly to avoid 'temp_' prefix and control the schema of the final table
    merge_query = f"""
    SELECT orig.*, temp.proper_noun_ratio, temp.tag_distribution, temp.email_count, temp.phone_count  -- specify more columns as needed
    FROM `{client.project}.{dataset_id}.{source_table_id}` AS orig
    JOIN `{client.project}.{temp_table_id}` AS temp
    ON orig.raw_filepath = temp.raw_filepath
    """

    # Define the destination table and job configuration
    destination_table_ref = client.dataset(destination_dataset_id).table(destination_table_id)
    job_config = bigquery.QueryJobConfig(destination=destination_table_ref, write_disposition='WRITE_TRUNCATE')

    # Execute the merge query and write the results to the destination table
    query_job = client.query(merge_query, job_config=job_config)
    query_job.result()  # Wait for the job to complete

    # Clean up the temporary table
    client.delete_table(temp_table_id)

    print(f"Merged data written to {destination_dataset_id}.{destination_table_id}")

def write_merged_data_to_new_table(client, dataset_id, source_table_id, selected_data, destination_dataset_id, destination_table_id):
    # Upload selected_data to a temporary BigQuery table
    temp_table_id = 'temp_selected_data'
    temp_table_ref = client.dataset(dataset_id).table(temp_table_id)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True
    )
    job = client.load_table_from_json(selected_data, temp_table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete

    # Get the schema of the temporary table to build the column list for the SELECT statement
    temp_table = client.get_table(temp_table_ref)
    column_names = [field.name for field in temp_table.schema if field.name not in ['raw_filepath', 'human_transcript_text']]

    # Prepare a query to merge the temporary table with the original table
    # Alias columns from the temporary table to avoid name collisions
    temp_select_columns = ', '.join([f'temp.{name} AS temp_{name}' for name in column_names])
    orig_select_columns = 'orig.*'  # or specify columns explicitly if necessary
    merge_query = f"""
    CREATE OR REPLACE TABLE `{client.project}.{destination_dataset_id}.{destination_table_id}` AS
    SELECT {orig_select_columns}, {temp_select_columns}
    FROM `{client.project}.{dataset_id}.{source_table_id}` AS orig
    JOIN `{client.project}.{dataset_id}.{temp_table_id}` AS temp
    ON orig.raw_filepath = temp.raw_filepath
    """
    query_job = client.query(merge_query)
    query_job.result()  # Wait for the job to complete

    # Clean up: delete the temporary table
    client.delete_table(temp_table_ref)

    print(f"Merged data written to {destination_dataset_id}.{destination_table_id}")

def fetch_and_filter_data(client, dataset_id, table_id, raw_file_path_column=None, order_by_column=None, apply_filter=True, limit=1000, select_columns=None):
    if select_columns is None:
        select_columns = '*'  # Fetch all columns if none are specified

    if apply_filter and raw_file_path_column and order_by_column:
        query = f"""
        WITH RankedData AS (
            SELECT {select_columns}, ROW_NUMBER() OVER (PARTITION BY {raw_file_path_column} ORDER BY {order_by_column} DESC) AS rn
            FROM `{client.project}.{dataset_id}.{table_id}`
        )
        SELECT *
        FROM RankedData
        WHERE rn = 1
        LIMIT {limit}
        """
    else:
        query = f"SELECT {select_columns} FROM `{client.project}.{dataset_id}.{table_id}` LIMIT {limit}"

    query_job = client.query(query)
    
    # Process rows as they are streamed from BigQuery
    filtered_data = []
    for row in query_job.result(page_size=limit):  # Use page_size to control fetch size
        filtered_data.append(row)

    total_rows = len(filtered_data)
    return filtered_data, total_rows

# Compute data metrics function that adapts based on the NLP model used
import re

def compute_data_metrics(transcript, nlp, use_transformer=False):
    if use_transformer:
        results = nlp(transcript)
        # Extract proper nouns with the entity filter applied
        proper_nouns = [result['word'] for result in results 
                        if result['entity'] in ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
                        and len(result['word']) > 1
                        and re.match(r'^[A-Za-z]+$', result['word'])
                        and re.match(r'^[A-Z][a-z]+$', result['word'])]
    else:
        doc = nlp(transcript)
        proper_nouns = [ent.text for ent in doc.ents 
                        if ent.label_ in ['PERSON', 'ORG', 'GPE']
                        and len(ent.text) > 1
                        and re.match(r'^[A-Za-z]+$', ent.text)
                        and re.match(r'^[A-Z][a-z]+$', ent.text)]

    tag_distribution = Counter([ent.label_ for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']])

    proper_nouns_count = len(proper_nouns)
    total_words = len(transcript.split())
    proper_noun_ratio = proper_nouns_count / total_words if total_words > 0 else 0

    email_count = len(extract_email_addresses(transcript))
    phone_count = len(extract_phone_numbers(transcript))

    return {
        'proper_noun_ratio': proper_noun_ratio,
        'tag_distribution': tag_distribution,
        'email_count': email_count,
        'phone_count': phone_count
    }

def compute_statistics(data):
    """
    Compute statistical metrics for a given array of data.

    Parameters:
    - data: An array or list of numerical values.

    Returns:
    A dictionary containing the mean, variance, P50, P90, and P99 of the input data.
    """
    statistics = {
        'mean': np.mean(data),
        'variance': np.var(data),
        'P50': np.percentile(data, 50),  # Median
        'P90': np.percentile(data, 90),
        'P99': np.percentile(data, 99)
    }
    return statistics

def extract_email_addresses(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

def extract_phone_numbers(text):
    phone_pattern = r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    return re.findall(phone_pattern, text)

def extract_high_proper_noun_sections_with_timestamps(transcripts_with_timestamps, threshold=3):
    high_proper_noun_sections = []
    
    for start_ms, end_ms, transcript in transcripts_with_timestamps:
        doc = nlp(transcript)
        current_section = ''
        for sentence in doc.sents:
            proper_nouns_count = sum(1 for token in sentence if token.pos_ == 'PROPN')
            if proper_nouns_count >= threshold:
                current_section += str(sentence) + ' '
            else:
                if current_section:
                    high_proper_noun_sections.append((start_ms, end_ms, current_section.strip()))
                    current_section = ''
    
    return high_proper_noun_sections

def compute_proper_nouns_ratio_and_distribution(transcript):
    """
    Compute the ratio of proper nouns in a given transcript and the distribution of proper noun tags.
    
    Args:
        transcript (str): The transcript to analyze.
        
    Returns:
        tuple: A tuple containing the ratio of proper nouns in the transcript and a Counter object
               representing the distribution of proper noun tags.
    """
    if not transcript:
        return 0, Counter()

    doc = nlp(transcript)
    proper_nouns = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']]
    proper_nouns_count = len(proper_nouns)
    total_words = len(doc)
    ratio = proper_nouns_count / total_words if total_words > 0 else 0

    tag_distribution = Counter([ent.label_ for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']])
    
    return ratio, tag_distribution

def process_timestamps(timestamps, compute_metrics=True):
    timestamp_data = []
    for segment in timestamps:
        start_ms = segment['start_ms']
        end_ms = segment['end_ms']
        transcript_segment = segment['text']  # Adjust based on actual field name
        
        segment_metrics = compute_data_metrics(transcript_segment) if compute_metrics else {}
        timestamp_data.append({
            'start_ms': start_ms,
            'end_ms': end_ms,
            'transcript_segment': transcript_segment,
            **segment_metrics
        })
    
    return timestamp_data

def save_results_to_json(selected_data, selected_tag_distribution, json_path):
    if not os.path.exists(json_path):
        os.makedirs(json_path)

    with open(f'{json_path}/selected_data.json', 'w') as f:
        json.dump(selected_data, f, indent=4)

    with open(f'{json_path}/selected_tag_distribution.json', 'w') as f:
        json.dump(selected_tag_distribution, f, indent=4)

def compute_and_save_statistics(selected_data, json_path, fields_to_include=None):
    """
    Compute statistics for the selected data, save the filtered selected data and statistics to JSON files.

    :param selected_data: List of dictionaries containing selected data including raw file paths.
    :param json_path: Path to the directory where JSON files will be saved.
    :param fields_to_include: Optional list of fields to include in the saved JSON for selected data.
    """
    # Compute statistics
    proper_noun_ratios = [entry['proper_noun_ratio'] for entry in selected_data if 'proper_noun_ratio' in entry]
    statistics = compute_statistics(proper_noun_ratios)

    # Filter selected_data if fields_to_include is set
    if fields_to_include:
        filtered_selected_data = [
            {field: entry[field] for field in fields_to_include if field in entry}
            for entry in selected_data
        ]
    else:
        filtered_selected_data = selected_data

    # Ensure the output directory exists
    os.makedirs(json_path, exist_ok=True)

    # Save filtered selected data to JSON
    selected_data_file = os.path.join(json_path, 'selected_data.json')
    with open(selected_data_file, 'w') as f:
        json.dump(filtered_selected_data, f, indent=4)
    
    # Save statistics to JSON
    statistics_file = os.path.join(json_path, 'PNR_statistics.json')
    with open(statistics_file, 'w') as f:
        json.dump(statistics, f, indent=4)

    print(f"Selected data and statistics have been saved to {json_path}")

    return statistics

def compute_data_metrics_parallel(data, compute_metrics=True, transcript_column='human_transcript_text'):
    # Initialize the necessary resources inside the function
    nlp = spacy.load("en_core_web_sm")  # For example, load the spaCy model here if needed

    # Proceed with the processing using the locally initialized resources
    if compute_metrics and transcript_column in data:
        transcript = data[transcript_column]
        return compute_data_metrics(transcript, nlp)  # Assuming compute_data_metrics uses nlp
    else:
        return data  # or some default processing

def process_and_select_data(filtered_data, client, dataset_id, table_id, destination_dataset_id, destination_table_id,
                            transcript_column='human_transcript_text', 
                            timestamp_column='human_transcript_timestamps',
                            select_with_timestamp=False, data_selection_hrs=10, batch_size=1000, nlp=None, use_transformer=False):
    logging.info("Starting to process data...")

    total_processed = 0
    all_data_with_metrics = []
    
    # Process data in batches
    for i in range(0, len(filtered_data), batch_size):
        batch = filtered_data[i:i + batch_size]
        batch_data_with_metrics = []

        for data in batch:
            if select_with_timestamp and timestamp_column in data:
                timestamp_data = process_timestamps(data[timestamp_column])
                for segment in timestamp_data:
                    metrics = compute_data_metrics(segment['transcript_segment'], nlp, use_transformer=use_transformer)
                    batch_data_with_metrics.append({**segment, **metrics})
            else:
                metrics = compute_data_metrics(data[transcript_column], nlp, use_transformer=use_transformer)
                batch_data_with_metrics.append({**data, **metrics})
        
        all_data_with_metrics.extend(batch_data_with_metrics)
        total_processed += len(batch)
        logging.info(f"Processed {total_processed}/{len(filtered_data)} rows.")

    logging.info("Data processing completed. Starting sorting and selection...")
    
    
    sorted_data = sorted(all_data_with_metrics, key=lambda x: x['proper_noun_ratio'], reverse=True)

    total_duration = 0
    selected_data = []
    selected_tag_distribution = Counter()
    for data in sorted_data:
        total_duration += data.get('duration', 0)
        if total_duration > data_selection_hrs * 3600:
            break
        selected_data.append(data)
        selected_tag_distribution += data['tag_distribution']

    logging.info("Data selection completed. Starting to write data to the new BigQuery table...")

    return selected_data, selected_tag_distribution

def merge_with_original_data(client, dataset_id, table_id, selected_data, key_column):
    # Fetch original data based on keys in selected_data
    keys = [item[key_column] for item in selected_data]
    formatted_keys = ', '.join(f"'{key}'" for key in keys)
    query = f"""
    SELECT *
    FROM `{client.project}.{dataset_id}.{table_id}`
    WHERE {key_column} IN ({formatted_keys})
    """
    query_job = client.query(query)
    original_data = list(query_job.result())

    # Convert to dictionary for easy lookup
    original_data_dict = {row[key_column]: dict(row) for row in original_data}

    # Merge original data with selected_data
    merged_data = []
    for item in selected_data:
        key = item[key_column]
        original_row = original_data_dict.get(key, {})
        merged_row = {**original_row, **item}  # Original data merged with selected data
        merged_data.append(merged_row)

    return merged_data

def main(use_transformer=True):
    nlp = get_nlp_model(use_transformer=use_transformer)
    project_id = 'assemblyai-nlp'
    dataset_id = 'youtube_usm_scraped_dataset'
    # This table filtered out duplicate raw file paths and ordered by duration in BigQuery. 
    # So we no longer need to apply the filter in the fetch_and_filter_data function and can set apply_filter=False.
    table_id = '2024-03-scrape-human-transcripts-selected-data-join-distinct' 
    limit = 100 # If None, it will fetch all rows from the table. Otherwise, it will fetch the specified number of rows.
    batch_size = 50
    destination_dataset_id = 'youtube_usm_scraped_dataset'
    destination_table_id = f'2024-03-scrape-human-transcripts-selected-data-proper-nouns-ratio-10hrs-limit-{limit}-use-transformer-{use_transformer}'
    select_with_timestamp = False # Set to True if you want to process timestamp segments
    client = bigquery.Client(project=project_id)

    # Get the number of processors available
    num_processors = os.cpu_count()

    # Print the number of processors
    print(f"Number of processors available: {num_processors}")

    # Step 1: Fetch and Filter Data
    select_columns = 'raw_filepath, duration, human_transcript_text, human_transcript_timestamps'  # Specify the columns you need
    filtered_data, total_rows = fetch_and_filter_data(client, dataset_id, table_id, 'raw_filepath', 'duration', apply_filter=False, limit=limit, select_columns=select_columns)
    print(f"Filtered data: {len(filtered_data)} out of {total_rows} rows")
    # Step 2: Process and Select Data
    selected_data, selected_tag_distribution = process_and_select_data(filtered_data, client, dataset_id, table_id, 
                                                                       destination_dataset_id, destination_table_id,
                                                                       'human_transcript_text', 'human_transcript_timestamps',
                                                                       select_with_timestamp, 10, batch_size, nlp=nlp, use_transformer=use_transformer)

    
    # Step 3: Write Data to New Table
    # Merge selected_data with original fields and write to a new table
    write_merged_data_to_new_table(client, dataset_id, table_id, selected_data, destination_dataset_id, destination_table_id)

    # Step 4: Compute and Save Statistics
    json_path = 'data'
    # Define the fields you want to include in your JSON file
    fields_to_include = ['raw_filepath', 'proper_noun_ratio', 'duration', 'tag_distribution']
    selected_statistics = compute_and_save_statistics(selected_data, json_path, fields_to_include)

    print(f"Process completed. Statistics: {selected_statistics}")

    with open(f'{json_path}/selected_tag_distribution.json', 'w') as f:
        json.dump(selected_statistics, f, indent=4)
        json.dump(selected_tag_distribution, f, indent=4)

if __name__ == "__main__":
    print("Starting process...")
    main(use_transformer=False)
    print("Process completed.")