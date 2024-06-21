from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Initialize the BigQuery client
client = bigquery.Client(project="assemblyai-nlp")

# List of segments to process
segments = ['email', 'phone_number', 'address', 'alphanumeric', 'website']

# Base dataset and project ID
base_dataset = "youtube_scrape_2024_04_18"

for segment in segments:
    table_id = f"{base_dataset}.filtered_{segment}_10hrs_youtube_usm_scraped_dataset_w_Roberta_tags"
    new_table_id = f"{table_id}_w_HappyScribe_transcript"
    consolidated_table_id = f"{base_dataset}.consolidated_benchmarking_dataset_happy_scribe_transcribed"

    # Step 1: Create new table copying existing one
    create_table_query = f"""
    CREATE OR REPLACE TABLE `{new_table_id}` AS
    SELECT * FROM `{table_id}`;
    """
    client.query(create_table_query).result()

    # Step 2: Add new column to the newly created table
    try:
        table = client.get_table(new_table_id)
        schema = list(table.schema)
        schema.append(bigquery.SchemaField("HappyScribed_transcript", "STRING"))
        table.schema = schema
        client.update_table(table, ["schema"])
    except NotFound:
        print(f"Table {new_table_id} not found.")

    # Step 3: Merge data into the new table
    merge_query = f"""
    MERGE INTO `{new_table_id}` AS original
    USING `{consolidated_table_id}` AS consolidated
    ON original.id = consolidated.id
    WHEN MATCHED THEN
      UPDATE SET original.HappyScribed_transcript = consolidated.transcript_gcs_uri;
    """
    client.query(merge_query).result()

    print(f"Processed segment: {segment}")