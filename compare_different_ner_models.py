""""
This script is designed for evaluating Named Entity Recognition (NER) models, 
particularly focusing on a comparative analysis between a baseline model and a newly proposed model. 
It employs the WNUT2017 dataset to provide a consistent evaluation framework, 
ensuring that both models are assessed under the same conditions.

The script operates by generating NER predictions from each model, 
followed by a meticulous process of aligning and mapping these predictions to 
the standardized tagging scheme of the dataset. This is crucial for addressing the variances 
in tokenization between the models' outputs and the actual data annotations, with a special 
focus on correctly handling subtokens and other tokenization nuances.

A significant feature of this evaluation is the manual mapping of model-specific entity tags 
to those in the dataset, facilitating an apples-to-apples comparison in the subsequent analysis. 
The core of the evaluation lies in computing key metrics such as precision, recall, and the F1 score for both models, 
alongside generating a comprehensive classification report that offers a detailed breakdown of each model's 
performance across different entity types.

This methodical approach allows for an in-depth comparative study, illuminating 
the strengths and potential areas for improvement in the NER capabilities of each model. 
Through this script, users can gain valuable insights into the relative performance of the models, 
guiding future developments and refinements in NER model implementations.
"""

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# Load the WNUT2017 test dataset
wnut2017_dataset = load_dataset("wnut_17", split='test')

# Mapping dictionary for entity tags
entity_mapping = {
    'LOC': 'LOCATION',
    'ORG': 'ORGANIZATION',
    'PER': 'PERSON',
    # Add more mappings as needed
}

manual_sub_mappings = {
    'O': 'O',
    'B-MISC': 'O',  # Assuming no direct match, defaulting to 'O'
    'I-MISC': 'O',  # Assuming no direct match, defaulting to 'O'
    'B-PER': 'B-person',
    'I-PER': 'I-person',
    'B-ORG': 'B-corporation',  # Assuming 'corporation' is a suitable match for 'ORG'
    'I-ORG': 'I-corporation',
    'B-LOC': 'B-location',
    'I-LOC': 'I-location'
}

def compute_metrics(predictions, labels):
    """
    Compute precision, recall, and F1 metrics for NER predictions.

    Args:
        predictions (list): List of predicted labels.
        labels (list): List of true labels.

    Returns:
        A dictionary with precision, recall, and F1-score.
    """
    # Convert y_pred and y_label to list of lists format
    flat_predictions = [[label for label in seq] for seq in predictions]
    flat_labels = [[label for label in seq] for seq in labels]

    # Compute precision, recall, and F1 score
    precision = precision_score(flat_labels, flat_predictions)
    
    recall = recall_score(flat_labels, flat_predictions)
    f1 = f1_score(flat_labels, flat_predictions)

    # Optionally, generate a classification report
    report = classification_report(flat_labels, flat_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }

def merge_entities(predicted_entities):
    # Merge consecutive 'I-' entities with their corresponding 'B-' entity and process tags
    merged_entities = []
    current_entity = None
    for word, label in predicted_entities:
        if label.startswith('B-'):
            # Start a new entity
            if current_entity:
                merged_entities.append(current_entity)
            current_entity = (word, label.replace('B-', '').upper())  # Remove 'B-' and process tag
        elif label.startswith('I-'):
            # Append to the current entity and process tag
            if current_entity:
                current_entity = (current_entity[0] + word.replace('##', ''), current_entity[1])  # Merge words and remove '##'
        else:
            # Non-entity token, ignore
            pass

    # Append the last entity if exists
    if current_entity:
        merged_entities.append(current_entity)
    # Convert abbreviated entity tags to full forms using the mapping dictionary
    merged_entities_mapped = []
    for word, tag in merged_entities:
        if tag in entity_mapping:
            tag = entity_mapping[tag]
        merged_entities_mapped.append((word, tag))

    return merged_entities_mapped

def align_tokens(actual_tokens, pre_tokens, pre_labels):
    aligned_labels = ['O'] * len(actual_tokens)  # Initialize with 'O' labels
    pre_index = 0  # Index for predicted tokens

    for actual_index, actual_token in enumerate(actual_tokens):
        if pre_index >= len(pre_tokens):
            break  # Stop if we've processed all predicted tokens

        reconstructed_token = ""
        while pre_index < len(pre_tokens):
            part_of_token = pre_tokens[pre_index].replace("##", "")
            reconstructed_token += part_of_token

            # Move to the next predicted token if it's part of the current actual token
            if actual_token.startswith(reconstructed_token):
                if actual_token == reconstructed_token:
                    aligned_labels[actual_index] = pre_labels[pre_index]
                    pre_index += 1  # Only move to the next predicted token if we fully matched
                    break
                pre_index += 1
            else:
                # This handles cases where the predicted tokenization moves beyond the current actual token
                break

    return aligned_labels

def map_model_tags_to_dataset_tags(predicted_tags, mapping):
    return [mapping.get(tag, 'O') for tag in predicted_tags]

# Function to create a mapping between model tags and dataset tags
def create_mapping(model_tags, dataset_tags):
    mapping = {}
    
    # Create a simplified mapping of just the entity types to full dataset tags
    entity_to_dataset_tag = {}
    for tag in dataset_tags:
        if '-' in tag:
            entity = tag.split('-')[-1]
            entity_to_dataset_tag[entity] = tag

    # Map model tags to dataset tags
    for tag in model_tags:
        if tag == 'O':
            mapping[tag] = 'O'
            continue

        prefix, entity = tag.split('-')
        
        # Find the corresponding dataset tag with the same entity
        if entity in entity_to_dataset_tag:
            mapped_tag = entity_to_dataset_tag[entity]
            # Ensure the prefix in the dataset tag matches the model tag
            if mapped_tag.startswith(prefix):
                mapping[tag] = mapped_tag
            else:
                # If no exact match, use the 'O' tag
                mapping[tag] = 'O'
        else:
            mapping[tag] = 'O'

    return mapping

def evaluate_model(model_name, dataset):
    # Load the model and tokenizer using the pipeline
    nlp = pipeline("ner", model=model_name, tokenizer=model_name)

    predictions = []
    labels = []

    for example in dataset:
        tokens = example['tokens']
        text = " ".join(tokens) if isinstance(tokens, list) else tokens
        ner_results = nlp(text)

        # Tokenize the text using the model's tokenizer to get a list of tokens that should align with ner_results
        tokenized_text = nlp.tokenizer.tokenize(text)

        # Initialize predicted labels with 'O'
        pre_labels = ['O'] * len(tokenized_text)
        
        # Update predicted labels based on ner_results
        for entity in ner_results:
            # Adjust the index to align with tokenized_text
            index = entity['index'] - 1  # Adjusting the index to match the tokenized text
            entity_word = entity['word'].replace("##", "")
            
            # Find the start index of the entity word in the tokenized text
            for i in range(index, min(index + 2, len(tokenized_text))):
                if tokenized_text[i].endswith(entity_word):
                    pre_labels[i] = entity['entity']
                    break

        # Align and map the predicted labels to the dataset labels
        
        aligned_pre_labels = align_tokens(tokens, tokenized_text, pre_labels)
        mapped_pre_labels = map_model_tags_to_dataset_tags(aligned_pre_labels, manual_sub_mappings)

        predictions.append(mapped_pre_labels)
        actual_labels = [dataset.features['ner_tags'].feature.int2str(tag) for tag in example['ner_tags']]
        labels.append(actual_labels)
    metrics = compute_metrics(predictions, labels)
    return metrics

def main():
    # Model names
    models = [
        'dslim/bert-base-NER',
        'dslim/bert-base-NER-uncased',
        'Babelscape/wikineural-multilingual-ner',
        'tner/roberta-large-wnut2017'
    ]

    # Evaluate each model
    for model in models:
        metrics = evaluate_model(model, wnut2017_dataset)
        print(f"Metrics for {model}: {metrics}")

if __name__ == "__main__":
    main()
