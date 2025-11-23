import pandas as pd
import json
import vertexai
from vertexai.preview.tuning import sft

# Configuration Constants
PROJECT_ID = "your-google-cloud-project-id"
LOCATION = "us-central1"
DATA_FILE_PATH = "../data/medquad.csv"
TUNING_DATA_OUTPUT = "medquad_tuning_dataset.jsonl"
GCS_BUCKET_URI = "gs://your-bucket-name/medquad-tuning/"  # Must exist in your Google Cloud Storage

def prepare_training_data(input_csv, output_jsonl):
    """
    Reads the MedQuAD CSV and converts it into JSONL format required by Vertex AI
    for supervised fine-tuning.
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Ensure required columns exist (adjust column names based on actual CSV headers)
    # Assuming columns are 'question' and 'answer' based on MedQuAD standard
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("CSV must contain 'question' and 'answer' columns.")

    print("Processing records...")
    with open(output_jsonl, "w") as f:
        for _, row in df.iterrows():
            # Construct the chat message structure for Gemini
            tuning_entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful medical assistant providing accurate answers based on clinical data."
                    },
                    {
                        "role": "user",
                        "content": row['question']
                    },
                    {
                        "role": "model",
                        "content": row['answer']
                    }
                ]
            }
            f.write(json.dumps(tuning_entry) + "\n")
    
    print(f"Data preparation complete. File saved to: {output_jsonl}")

def start_fine_tuning_job(local_file, bucket_uri):
    """
    Uploads the dataset to GCS and starts a Supervised Fine-Tuning job on Vertex AI.
    """
    # Initialize Vertex AI SDK
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Note: For this script to work, you must upload the local_file to the bucket_uri manually 
    # or use the google-cloud-storage library to automate it.
    # Assuming the file is already at: {bucket_uri}{local_file}
    gcs_path = f"{bucket_uri}{local_file}"
    
    print(f"Starting Fine-Tuning job using data from: {gcs_path}")
    
    # Start the training job
    sft_tuning_job = sft.train(
        source_model="gemini-1.5-flash-002",
        train_dataset=gcs_path,
        epochs=2,  # 2-3 epochs are usually sufficient for text adaptation
        adapter_size=4,
        tuned_model_display_name="gemini-medquad-expert"
    )
    
    print("Training job submitted. Monitor progress in the Google Cloud Console.")
    # The job will return a tuned model endpoint once finished.
    return sft_tuning_job

if __name__ == "__main__":
    # 1. Prepare Data
    prepare_training_data(DATA_FILE_PATH, TUNING_DATA_OUTPUT)
    
    # 2. Trigger Fine-Tuning (Uncomment below if you have set up GCS and want to run it)
    # start_fine_tuning_job(TUNING_DATA_OUTPUT, GCS_BUCKET_URI)