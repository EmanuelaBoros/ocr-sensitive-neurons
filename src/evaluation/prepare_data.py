import glob
from datasets import load_dataset, Dataset, DatasetDict
import json
import bz2
import os
import ray
from tqdm import tqdm

local_dir = "../../../impresso-semantic-enrichment-deployment/temp_downloads/newspapers/GDL/"  # Set this to your directory containing the .jsonl.bz2 files
columns_to_keep = ["ft", "lg", "id"]


# Function to load data from a single .jsonl.bz2 file
def load_data_from_file(file_path):
    with bz2.open(file_path, "rt") as bz_file:  # Open the file in text read mode
        # Load lines and parse JSON, filter out unnecessary columns
        data = [json.loads(line) for line in bz_file]
        filtered_data = [{key: d[key] for key in columns_to_keep if key in d} for d in data]
    return filtered_data


# Function to load all files and combine them into a single dataset
def load_dataset_from_files(files):
    all_data = []
    for file in tqdm(files, desc="Loading files", total=len(files)):
        file_data = load_data_from_file(file)
        all_data.extend(file_data)
    dataset = Dataset.from_dict({col: [d[col] for d in all_data if col in d] for col in columns_to_keep})
    return dataset


# Function to save a dataset to a CSV file
def save_dataset_to_csv(dataset, filename):
    file_path = os.path.join(data_dir, filename)
    dataset.to_csv(file_path, index=False)
    print(f"Saved {filename}")


def to_huggingface_dataset(ray_dataset):
    # Collecting all pandas dataframes from the Ray dataset
    pandas_dfs = ray_dataset.to_pandas()
    # Converting the pandas dataframe to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(pandas_dfs)
    return hf_dataset


# Use glob to find all the .jsonl.bz2 files in the directory
print(f"Searching for .jsonl.bz2 files in {local_dir}")
archives = glob.glob(f"{local_dir}/*.jsonl.bz2", recursive=True)

local_dir = "../../../impresso-semantic-enrichment-deployment/temp_downloads/newspapers/GDL/"
columns_to_keep = ["ft", "lg", "id"]

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Use glob to find all the .jsonl.bz2 files in the directory
archives = glob.glob(f"{local_dir}/*.jsonl.bz2", recursive=True)

# Read the dataset using Ray
ray_ds = ray.data.read_json(archives)

# Keep only the required columns
ray_ds = ray_ds.map_batches(lambda batch: batch[columns_to_keep], batch_format="pandas")


# # Load all the files into a single Hugging Face Dataset
# print(f"Loading dataset from {len(archives)} files")
# dataset = load_dataset_from_files(archives)
# Convert the Ray dataset to a Hugging Face Dataset


# Convert to Hugging Face Dataset
dataset = to_huggingface_dataset(ray_ds)

print(f"Loaded dataset: {dataset}")
print(f"Number of examples: {len(dataset)}")

# Splitting the dataset into train, test, and dev sets with an 80/10/10 split
train_test_ratio = 0.8
test_dev_ratio = 0.5  # Split the remaining 20% equally into test and dev
train_dataset, test_dev_dataset = dataset.train_test_split(test_size=(1 - train_test_ratio)).values()
test_dataset, dev_dataset = test_dev_dataset.train_test_split(test_size=test_dev_ratio).values()

# Create the directory for storing CSV files if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

print(f"Saving datasets to {data_dir}")
# Save the datasets into CSV files
save_dataset_to_csv(train_dataset, "train.csv")
save_dataset_to_csv(test_dataset, "test.csv")
save_dataset_to_csv(dev_dataset, "dev.csv")

print(f"Train dataset: {len(train_dataset)} records")
print(f"Test dataset: {len(test_dataset)} records")
print(f"Dev dataset: {len(dev_dataset)} records")
