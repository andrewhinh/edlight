"""Fine-tune image-to-text model on the data."""
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from utils.create_dataloader import collator
from utils.create_dataset import ImageCaptioningDataset
from utils.eval_model import run_evaluation
from utils.load_data import preprocess_data
from utils.train_model import run_training

# Define constants
DATA_PATH = "data/"
IMAGES_PATH = DATA_PATH + "images/"
SRC_PATH = "descriptions.csv"

TEST_SIZE = 0.1
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

MAX_IMG_SIZE = 224
TRAIN_MAX_PATCHES = 512  # max num of patches per image (that fit in memory)
TEST_MAX_PATCHES = 128
MAX_LENGTH = 100  # max num of tokens in generated description

TRAIN_BS = 2
TEST_BS = 1

MODEL_PATH = "pix2struct-textcaps-base"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
MODEL = Pix2StructForConditionalGeneration.from_pretrained(MODEL_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
LR = 1e-5
MODEL_SAVE_PATH = "pix2struct-textcaps-base-finetuned"

METRIC_NAME = "bleu"

os.environ["TOKENIZERS_PARALLELISM"] = "False"  # To prevent warnings

# Load the data
train_df, test_df = preprocess_data(DATA_PATH, SRC_PATH, IMAGES_PATH, TRAIN_PATH, TEST_PATH, TEST_SIZE)
print(f"Data size: {len(train_df) + len(test_df)}")
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")


# Convert data to dataset
train_dataset = ImageCaptioningDataset(train_df, MAX_IMG_SIZE, PROCESSOR, TRAIN_MAX_PATCHES)
test_dataset = ImageCaptioningDataset(test_df, MAX_IMG_SIZE, PROCESSOR, TEST_MAX_PATCHES)
print(f"Max image size: {MAX_IMG_SIZE}")
print(f"Train max patches: {TRAIN_MAX_PATCHES}")
print(f"Test max patches: {TEST_MAX_PATCHES}")


# Convert dataset to dataloader
train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=TRAIN_BS, collate_fn=lambda x: collator(x, PROCESSOR, MAX_LENGTH)
)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BS, collate_fn=lambda x: collator(x, PROCESSOR, MAX_LENGTH))
print(f"Train batch size: {TRAIN_BS}")
print(f"Test batch size: {TEST_BS}")
print(f"Max text length: {MAX_LENGTH}")


# # Train the model
print(f"Number of epochs: {EPOCHS}")
print(f"Learning rate: {LR}")
print("Training...")
run_training(
    train_dataloader,
    MODEL,
    EPOCHS,
    LR,
    DEVICE,
    MODEL_SAVE_PATH,
)


# Evaluate the model
MODEL = Pix2StructForConditionalGeneration.from_pretrained(MODEL_SAVE_PATH)
print("Evaluating...")
results = run_evaluation(
    test_dataloader,
    MODEL,
    PROCESSOR,
    DEVICE,
    METRIC_NAME,
    MAX_LENGTH,
)
print(results)
