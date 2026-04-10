# Vexoo Labs AI Engineer Assignment

## Setup
pip install transformers datasets peft torch

## Part 1
Run:
python ingestion.py

Input:
- sample.txt (your document)

Output:
- Relevant retrieved response

## Part 2
Run:
python train.py

Details:
- Dataset: GSM8K
- Train: 3000 samples
- Test: 1000 samples
- Model: LLaMA 3.2 1B (LoRA fine-tuning)

## Notes
- Summarization and embeddings are simulated
- Focus is on architecture and pipeline design
