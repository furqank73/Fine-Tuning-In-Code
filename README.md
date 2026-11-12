# Fine-Tuning notebooks

This folder contains seven notebooks demonstrating different ways to adapt Transformer models to text tasks. They cover ordinary fine-tuning, Hugging Face's higher-level training tools, parameter-efficient adapters, LoRA/QLoRA updates, and transferring a model between datasets.

## Notebooks

### 1. [Full-Fine-Tuning.ipynb](./Full-Fine-Tuning.ipynb)

**Goal:** Train a pretrained DistilBERT model to classify IMDb movie reviews as positive or negative.

**What happens:**
1. Load the IMDb data and determine the number of labels.
2. Load DistilBERT's tokenizer and convert review text into token IDs.
3. Use a padding collator and PyTorch data loaders to form batches.
4. Add a sequence-classification head to pretrained DistilBERT.
5. Train with AdamW, evaluate predictions, and save the model and tokenizer.
6. Use the saved model to predict the sentiment of a new review.

**Main idea:** Full fine-tuning updates the pretrained model's parameters as well as its classification head. This is a straightforward starting point, but it can take substantial compute when using all the data and the notebook's configured training duration.

### 2. [HuggingFace_Transformer_Fine_Tuning_Explained (1).ipynb](./HuggingFace_Transformer_Fine_Tuning_Explained%20(1).ipynb)

**Goal:** Demonstrate two distinct Transformer workflows using Hugging Face and PyTorch.

**Workflow A — Yelp review classification:** Fine-tune pretrained BERT to assign one of five ratings to a review. The notebook tokenizes the reviews, creates batches, adds a classification head, and trains it with an explicit PyTorch training loop.

**Workflow B — IMDb masked language modeling:** Further train pretrained BERT on IMDb review text by hiding selected tokens and asking the model to predict them. This uses Hugging Face's `Trainer` and saves the resulting model and tokenizer.

**Main idea:** The first workflow is supervised classification; the second is a language-modeling task, not a positive/negative sentiment classifier. Comparing the manual loop with `Trainer` also shows two different levels of control over training.

### 3. [Pretrained_DistilBERT_Adapter_AGNEWS.ipynb](./Pretrained_DistilBERT_Adapter_AGNEWS.ipynb)

**Goal:** Classify AG News articles into four categories using DistilBERT with adapters.

**What happens:** The notebook tokenizes and batches the news articles, freezes the pretrained DistilBERT blocks, and adds small trainable adapter layers. It trains the adapters and classification head, then evaluates the model and maps class IDs to category names.

**Main idea:** Rather than updating all of DistilBERT, adapter fine-tuning learns a much smaller set of task-specific parameters. This can reduce the number of trainable weights, although the base model still needs to be loaded and used for inference.

### 4. [Pretrained_DistilBERT_Adapter_IMDb.ipynb](./Pretrained_DistilBERT_Adapter_IMDb.ipynb)

**Goal:** Apply the adapter approach to binary IMDb sentiment classification.

**What happens:** The notebook loads IMDb reviews, optionally limits the data for a faster practice run, tokenizes and batches them, and attaches adapters to the pretrained DistilBERT blocks. It trains the adapters and a new classifier, evaluates sentiment predictions, demonstrates a prediction on custom text, and saves the trainable adapter/classifier weights.

**Main idea:** The base DistilBERT weights stay frozen; the adapters and classification head learn the sentiment task. The saved weights are not a complete standalone model: loading them requires the compatible DistilBERT base model and matching model structure.

### 5. [LoRA_PyTorch_and_HuggingFace_Practice.ipynb](./LoRA_PyTorch_and_HuggingFace_Practice.ipynb)

**Goal:** Train an IMDb sentiment classifier using LoRA in two ways: a manual PyTorch implementation and a Hugging Face PEFT workflow.

**What happens:**
1. Load a pretrained DistilBERT model and tokenizer for binary sentiment classification.
2. Create LoRA update matrices for selected attention layers in a manual PyTorch setup.
3. Train the LoRA weights with a custom training loop and compare the result to a full-model update.
4. Repeat the same task using Hugging Face's `LoraConfig` and `Trainer` integration.
5. Use the same model family, dataset, and target layers so the two approaches are directly comparable.

**Main idea:** LoRA learns a small set of low-rank update matrices instead of changing the full pretrained backbone. This keeps fine-tuning lighter and is a common parameter-efficient alternative to full fine-tuning.

### 6. [QLoRA_IMDb_Beginner_Practice.ipynb](./QLoRA_IMDb_Beginner_Practice.ipynb)

**Goal:** Classify IMDb reviews with QLoRA, a quantized version of low-rank adaptation.

**What happens:** The notebook loads a pretrained DistilBERT model in 4-bit precision, keeps the frozen base model quantized, adds small trainable LoRA layers, and trains the adapter weights together with a classification head. It uses PyTorch, Transformers, bitsandbytes, and PEFT to keep the workflow beginner-friendly while still demonstrating an efficient fine-tuning pattern.

**Main idea:** QLoRA reduces the memory footprint of the base model by loading it in lower precision while still learning task-specific adapters. This makes large-model adaptation more practical on limited hardware while preserving most of the pretrained knowledge.

### 7. [Transformer_Fine_Tuning_AG_News_to_IMDB (1).ipynb](./Transformer_Fine_Tuning_AG_News_to_IMDB%20(1).ipynb)

**Goal:** Teach transfer learning by training a Transformer classifier on AG News, then transferring its learned representations to IMDb sentiment classification.

**What happens:**
1. Build a simple vocabulary from AG News training text and encode both datasets with it.
2. Train a Transformer encoder to classify four AG News categories.
3. Create an IMDb classifier with a new two-class output layer and copy over the source model's embeddings and encoder.
4. Compare two IMDb strategies: full fine-tuning (update the encoder and classifier) and classifier-only fine-tuning (freeze the encoder).
5. Compare validation and held-out test results, inspect classification metrics, and try a custom review.

**Main idea:** This is a teaching implementation with its own tokenizer, vocabulary, and Transformer—not a pretrained DistilBERT model. It demonstrates how transferring representations and choosing which layers to update affect a target task.

## Suggested reading order

1. Start with **Full-Fine-Tuning** for the basic classification pipeline.
2. Read **HuggingFace Transformer Fine-Tuning Explained** to compare a manual training loop with `Trainer` and to distinguish classification from masked language modeling.
3. Compare the two **DistilBERT Adapter** notebooks to see the same parameter-efficient method applied to four-class news and binary sentiment tasks.
4. Continue with **LoRA Practice** and **QLoRA Beginner Practice** to see parameter-efficient adaptation in both standard and quantized forms.
5. Finish with **Transformer Fine-Tuning AG News to IMDB** for a source-to-target transfer experiment and a comparison of full versus classifier-only updates.

## Running the notebooks

- Run cells from top to bottom in a Python environment with the notebook's required packages installed. The notebooks use PyTorch, Hugging Face `transformers` and `datasets`, and, depending on the notebook, other utilities such as `scikit-learn`, `accelerate`, or `matplotlib`.
- A working internet connection is generally needed to download datasets and pretrained checkpoints.
- Training time and memory use depend on the dataset limits, sequence length, batch size, epoch count, and available hardware. Start with any sample limits provided in a notebook before increasing the workload.
- Each notebook saves its own model artifacts when its save cells are run. The separate `adapter_weights.pth` file in this folder is an artifact, not a notebook.
