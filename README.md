# Fine-Tuning Phi-2 for Abstractive Text Summarization

## Group 7 – Deep Learning

---

## Overview

This project focuses on fine-tuning **Microsoft Phi-2**, a decoder-only causal language model, for **abstractive text summarization** using the **XSum (Extreme Summarization)** dataset.  
The implementation adopts an **instruction-based prompting** strategy combined with **parameter-efficient fine-tuning (QLoRA)** to enable training and experimentation under limited computational resources.

All stages of the workflow—including preprocessing, training, evaluation, and qualitative analysis—are implemented within a single Jupyter Notebook.

---
## Pre-trained Model

The fine-tuned model artifacts (LoRA adapters and tokenizer) are available at the following link:

[File Model](https://drive.google.com/drive/folders/1Aap1rQI_IEZEnLqCPF7THswW6ivVSRD_)

Only the adapter weights and tokenizer files are provided; the Phi-2 base model must be loaded separately during inference.

---
## Objectives

- Apply instruction-tuned causal language modeling for summarization  
- Perform memory-efficient fine-tuning using QLoRA  
- Generate concise, one-sentence abstractive summaries  
- Evaluate model performance quantitatively using ROUGE metrics  
- Analyze qualitative strengths and limitations of generated summaries  

---

## Model and Dataset

### Model
- **Base Model**: microsoft/phi-2  
- **Architecture**: Decoder-only Transformer (Causal LM)  
- **Parameter Count**: ~2.7B  
- **Fine-Tuning Method**: QLoRA (4-bit quantization with LoRA adapters)

### Dataset
- **Dataset**: EdinburghNLP / XSum  
- **Task Type**: Extreme abstractive summarization  
- **Target Output**: Single-sentence summaries  

---

## Methodology

### Instruction-Based Fine-Tuning
Each article is reformatted into an instruction-style prompt that explicitly asks the model to produce a concise summary.  
Training is performed using causal language modeling, where loss is calculated **only on the summary tokens**, while instruction tokens are masked.

### Parameter-Efficient Training
To ensure efficiency and reduce overfitting:
- 4-bit NF4 quantization is applied
- Only LoRA adapter parameters are trained
- Gradient accumulation and gradient checkpointing are used

This setup allows stable training on GPUs with limited VRAM.

---

## Training Configuration

| Parameter | Value |
|---------|------|
| Training Samples | 1000 |
| Validation Samples | 100 |
| Epochs | 2 |
| Batch Size | 2 |
| Gradient Accumulation | 8 (effective batch size = 16) |
| Learning Rate | 2e-4 |
| Scheduler | Cosine |
| Optimizer | Paged AdamW (8-bit) |
| Max Sequence Length | 1024 |
| Precision | FP16 |

---

## Evaluation Results

Evaluation was conducted on **100 validation samples** using ROUGE metrics.

| Metric | Score |
|------|------|
| **ROUGE-1** | **32.08%** |
| **ROUGE-2** | **11.85%** |
| **ROUGE-L** | **24.66%** |
| **ROUGE-Lsum** | **23.87%** |

### Interpretation
- **ROUGE-1** indicates strong coverage of core content  
- **ROUGE-2** reflects reasonable phrase-level coherence  
- **ROUGE-L / ROUGE-Lsum** show that the generated summaries preserve overall structure  

Overall, these results demonstrate good generalization performance for a decoder-only model fine-tuned on a limited subset of XSum.

---

## Qualitative Analysis

Qualitative evaluation shows that the model performs very well on articles describing a **single, well-defined event**, producing concise and factually consistent summaries.  
Performance decreases on articles whose reference summaries are **conceptual or meta-level**, where the model tends to generate descriptive summaries instead of abstract interpretations.

This behavior is consistent with known challenges of abstractive summarization on the XSum dataset and does not indicate overfitting.

---

## Repository Structure
├── finetuning-phi-2-text-summarization.ipynb # Main notebook

├── model_trained/ # Fine-tuned LoRA adapters and tokenizer

├── requirements.txt/ # Requirements for running this program

└── README.md # Project documentation

---


## Notes

- The saved model contains LoRA adapters only, not full model weights  
- Results may vary depending on generation parameters and evaluation subsets  
- This project is intended for educational and research purposes  

---

## Conclusion

This project demonstrates that instruction-tuned fine-tuning of a decoder-only language model using QLoRA can achieve strong abstractive summarization performance on the XSum dataset, even with limited training data and computational resources.


