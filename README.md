You have two solution one  based on DSE and the other on the colpali. 
I build everything with local models in dual gpu setup if you have one GPU Vram larger than 24 you can run the both models on one gpu. 
****

DSE Notebook: Multimodal Document Retrieval and Generation
Overview
This notebook implements a multimodal system for document retrieval and question answering using VDRE (Vision-Document Retrieval Embedding) for image/text embeddings and Qwen2.5-VL for vision-language generation. It processes text and image queries to retrieve relevant document images and generate answers, evaluating retrieval (Precision@1, Recall@1) and generation (ROUGE-L, SBERT).
Requirements

Python 3.19+

Setup

Install dependencies:pip install -r requirements.txt


Download NLTK data:import nltk
nltk.download('punkt')


Ensure CUDA-enabled GPU for VDRE (cuda:1) and Qwen2.5-VL (cuda:0).

Usage

Run the notebook to:
Load images and generate embeddings using VDRE.
Store embeddings in a Qdrant vector store.
Process 5 text queries and 1 image query (./images/similar.png).
Retrieve images and generate answers using Qwen2.5-VL.


Evaluate:
Retrieval: Precision@1, Recall@1 (target: 0.8–1.0).
Generation: ROUGE-L F1 (target: 0.3–0.5), SBERT (target: 0.8–0.9).



Evaluation

Retrieval: Measures if the top-1 retrieved image matches the ground truth (e.g., page4.png for attention queries).
Generation: Assesses answer quality using ROUGE-L (content overlap) and SBERT (semantic similarity).
Example output:Retrieval Metrics: {'Precision@1': 1.0, 'Recall@1': 1.0}
Generation Metrics: {'Average ROUGE-L F1': 0.35, 'Average SBERT Score': 0.80}



Notes

Ensure ./data_images contains relevant document images.
Update image_query_path if similar.png is relocated.
Improve generation by refining VLM prompts or fine-tuning Qwen2.5-VL.

