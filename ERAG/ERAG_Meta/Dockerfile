# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:latest

# Set the working directory inside the container
WORKDIR /app

# Copy all project files, including LLaMa and RAG scripts
COPY . .

# Install required dependencies
RUN pip install -e .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install memory_profiler

# Additional Dependencies for RAG
RUN pip install --no-cache-dir \
    pandas \
    torch \
    faiss-cpu \
    langchain \
    langchain-community \
    sentence-transformers \
    fire  # Required for LLaMa CLI

# Set the entry point to execute the RAG script using Meta's LLaMa
#CMD python3 /app/RAG_meta/RAG_meta.py \
#     --ckpt_dir /mnt/model \
#     --tokenizer_path /mnt/model/tokenizer.model \
#     --max_seq_len 8192 \
#     --max_batch_size 4

CMD torchrun --nproc_per_node 1 /app/RAG_meta/RAG_meta.py \
     #--ckpt_dir /mnt/model/llama-2-7b-chat \
     --ckpt_dir /mnt/model \
     --tokenizer_path /mnt/model/tokenizer.model \
     #--tokenizer_path tokenizer.model \
     #--data_dir /app/Re-check \
    # --output_csv_dir /app/output \
     --max_seq_len 8192 \
     --max_batch_size 4
