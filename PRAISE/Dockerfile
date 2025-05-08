# Use a base image with CUDA support
#FROM pytorch/pytorch:1.10.0-cuda11.1-cudnn8-runtime

FROM pytorch/pytorch:latest



# Set the working directory in the container
WORKDIR /app

# Copy the entire llama-main directory into the container
COPY . .

RUN pip install -e .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install memory_profiler

# Install dependencies from requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to your script
CMD torchrun --nproc_per_node 1 /app/PRW_Meta/SNOMED_coding_meta.py \
     #--ckpt_dir /mnt/model/llama-2-7b-chat \
     --ckpt_dir /mnt/model \
     --tokenizer_path /mnt/model/tokenizer.model \
     #--tokenizer_path tokenizer.model \
     --data_dir /app/Re-check \
     --output_csv_dir /app/output \
     --max_seq_len 8192 \
     --max_batch_size 4

#CMD python Input_handler.py \
#    --ckpt_dir /mnt/model \
#    --tokenizer_path /mnt/model/tokenizer.model \
#    --max_seq_len 8192 \
#    --max_batch_size 4


#CMD torchrun --nproc_per_node 2 /app/example_chat_inccorporation_code_ICD_topography.py \
#    --ckpt_dir /data/Jennil/llama-2_docker_13B/llama-main/llama-2-13b-chat \
#    --tokenizer_path /data/Jennil/llama-2_docker_13B/llama-main/tokenizer.model \
#    --max_seq_len 4096 \
#    --max_batch_size 4
