This repository comprises the  SNOMED CT(Systematized Nomenclature of Medicine Clinical Terms) coding of colon pathology reports through prompt engineering (PRW) and Enhanced-Retrieval Augmented Generation (ERAG). This study evaluated PRW and ERAG through multiple LLMs provided by Meta-Llama 2, Llama 3 and Llama 3.1. The PRW consists of five phases
![screenshot](PRW_5_phases.png)
<p align="center"><em>The development of five phases of PRW</em></p>
The user prompts are developed through the five phases as shown in Figure. Each phase is developed based on addressing the error observed in the previous phase. So, in the last phase (<sup>5th</sup>), we have a comprehensive set of prompts that bring out the best of the LLama models in SNOMED coding.

## Fetching LLaMa Models from Meta
To obtain the model weights and tokenizer, visit the **Meta website** and agree to their license terms. Once your request is approved, Meta will send you a signed URL via email. Use this URL when prompted while executing the download script to begin the download. Ensure that `wget` and `md5sum` are installed on your system. Then, run the following command:
```bash
./download.sh
```
Note: The download links are valid for 24 hours and have a limited number of downloads. If you encounter errors such as 403: Forbidden, re-request a fresh link from Meta.

## Setting Up the Docker Container for PRW

Once the models are downloaded, you can proceed with setting up a **Docker container for PRW**. The `Dockerfile` is provided in: PRW_Meta/Dockerfile


### Managing Model Files Efficiently
To optimize memory usage, **move the downloaded model and tokenizer** to a separate directory outside the project directory. Instead of storing them inside the container, dynamically reference them using the `docker run` command.

---

### **Building the Docker Container**
To build the container, navigate to the directory containing the `Dockerfile` and run:

```bash
docker build -t PRW .
```
### **Running the Docker Container**
Run the container while dynamically mounting the model and tokenizer using the -v flag:

```bash
docker run --gpus '"device=6,7"' --rm -it \
    -v /data/Jennil/llama3/llama3/Meta-Llama-3-8B-Instruct/:/mnt/model \
    PRW
```
Replace /`` `/data/Jennil/llama3/llama3/Meta-Llama-3-8B-Instruct/` `` with the actual path where the model and tokenizer are stored. This approach ensures efficient memory usage while keeping the Docker container lightweight.






