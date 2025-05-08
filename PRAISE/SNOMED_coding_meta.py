from llama import Llama
import fire
import gc
from typing import Optional


def generate_summary(generator, user_input: str, max_gen_len: Optional[int], temperature: float, top_p: float, max_seq_len: int) -> str:
    """Generate a summary from the given pathology report."""
    system_content = (
        "You are a highly skilled pathologist assistant tasked with accurately identifying the morphology and topography of the primary tumor. Tumor morphology refers to the histological classification of the tumor, while topography specifies the anatomical site of the tumor. Your expertise in analyzing detailed medical reports will ensure precise identification of these attributes. "
            "Pathologic diagnosis: 1. Colon, 60 cm from anal verge, A, polypectomy --- Tubulovillous  adenoma. 2. Colon, 40 cm from anal verge, B, polypectomy --- Tubular adenoma. 3. Colon, 20 cm from anal verge, C, polypectomy --- Tubular adenoma. Prognostic and predictive factor: 1. Degree of dysplasia: Low grade. 2. Margin of stalk: Free in B./ Involved in A; Cannot be well evaluated in C. Gross description: The specimen consists of 1) a piece of tan soft tissue, 1.1x0.7x0.4 cm, labeled as 60 cm. 2) a piece of tan soft tissue, 0.6x0.4x0.2 cm, labeled as 40 cm. 3) a piece of tan soft tissue, 0.2x0.2x0.1 cm, labeled as 20 cm. All for section: A) specimen A B) specimen B C) specimen C. Microscopic description: Section A shows colon mucosa with tubulovillous glands proliferation with dysplasia. Sections B and C show colon mucosa with tubular glands proliferation with dysplasia. Summarised the givien pathology report based on fidning morphology and topogrpahy of the seriosu tumor."
    )
    user_content = f"Summarize the given pathology report based on finding of  morphology and topography of a serious tumor. Report: {user_input}"

    # Tokenize and process the input
    system_tokens = generator.tokenizer.encode(system_content, bos=True, eos=True)
    user_tokens = generator.tokenizer.encode(user_content, bos=True, eos=True)

    if len(system_tokens) + len(user_tokens) > max_seq_len:
        print("Input exceeds the maximum sequence length. Please provide a shorter input.")
        return ""

    dialog = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    response = generator.chat_completion(
        [dialog],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    summary = response[0]["generation"]["content"]
    print("\nGenerated Summary:")
    print(summary)
    return summary


def assign_snomed(generator, summary: str, max_gen_len: Optional[int], temperature: float, top_p: float, max_seq_len: int) -> str:
    """Assign SNOMED codes based on the generated summary."""
    system_content = """
    Topography Codes:
1. 67000	COLON, NOS
2. 67100	CECUM
3. 67200	ASCENDING COLON
4. 67400	TRANSVERSE COLON
5. 67600	DESCENDING COLON
6. 67700	SIGMOID COLON
7. 67800	MESENTERY OF COLON, MESOCOLON
8. 67950	COLON AND SKIN, CS
9. 67965	COLON, RIGHT
10. 67995	COLON, LEFT
11. 68000	RECTUM, NOS
13. 69000	ANUS, NOS
14. 69900	ANORECTUM, CS
15. 64000 	SMALL INTESTINE


Distance to Topography Mapping:
f the site of the tumour is unclear and the report does not mention the topography explicitly but includes the distance of the site from the anal verge, use the Distance to Topography Mapping provided below:
0–4 cm: Anus
4–15 cm: Rectum
15–17 cm: Rectosigmoid Junction
17–57 cm: Sigmoid Colon
57–82 cm: Descending Colon
82–132 cm: Transverse Colon
132–147 cm: Ascending Colon
150 cm: Cecum
For example, If the primary tumor is located 40 cm from the anal verge, then the specific location can be identified from the Distance to Topography Mapping as the Sigmoid Colon.

Important Notes:

1. If the tumor's location spans multiple anatomical sites, classify the topography based on the broader regional location (e.g., the side of the colon) rather than individual sites. For instance, when the tumor location is described as "Rectum and Sigmoid colon," "Rectum to Sigmoid colon," or "Rectum-Sigmoid colon," these indicate involvement of multiple sites within the left side of the colon. In such cases, assign the topography to "Left Colon," corresponding to the code 67995. Similarly, if the tumor spans locations on the right side of the colon, such as the ascending and transverse colon, classify it under "Right Colon." This approach ensures accurate representation of the tumor's regional extent.
2. For distances from the anal verge, accurately map them to the provided distance-to-topography table. For example, "50 cm from the anal verge" maps to Sigmoid colon (17–57 cm).
3. Accuracy in Identification: Ensure you do not confuse the identified topography and the codes. Example: If you identify the topography as CECUM, return 67100, not 67000 (COLON, NOS).
4. Keep your response short and concise. So return in one line like "The topography  is " ----- "and its SNOMED code is"----

    """
    user_content = f"You have been provided with a summarized pathology report highlighting the tumor's detail. Using the SNOMED codebook provided, your task is to assign the most appropriate SNOMED codes for the topography, remember only topogrpahy. Summary: {summary}"

    # Tokenize and process the input
    system_tokens = generator.tokenizer.encode(system_content, bos=True, eos=True)
    user_tokens = generator.tokenizer.encode(user_content, bos=True, eos=True)

    if len(system_tokens) + len(user_tokens) > max_seq_len:
        print("Input exceeds the maximum sequence length.")
        return ""

    dialog = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    response = generator.chat_completion(
        [dialog],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    snomed_codes = response[0]["generation"]["content"]
    print("\nAssigned SNOMED Codes:")
    print(snomed_codes)
    return snomed_codes


def assign_mcode(generator, summary: str, max_gen_len: Optional[int], temperature: float, top_p: float, max_seq_len: int) -> str:
    """Assign Mcode (Morphology Codes) based on the summary."""
    system_content = """
    I am providing you a finding about a tumor. Your task is to return the assign SNOMED code from the provided list. Carefully read the findings, determine the morphology of the tumor, and follow the instructions step-by-step before assigning the code.
Instructions:


Step 1: Identify the Morphology 
1. In-Situ Tumors:
   - Intramucosal adenocarcinoma is also considered in-situ because it is confined to the mucosal layer and does not invade beyond it. 
   - For any malignant tumor, even for in-situ, consider the polyp also where it arise if there is any, while defining the morphology. For example, if the tumor morphology is adenocarcinoma in-situ arising in tubulovillous adenoma, then we should focus both on "adenocarcinoma in-situ" and "tubular adenoma"  


2. Additional Descriptive Features:
   - If the report mentions features or components like "signet ring cells," "mucinous component," or "mucinous product" as part of the tumor description, do not classify the morphology based on these features unless they are explicitly stated as the main tumor type.
   - Always assign the Mcode based on the primary morphology term mentioned first in the report (e.g., “Adenocarcinoma” in “Adenocarcinoma with mucinous component”). Treat additional descriptions or cell features as secondary, not as the main morphology.

3. High-Grade Dysplasia:
   - High-grade dysplasia is not cancer.
   - If the report states “high-grade dysplasia” without mentioning invasive carcinoma, classify the tumor as benign.
   - Only classify as malignant if the report explicitly mentions adenocarcinoma or invasive carcinoma.

Step 2: Assign the Correct Mcode
- After identifying the morphology based on the rules above, look up the corresponding Mcode from the provided list and assign it.
- Important: Do not skip any steps or modify any parts of the morphology description.



Final Notes:
- Do not perform any tasks beyond the specified objectives. Focus on assigning morphology code only.
- When answering, do not hallucinate or make assumptions.


The Mcode list and their corresponding Morphology is provide below:

Mcode   Morphology
M80102: CARCINOMA IN SITU
M80103: CARCINOMA, NOS
M80203: UNDIFFERENTIATED CARCINOMA
M80702:	Squamous cell carcinoma in situ, NOS
M80703:	Squamous cell carcinoma, NOS
M80706:	Squamous cell carcinoma, NOS, metastatic
M80709:	Squamous cell carcinoma, NOS, unknown if primary or metastatic
M81402: ADENOCARCINOMA IN-SITU
M81403: ADENOCARCINOMA, NOS
M82103: ADENOCARCINOMA IN ADENOMATOUS POLYP
M82203: ADENOCA ARISING FROM ADENOMATOUS POLYP
M82403: CARCINOID TUMOR, MALIGNANT
M82603: PAPILLARY ADENOCARCINOMA
M82613: ADENOCARCINOMA IN VILLOUS ADENOMA
M82632: Adenocarcinoma in situ in tubulovillous adenoma
M82636:	Adenocarcinoma in tubulovillous adenoma, metastatic
M82633: ADENOCARCINOMA IN TUBULOVILLOUS ADENOMA
M83803: ENDOMETRIOID CARCINOMA
M84103: SEBACEOUS CARCINOMA
M84803: MUCINOUS ADENOCARCINOMA
M84903: SIGNET RING CELL CARCINOMA
M88403: MYXOSARCOMA, MALIGNANT MYXOMA
M89303: STROMAL SARCOMA, ENDOMETRIAL STROMAL SARCOMA
M82630: TUBULOVILLOUS ADENOMA
M82110: TUBULAR ADENOMA
M43000: CHRONIC INFLAMMATION
M09450: NO EVIDENCE OF MALIGNANCY
M81406: ADENOCARCINOMA, METASTATIC
M81407: ADENOCARCINOMA, RECURRENT
M81404: ADENOCARCINOMA, CONTIGUOUS SPREAD
M82100: ADENOMATOUS POLYP
M82101:	Adenomatous polyp, NOS, uncertain, borderline
M88500: LIPOMA, NOS
M85603: ADENOSQUAMOUS CARCINOMA
M80213: ANAPLASTIC CARCINOMA
M82600: PAPILLARY ADENOMA
M84701: MUCINOUS CYSTADENOMA, BORDERLINE MALIGNANCY
M84030: ECCRINE SPIRADENOMA     
M82401: CARCINOID TUMOR, NOS
M82610: VILLOUS ADENOMA, NOS
M82611:	Villous adenoma, NOS, uncertain, borderline
M82612:	Adenocarcinoma in situ in villous adenoma
M81703	HEPATOCELLULAR CARCINOMA, HEPATOMA
M81409	ADENOCARCINOMA, 1' OR 2'
M84804	MUCINOUS ADENOCARCINOMA, CONTIGUOUS SPREAD
M81400	ADENOMA, NOS
M84807	MUCINOUS ADENOCARCINOMA, RECURRENT
M88900	LEIOMYOMA, NOS, FIBROMYOMA		
M82113	ADENOCARCINOMA IN TUBULAR ADENOMA 
M82112  ADENOCARCINOMA IN SITU IN TUBULAR ADENOMA
M80106	CARCINOMA, METASTATIC				
M84416	SEROUS CYSTADENOCARCINOMA, METASTATIC		
M81405	ADENOCARCINOMA, MICROINVASIVE	
M80127	LARGE CELL CARCINOMA, RECURRENT	
M85103	MEDULLARY CARCINOMA	
M82040	Lactating adenoma
M87402	 melanoma in junctional nevus in situ, noninfiltrating, noninvasive

Keep your response short and concise. So return in one line like "The morphology is " ----- "and its SNOMED code is"----
	
    """
    user_content = f"The finding is: {summary}"

    # Tokenize and process the input
    system_tokens = generator.tokenizer.encode(system_content, bos=True, eos=True)
    user_tokens = generator.tokenizer.encode(user_content, bos=True, eos=True)

    if len(system_tokens) + len(user_tokens) > max_seq_len:
        print("Input exceeds the maximum sequence length.")
        return ""

    dialog = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    response = generator.chat_completion(
        [dialog],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    mcode = response[0]["generation"]["content"]
    print("\nAssigned Mcode:")
    print(mcode)
    return mcode


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 8192,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None,
):

    # Load the model once
    print("Loading model...")
    generator = Llama.build(
	ckpt_dir=ckpt_dir,
	tokenizer_path=tokenizer_path,
	max_seq_len=max_seq_len,
	max_batch_size=max_batch_size,
    )
    print("Model loaded successfully.")

    while True:
        print("\nEnter your pathology report (or type 'exit' to quit):")
        user_input = input("> ")

        if user_input.lower() == "exit":
            print("Exiting the program.")
            break

        # Step 1: Generate summary
        summary = generate_summary(generator, user_input, max_gen_len, temperature, top_p, max_seq_len)
        if not summary.strip():
            print("Summary generation failed. Please try again.")
            continue

        # Step 2: Assign SNOMED codes
        snomed_codes = assign_snomed(generator, summary, max_gen_len, temperature, top_p, max_seq_len)

        # Step 3: Assign Mcode
        assign_mcode(generator, summary, max_gen_len, temperature, top_p, max_seq_len)

        # Release GPU memory
        gc.collect()
        print("Completed processing.\n")

if __name__ == "__main__":
    try:
        fire.Fire(main)
    except SystemExit:
        # Gracefully exit without an error
        print("Program exited successfully.")
