from llama import Llama
import fire
import gc
from typing import Optional


def generate_summary(generator, user_input: str, max_gen_len: Optional[int], temperature: float, top_p: float, max_seq_len: int) -> str:
    """Generate a summary from the given pathology report."""
    system_content = """
        You are a clinical language model specialized in pathology. You are highly skilled at summarizing pathology reports with clarity, precision, and clinical relevance.
	Any given pathology report contains other information which are not needed while finding the topography and morphology of the tumor.
	So, summarize the given report from the perspective of finding the topography and morphology of the serious tumor.  Ensure that the summary contains only relevant information.
	Consider this example of pathology report, “Pathologic diagnosis: Ascending colon, 80 cm from anal verge, endoscopic biopsy --- Tubulovillous adenoma with some atypical glands and cells clusters.
	See Comment. Ancillary study for diagnosis: 1. Deep cut is done. 2. IHC stain for AE1/AE3 (highlight cell clusters) is done. Gross description: The specimen consists of a piece of tan soft tissue, 
	0.1x0.1x0.1 cm. All for section. Microscopic description: Sections show colon mucosa with superficial tubulovillous glands proliferation with high-grade dysplasia. There are some atypical glands and 
	cell clusters seen. Comment: The possibility of carcinoma arising in tubulovillous adenoma is highly suspicious; however, this specimen is too superficial to make a definited pathological diagnosis. 
	Further clinicopathological correlation and study are suggested. Note: The diagnosis is concurred in intradepartment consensus meeting on 2014/03/07.”
	This report has various information, but we need only a summary with respect to the morphology and topography of the serious tumour. 
	So, the summary of this report must have information about a serious tumor, its morphology and topography.  For this particular example, the user is expecting:
	The serious tumor has the morphology of Tubulovillous adenoma, and its topography or location is 80 cm from the anal verge.
        Important note:
	You sometimes tend to provide the wrong response. We analysed your error patterns and identified the following areas for improvement. Follow these instructions carefully to avoid making further errors:
	1. Do not infer or hallucinate information. Use only what is explicitly stated in the report. If unclear, follow the provided instructions for handling such cases.
	2. Verify the extracted morphology and topography before assigning a code. Ensure the assigned code aligns with the extracted information.
    """
    user_content = f"Summarised the given pathology report based on finding the single most serious tumour and its morphology and topography: {user_input}"

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
    """Assign Tcode (Topogrpahy Codes) based on the summary."""
    system_content = """
	You are an expert SNOMED coding assistant. When provided with the topography of a case, you accurately assign the most precise SNOMED codes with confidence and consistency.
	If the summarised report described the  topography or location of a tumor is 60 cm from the anal verge, then the toporgaohy is Descending colon. So, the SNOMED code for descending colon is 67200
	You have been provided with a summarized pathology report highlighting the tumor's topography. Using the SNOMED codebook provided, your task is to assign the most appropriate SNOMED codes
	for topography accurately.
 
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
	12. 64000 	SMALL INTESTINE


	Distance to Topography Mapping:
	If the site of the tumor is unclear and the report does not mention the topography explicitly but includes the distance of the site from the anal verge, 
	use the Distance to Topography Mapping provided below:
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
	
	1. If the tumor's location spans multiple anatomical sites, classify the topography based on the broader regional location (e.g., the side of the colon) rather than individual sites. 
	For instance, when the tumor location is described as "Rectum and Sigmoid colon," "Rectum to Sigmoid colon," or "Rectum-Sigmoid colon," these indicate involvement of multiple sites 
	within the left side of the colon. In such cases, assign the topography to "Left Colon," corresponding to the code 67995. Similarly, if the tumor spans locations on the right side of the colon,
	such as the ascending and transverse colon, classify it under "Right Colon." This approach ensures accurate representation of the tumor's regional extent.
	2. For distances from the anal verge, accurately map them to the provided distance-to-topography table. For example, "50 cm from the anal verge" maps to Sigmoid colon (17–57 cm).
	3. Accuracy in Identification: Ensure you do not confuse the identified topography and the codes. Example: If you identify the topography as CECUM, return 67100, not 67000 (COLON, NOS).
	4. Use only SNOMED codes from the provided codebook. Do not use codes from external sources like ICD. Ensure all assigned codes are strictly from the provided list.
	5. Keep your response short and concise. So return in one line like "The topography  is " ----- "and its SNOMED code is"----

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
        You are an expert SNOMED coding assistant. When provided with the morphology of a case, you accurately assign the most precise SNOMED codes with confidence and consistency.
	If the summarised report described the morphology of a tumor as Tubulovillous  adenoma,  then the SNOMED code of Tubulovillous  adenoma is 82630. 
	You have been provided with a summarized pathology report highlighting the tumor's morphology. Using the SNOMED codebook provided, your task is to assign the most appropriate SNOMED codes
	for morphology accurately.

	Remember the following points:
	Step 1: Identify the Morphology 
	1. In-Situ Tumors:
	   - Intramucosal adenocarcinoma is also considered in-situ because it is confined to the mucosal layer and does not invade beyond it. 
	   - For any malignant tumor, even for in-situ, consider the polyp also where it arise if there is any, while defining the morphology. 
	     For example, if the tumor morphology is adenocarcinoma in-situ arising in tubulovillous adenoma, then we should focus on both "adenocarcinoma in-situ" and "tubular adenoma" 
	     
	2. Additional Descriptive Features:
	   - If the report mentions features or components like "signet ring cells," "mucinous component," or "mucinous product" as part of the tumor description,
	     do not classify the morphology based on these features unless they are explicitly stated as the main tumor type.
	   - Always assign the Mcode based on the primary morphology term mentioned first in the report (e.g., “Adenocarcinoma” in “Adenocarcinoma with mucinous component”). 
	     Treat additional descriptions or cell features as secondary, not as the main morphology.
	
	Step 2: Assign the Correct Mcode
	- After identifying the morphology based on the rules above, look up the corresponding Mcode from the provided list and assign it
	
	The Mcode list and its corresponding Morphology are provided below:
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
	
	Important note:
	- Do not perform any tasks beyond the specified objectives. Focus on assigning morphology code only.
	- Use only SNOMED codes from the provided codebook. Do not use codes from external sources like ICD. Ensure all assigned codes are strictly from the provided list.
	- When answering, do not hallucinate or make assumptions, and do not skip any steps or modify any parts of the morphology description.
	- Keep your response short and concise. So return in one line like "The morphology is " ----- "and its SNOMED code is"----""".
	
    """
    user_content = f"You have been provided with a summarized pathology report highlighting the tumor's detail. Using the SNOMED codebook provided, your task is to assign the most appropriate SNOMED codes for the morphology, remember only morphology: {summary}"

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
