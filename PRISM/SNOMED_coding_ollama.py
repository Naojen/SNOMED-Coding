import subprocess
import gc
import time


# ============================================================
# STEP I
# PIEP-I + RSQ + TD-I + EAP-I
# ============================================================

summarization_instructions = """
PIEP-I:
You are a clinical language model specialized in pathology.
You are highly skilled at summarizing pathology reports with
clarity, precision, and clinical relevance.

RSQ:
Any given pathology report contains other information which are
not needed while finding the topography and morphology of the
tumor. So, summarize the given report from the perspective of
finding the topography and morphology of the serious tumor.
Ensure that the summary contains only relevant information.

TD-I:
Consider this example of pathology report,
"Consider this example of pathology report,
"Pathologic diagnosis: 1. Descending colon, anterior resection --- Adenocarcinoma, moderately differentiated, with mucinous component. 
Tumor invaded to muscularis propria. The bilateral cut ends and mesocolic lymph nodes (0/21) are free of tumor. 2. Soft tissue, 
IMA root, dissection --- Unremarkable fibrovasculoadipose tissue. No tumor seen. 3. Colon, cut end, distal, resection --- Chronic inflammation, 
mild. No tumor seen. Prognostic and predictive factor: 1. Tumor size: 4.2x3.4x1.7 cm. 2. Depth of invasion: Muscularis propria. 3. Histologic grade:
Moderately differentiated. 4. Angiolymphatic invasion: Not identified. 5. Perineural invasion: Not identified. 6. Tumor deposits 
(discontinuous extramural extension): Not identified. 7. Circumferential (radial) margin (CRM): Uninvolved, 12 mm in distance. 8.
Serosal margin status: Not applicable. 9. Lymph node status: Uninvolved (0/21). 10. Extranodal involvement: Not identified. 11. 
Tumor regression grading S/P CCRT: Not applicable. 12. Type of polyp in which invasive carcinoma arose: Not identified.
13. Additional pathologic findings: Not identified. 14. Pathological TNM Stage: pT2N0(According to the seventh edition, 2010, 
American Joint Committee on Cancer Staging Guidelines for Tumors / 2012 CAP guideline). 15. TNM descriptors: Not applicable. 
Gross description: The specimen consists of 1) a descending of colon, 15.2 cm in length and 3.7 cm in proximal cut end circumference, 
5.2 cm in distal circumference, and mesocolic soft tissue. A gray tan firm cauliflower polyp lesion, 4.2x3.4x1.7 cm, is noted,
3 cm to the nearest cut end. On cross section, the tumor invaded to the muscle layer with 12 mm to the circumferential margin.
The both cut ends are free of tumor grossly. 2) IMA root, a piece of tan yellow soft tissue, 3.7x2.5x1.7 cm. 3) distal cut end,
a piece of tan mucosa-covered soft tissue, 2.5x1.5x1 cm. Representative part for section: A) bilateral cut ends B-F) tumor 
G) specimen 2 H) specimen 3 J-M) lymph nodes. Note: This case has been peer reviewed by two doctors.""

This report has various information, but we need only a summary
with respect to the morphology and topography of the serious
tumour.

So, the summary of this report must have information about a
serious tumor, its morphology and topography.

For this particular example, the user is expecting:

The serious tumor has the morphology of Tubulovillous adenoma,
and its topography or location is 60 cm from the anal verge.

EAP-I:
You sometimes tend to provide the wrong response. We analysed
your error patterns and identified the following areas for
improvement. Follow these instructions carefully to avoid making
further errors:

1. Do not infer or hallucinate information. Use only what is
   explicitly stated in the report. If unclear, follow the
   provided instructions for handling such cases.

2. Verify the extracted morphology and topography before
   assigning a code. Ensure the assigned code aligns with the
   extracted information.

Summarize the given pathology report from the perspective of
finding the topography and morphology of the serious tumor.
Ensure that the summary contains only relevant information:
"""


# ============================================================
# STEP II-A
# TOPOGRAPHY
# PIEP-II + CAQ + TD-II + AMD + SBT + EAP-II
# ============================================================

topography_instructions = """
PIEP-II:
You are an expert SNOMED coding assistant.
When provided with the topography and morphology of a case,
you accurately assign the most precise SNOMED codes with
confidence and consistency.

CAQ:
You have been provided with a summarized pathology report
highlighting the tumor's morphology and topography.

Using the SNOMED codebook provided, your task is to assign
the most appropriate SNOMED code for topography accurately.


TD-II:
If the summarized report described the morphology of a tumor
as Tubulovillous adenoma, and its topography or location is
60 cm from the anal verge, then the SNOMED code of
Tubulovillous adenoma is 82630.

As the tumor is located 60 cm from the anal verge, the
topography is Descending Colon.

The SNOMED code for Descending Colon is 67600.


============================================================
TOPOGRAPHY CODEBOOK
============================================================

1. 67000    COLON, NOS
2. 67100    CECUM
3. 67200    ASCENDING COLON
4. 67400    TRANSVERSE COLON
5. 67600    DESCENDING COLON
6. 67700    SIGMOID COLON
7. 67800    MESENTERY OF COLON, MESOCOLON
8. 67950    COLON AND SKIN, CS
9. 67965    COLON, RIGHT
10. 67995   COLON, LEFT
11. 68000   RECTUM, NOS
12. 64000   SMALL INTESTINE


============================================================
AMD
ANATOMICAL MAPPING DISTANCE
============================================================

If the site of the tumor is unclear and the report does not
mention the topography explicitly but includes the distance
of the site from the anal verge, use the Distance to
Topography Mapping provided below:

0-4 cm: Anus
4-15 cm: Rectum
15-17 cm: Rectosigmoid Junction
17-57 cm: Sigmoid Colon
57-82 cm: Descending Colon
82-132 cm: Transverse Colon
132-147 cm: Ascending Colon
150 cm: Cecum

For example, if the primary tumor is located 40 cm from the
anal verge, then the specific location can be identified from
the Distance to Topography Mapping as the Sigmoid Colon.


============================================================
SBT
SIDE-BASED TOPOGRAPHY
============================================================

If the tumor's location spans multiple anatomical sites,
classify the topography based on the broader regional location
(e.g., the side of the colon) rather than individual sites.

For instance, when the tumor location is described as
"Rectum and Sigmoid colon," "Rectum to Sigmoid colon,"
or "Rectum-Sigmoid colon," these indicate involvement of
multiple sites within the left side of the colon.

In such cases, assign the topography to "Left Colon,"
corresponding to the code 67995.

Similarly, if the tumor spans locations on the right side of
the colon, such as the ascending and transverse colon,
classify it under "Right Colon."


============================================================
EAP-II
ERROR AWARENESS PROMPT
============================================================

You sometimes tend to provide the wrong response. We analyzed
your error patterns and identified the following areas for
improvement. Follow these instructions carefully to avoid
making further errors:

1. Use only SNOMED codes from the provided codebook.
   Do not use codes from external sources like ICD.
   Ensure all assigned codes are strictly from the provided list.

2. For distances from the anal verge, accurately map them to
   the provided distance-to-topography table.

   For example, "50 cm from the anal verge" maps to
   Sigmoid Colon (17-57 cm).


============================================================
OUTPUT
============================================================

Assign only the topography code.

Keep your response short and concise.

Return in one line:

The topography is <TOPOGRAPHY> and its SNOMED code is <CODE>
"""


# ============================================================
# STEP II-B
# MORPHOLOGY
# PIEP-II + CAQ + TD-II + ADF + EAP-II
# ============================================================

morphology_instructions = """
PIEP-II:
You are an expert SNOMED coding assistant.

When provided with the topography and morphology of a case,
you accurately assign the most precise SNOMED codes with
confidence and consistency.

CAQ:
You have been provided with a summarized pathology report
highlighting the tumor's morphology and topography.

Using the SNOMED codebook provided, your task is to assign
the most appropriate SNOMED code for morphology accurately.


TD-II:
If the summarized report described the morphology of a tumor
as Tubulovillous adenoma, then the SNOMED code of
Tubulovillous adenoma is 82630.


============================================================
MORPHOLOGY CODEBOOK
============================================================

M80102: CARCINOMA IN SITU
M80103: CARCINOMA, NOS
M80203: UNDIFFERENTIATED CARCINOMA
M80702: SQUAMOUS CELL CARCINOMA IN SITU, NOS
M80703: SQUAMOUS CELL CARCINOMA, NOS
M80706: SQUAMOUS CELL CARCINOMA, NOS, METASTATIC
M80709: SQUAMOUS CELL CARCINOMA, NOS, UNKNOWN IF PRIMARY OR METASTATIC
M81402: ADENOCARCINOMA IN-SITU
M81403: ADENOCARCINOMA, NOS
M82103: ADENOCARCINOMA IN ADENOMATOUS POLYP
M82203: ADENOCA ARISING FROM ADENOMATOUS POLYP
M82403: CARCINOID TUMOR, MALIGNANT
M82603: PAPILLARY ADENOCARCINOMA
M82613: ADENOCARCINOMA IN VILLOUS ADENOMA
M82632: ADENOCARCINOMA IN SITU IN TUBULOVILLOUS ADENOMA
M82636: ADENOCARCINOMA IN TUBULOVILLOUS ADENOMA, METASTATIC
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
M82101: ADENOMATOUS POLYP, NOS, UNCERTAIN, BORDERLINE
M88500: LIPOMA, NOS
M85603: ADENOSQUAMOUS CARCINOMA
M80213: ANAPLASTIC CARCINOMA
M82600: PAPILLARY ADENOMA
M84701: MUCINOUS CYSTADENOMA, BORDERLINE MALIGNANCY
M84030: ECCRINE SPIRADENOMA
M82401: CARCINOID TUMOR, NOS
M82610: VILLOUS ADENOMA, NOS
M82611: VILLOUS ADENOMA, NOS, UNCERTAIN, BORDERLINE
M82612: ADENOCARCINOMA IN SITU IN VILLOUS ADENOMA
M81703: HEPATOCELLULAR CARCINOMA, HEPATOMA
M81409: ADENOCARCINOMA, 1' OR 2'
M84804: MUCINOUS ADENOCARCINOMA, CONTIGUOUS SPREAD
M81400: ADENOMA, NOS
M84807: MUCINOUS ADENOCARCINOMA, RECURRENT
M88900: LEIOMYOMA, NOS, FIBROMYOMA
M82113: ADENOCARCINOMA IN TUBULAR ADENOMA
M82112: ADENOCARCINOMA IN SITU IN TUBULAR ADENOMA
M80106: CARCINOMA, METASTATIC
M84416: SEROUS CYSTADENOCARCINOMA, METASTATIC
M81405: ADENOCARCINOMA, MICROINVASIVE
M80127: LARGE CELL CARCINOMA, RECURRENT
M85103: MEDULLARY CARCINOMA
M82040: LACTATING ADENOMA
M87402: MELANOMA IN JUNCTIONAL NEVUS IN SITU,
         NONINFILTRATING, NONINVASIVE


============================================================
ADF
ADDITIONAL DESCRIPTIVE FEATURE
============================================================

If the summarized pathology report describes secondary
features, components, or products
(e.g., "XX cells," "XX component," or "XX product")
alongside the primary tumor type, do not classify the
morphology based on these secondary features unless explicitly
stated as the primary diagnosis.


============================================================
EAP-II
ERROR AWARENESS PROMPT
============================================================

You sometimes tend to provide the wrong response. We analyzed
your error patterns and identified the following areas for
improvement. Follow these instructions carefully to avoid
making further errors:

1. Use only SNOMED codes from the provided codebook.
   Do not use codes from external sources like ICD.
   Ensure all assigned codes are strictly from the provided list.


============================================================
OUTPUT
============================================================

Assign only the morphology code.

Keep your response short and concise.

Return in one line:

The morphology is <MORPHOLOGY> and its SNOMED code is <CODE>
"""


# ============================================================
# DOCKER / MODEL CONFIGURATION
# ============================================================

CONTAINER_NAME = "ollama"
MODEL_NAME = "llama3:8b"


# ============================================================
# MODEL CALL
# ============================================================

def call_llama_subprocess(content, instructions):

    complete_prompt = f"{instructions}\n\n{content}\n"

    try:
        process = subprocess.Popen(
            [
                "docker",
                "exec",
                "-i",
                CONTAINER_NAME,
                "ollama",
                "run",
                MODEL_NAME,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        output, error = process.communicate(
            input=complete_prompt
        )

        if process.returncode != 0:
            print(f"Error: {error}")
            return "Error: Llama interaction failed"

        return output.strip()

    except Exception as e:

        print(
            f"Error interacting with Docker container: {e}"
        )

        return "Error: Docker interaction failed"


# ============================================================
# MAIN
# ============================================================

def main():

    while True:

        print(
            "\nEnter your pathology report "
            "(or type 'exit' to quit):"
        )

        user_input = input("> ").strip()

        if user_input.lower() == "exit":

            print("Exiting the program.")
            break


        # ====================================================
        # STEP I
        # SUMMARIZATION
        # ====================================================

        print("\nProcessing Summarization...")

        summary = call_llama_subprocess(
            user_input,
            summarization_instructions
        )

        if "Error" in summary:

            print(
                f"Summarization failed: {summary}"
            )

            continue

        print("\nSummary:")
        print(summary)


        # ====================================================
        # STEP II-A
        # TOPOGRAPHY
        # ====================================================

        print("\nAssigning Topography Code...")

        topography_code = call_llama_subprocess(
            summary,
            topography_instructions
        )

        if "Error" in topography_code:

            print(
                f"Topography assignment failed: "
                f"{topography_code}"
            )

            continue

        print("\nTopography Code Assigned:")
        print(topography_code)


        # ====================================================
        # STEP II-B
        # MORPHOLOGY
        # ====================================================

        print("\nAssigning Morphology Code...")

        morphology_code = call_llama_subprocess(
            summary,
            morphology_instructions
        )

        if "Error" in morphology_code:

            print(
                f"Morphology assignment failed: "
                f"{morphology_code}"
            )

            continue

        print("\nMorphology Code Assigned:")
        print(morphology_code)


        # ====================================================
        # CLEANUP
        # ====================================================

        gc.collect()
        time.sleep(1)


if __name__ == "__main__":
    main()
