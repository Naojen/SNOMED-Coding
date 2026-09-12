import subprocess
import gc
import time


# ============================================================
# STEP I: INFORMATION EXTRACTION / SUMMARIZATION
#
# PIEP-I : Reused
# RSQ    : Modified for prostate
# TD-I   : Modified using two prostate demonstrations
# EAP-I  : Reused / adapted
# ============================================================

summarization_instructions = """
PIEP-I:

You are a clinical language model specialized in pathology. You are highly skilled at summarizing pathology reports with clarity, precision, and clinical relevance.

RSQ:

A prostate pathology report may contain clinical history, multiple biopsy specimens, grading information, ancillary studies, gross descriptions, microscopic findings, benign abnormalities, inflammatory findings, and other information that is not directly required for morphology and topography coding. Summarize the given prostate pathology report from the perspective of identifying the morphology-defining information of the clinically relevant primary diagnosis while preserving its primary anatomical site. The summary should contain only the information relevant to determining tumor morphology and topography.

TD-I:

Consider the following prostate pathology report. Pathologic diagnosis: Prostate, right peripheral, TRUS biopsy --- Adenocarcinoma, Gleason's score 3+3=6 (1/4, 1%). Prostate, right parasagittal, TRUS biopsy --- Adenocarcinoma, Gleason's score 3+3=6 (1/5, 3%). Prostate, left peripheral, TRUS biopsy --- Adenocarcinoma, Gleason's score 3+3=6 (1/4, 3%). Prostate, left parasagittal, TRUS biopsy --- Adenocarcinoma, Gleason's score 3+4=7 (3/4, 30%). Perineural invasion is absent. The gross description contains multiple strips of prostate tissue from the right peripheral, right parasagittal, left peripheral, and left parasagittal regions. Microscopic examination shows prostate tissue with small atypical distorted glands showing variation in size and shape, together with some ill-defined glands with poorly formed glandular lumina.

Although this report contains multiple specimens, Gleason scores, percentages of involvement, gross descriptions, and microscopic findings, the clinically relevant information for morphology and topography coding is that adenocarcinoma is present in multiple prostate biopsy specimens. Therefore, the expected summary is: The primary diagnosis is adenocarcinoma involving the prostate.

Consider another prostate pathology report. Pathologic diagnosis: Prostate, right peripheral, TRUS biopsy --- Adenocarcinoma, Gleason's score 3+3=6 (1/3, 1%). Prostate, right parasagittal, transrectal needle biopsy --- nodular hyperplasia and chronic prostatitis. Prostate, left peripheral, TRUS biopsy --- nodular hyperplasia. Prostate, left parasagittal, TRUS biopsy --- nodular hyperplasia. Immunohistochemistry stains for high molecular weight cytokeratin are negative and alpha-methylacyl-CoA racemase is positive for section A. Microscopic examination of section A shows prostate tissue with small atypical distorted glands with variation in size and shape, whereas the other sections show glandular and stromal hyperplasia.

This report contains both malignant and benign findings from different prostate specimens. The confirmed adenocarcinoma in the right peripheral prostate represents the clinically relevant malignant diagnosis, whereas nodular hyperplasia and chronic prostatitis are additional findings in other specimens. Therefore, the expected summary is: The primary diagnosis is adenocarcinoma of the prostate; nodular hyperplasia and chronic prostatitis are additional benign findings.

EAP-I:

You sometimes tend to provide the wrong response. We analysed your error patterns and identified areas for improvement. Do not infer or hallucinate information and use only findings explicitly stated in the pathology report. Identify the clinically relevant primary diagnosis and distinguish it from secondary, incidental, benign, historical, or unrelated findings. Verify that the extracted morphology corresponds to the diagnosis being coded and preserve the primary prostate site when it is stated. Do not assign SNOMED codes during this summarization step.

Summarize the following prostate pathology report based only on the information relevant to identifying the primary morphology and topography:
"""


# ============================================================
# STEP II-A: TOPOGRAPHY CODE ASSIGNMENT
#
# PIEP-II : Reused
# CAQ     : Modified for prostate
# TD-II   : Modified using prostate examples
# AMD     : Omitted
# SBT     : Omitted
# EAP-II  : Modified
# ============================================================

topography_instructions = """
PIEP-II:

You are an expert SNOMED coding assistant. When provided with the topography and morphology of a case, you accurately assign the most precise SNOMED code with confidence and consistency.

CAQ:

You have been provided with a summarized prostate pathology report highlighting the morphology and topography of the primary diagnosis. Using only the prostate topography candidate codebook provided below, assign the single most appropriate SNOMED topography code. Select the anatomical site corresponding to the primary pathological diagnosis and do not replace the primary site with an adjacent structure, metastatic site, lymph node, margin, or other secondary anatomical finding.

TD-II:

In the first demonstration, the summarized report states that the primary diagnosis is adenocarcinoma involving the prostate. The anatomical topography is therefore Prostate, NOS, which corresponds to T-77100.

In the second demonstration, the summarized report states that the primary diagnosis is adenocarcinoma of the prostate and that nodular hyperplasia and chronic prostatitis are additional benign findings. All of these findings arise from prostate biopsy specimens, and the primary coded site remains the prostate. Therefore, the appropriate topography is Prostate, NOS, corresponding to T-77100.


============================================================
PROSTATE TOPOGRAPHY CANDIDATE CODEBOOK
============================================================

T-77100: Prostate, NOS
T-77500: Seminal vesicle
T-74000: Urinary bladder
T-74400: Bladder neck
T-74320: Ureteral orifice
T-75000: Urethra
T-68000: Rectum
T-73000: Ureter, NOS
T-73010: Right ureter
T-73020: Left ureter
T-79800: Vas deferens, bilateral
T-79220: Left vas deferens
T-Y6000: Pelvis, NOS
T-08600: Pelvic lymph node
T-10500: Spine
T-10800: Sacrum
T-11710: Femur
T-12103: Intervertebral disc
T-56000: Liver
T-28200: Right upper lobe of lung
T-28300: Right middle lobe of lung
T-28400: Right lower lobe of lung
T-28600: Left upper lobe of lung
T-28700: Left lower lobe of lung
T-93000: Adrenal gland
T-X2000: Brain
T-X1120: Dura mater
T-Y4400: Peritoneum
T-00100: Surgical margin
T-01000: Skin
T-1X000: Soft tissue


EAP-II:

Use only SNOMED codes contained in the provided candidate codebook and do not generate or infer codes from external coding systems. Select exactly one topography code. Verify that the selected topography corresponds to the primary diagnosis extracted from the pathology report. When the primary diagnosis originates in the prostate, do not assign the code of an adjacent organ, metastatic site, lymph node, surgical margin, or secondary involved structure merely because it is also mentioned in the report.

Return only one concise line in the following format:

The topography is <TOPOGRAPHY> and its SNOMED code is <CODE>
"""


# ============================================================
# STEP II-B: MORPHOLOGY CODE ASSIGNMENT
#
# PIEP-II : Reused
# CAQ     : Modified for prostate
# TD-II   : Modified using prostate examples
# ADF     : Reused / adapted
# EAP-II  : Modified
# ============================================================

morphology_instructions = """
PIEP-II:

You are an expert SNOMED coding assistant. When provided with the topography and morphology of a case, you accurately assign the most precise SNOMED code with confidence and consistency.

CAQ:

You have been provided with a summarized prostate pathology report highlighting the morphology and topography of the primary diagnosis. Using only the prostate morphology candidate codebook provided below, assign the single most appropriate SNOMED morphology code. Give particular attention to distinguishing the primary morphology from secondary histological features, benign findings, inflammatory changes, hyperplasia, margins, metastatic findings, and other incidental pathological information.

TD-II:

In the first demonstration, the summarized report states that the primary diagnosis is adenocarcinoma involving the prostate. Because the report identifies adenocarcinoma as the primary morphology without specifying another morphology represented in the candidate codebook, the appropriate morphology is Adenocarcinoma, NOS, corresponding to M-81403.

In the second demonstration, the summarized report states that the primary diagnosis is adenocarcinoma of the prostate, while nodular hyperplasia and chronic prostatitis are additional benign findings in other biopsy specimens. The presence of multiple benign findings does not replace the confirmed adenocarcinoma as the primary malignant morphology. Therefore, the appropriate morphology is Adenocarcinoma, NOS, corresponding to M-81403.


============================================================
PROSTATE MORPHOLOGY CANDIDATE CODEBOOK
============================================================

M-00100: Unremarkable
M-09010: Tissue unsatisfactory
M-09350: Morphologic description only
M-09400: Surgical margins free of tumor
M-09450: No evidence of malignancy
M-30010: Lithiasis
M-36370: Exudate, NOS or fibrinous exudate
M-37000: Hemorrhage, NOS
M-41000: Acute inflammation
M-41700: Acute necrotizing inflammation
M-41740: Abscess, NOS
M-42100: Acute and chronic inflammation
M-43000: Chronic inflammation
M-44000: Histiocytic granuloma
M-44400: Suppurative granulomatous inflammation
M-44700: Caseating granuloma
M-45020: Granulation tissue, NOS
M-49000: Fibrosis, NOS
M-54000: Necrosis, NOS
M-55100: Amyloid deposition
M-55430: Dystrophic calcification
M-58000: Atrophy, NOS
M-67034: Dysplasia, endocervical glandular
M-69700: Cellular atypia, NOS
M-69760: Atypia suspicious for malignancy
M-72005: Atypical hyperplasia
M-72030: Nodular hyperplasia
M-72175: Atypical intraductal hyperplasia
M-72425: Atypical glandular or adenomatous hyperplasia
M-74000: Dysplasia, NOS
M-74007: Moderate dysplasia, CIN II
M-74008: Severe dysplasia
M-74200: Adenosis, NOS
M-76005: Atypical vascular lesion
M-80003: Malignant neoplasm, NOS
M-80009: Malignant neoplasm, uncertain whether primary or metastatic
M-80059: Vascular tumor thrombus
M-80069: Lymphatic tumor thrombus
M-80103: Carcinoma, NOS
M-80106: Carcinoma, metastatic
M-80413: Small cell carcinoma
M-81202: Transitional cell carcinoma in situ
M-81203: Transitional cell carcinoma
M-81204: Transitional cell carcinoma, contiguous spread
M-81206: Transitional cell carcinoma, metastatic
M-81303: Papillary transitional cell carcinoma
M-81304: Papillary transitional cell carcinoma, contiguous spread
M-81306: Papillary transitional cell carcinoma, metastatic
M-81403: Adenocarcinoma, NOS
M-81404: Adenocarcinoma, contiguous spread
M-81406: Adenocarcinoma, metastatic
M-81482: Glandular dysplasia or high-grade intraepithelial neoplasia
M-82401: Carcinoid tumor, NOS
M-88023: Pleomorphic undifferentiated sarcoma, except bone


ADF:

If the summarized prostate pathology report describes secondary features, components, products, grading information, percentages of tumor involvement, or other histological descriptors together with the primary morphology, do not classify the morphology based on those secondary features unless they are explicitly stated as the primary diagnosis. Distinguish the principal morphology from accompanying benign, inflammatory, hyperplastic, incidental, or secondary pathological findings.

EAP-II:

Use only SNOMED codes contained in the provided morphology candidate codebook and do not use codes from external coding systems. Select exactly one morphology code. Verify that the selected morphology corresponds to the primary diagnosis extracted in Step I. Do not replace a confirmed primary morphology with a secondary, incidental, benign, inflammatory, historical, or accompanying pathological finding merely because that finding also appears in the report.

Return only one concise line in the following format:

The morphology is <MORPHOLOGY> and its SNOMED code is <CODE>
"""


# ============================================================
# DOCKER / MODEL CONFIGURATION
# ============================================================

CONTAINER_NAME = "ollama"

# Change this for each prostate transferability experiment:
#
# LLaMA-3-70B       -> llama3:70b
# LLaMA-3.1-70B     -> llama3.1:70b
# LLaMA-3.1-405B    -> llama3.1:405b
#
MODEL_NAME = "llama3.1:405b"


# ============================================================
# MODEL CALL
# ============================================================

def call_llama_subprocess(content, instructions):

    complete_prompt = (
        f"{instructions.strip()}\n\n"
        f"PATHOLOGY INFORMATION:\n"
        f"{content.strip()}\n"
    )

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

            print(
                f"Error from Ollama:\n{error}"
            )

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
            "\nEnter your prostate pathology report "
            "(or type 'exit' to quit):"
        )

        user_input = input("> ").strip()

        if user_input.lower() == "exit":

            print("Exiting the program.")
            break


        # ----------------------------------------------------
        # STEP I
        # INFORMATION EXTRACTION / SUMMARIZATION
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STEP I: PROSTATE PATHOLOGY SUMMARIZATION"
        )

        print(
            "=" * 70
        )

        summary = call_llama_subprocess(
            user_input,
            summarization_instructions,
        )

        if "Error" in summary:

            print(
                f"Summarization failed: {summary}"
            )

            continue

        print("\nGenerated Summary:\n")
        print(summary)


        # ----------------------------------------------------
        # STEP II-A
        # TOPOGRAPHY CODE ASSIGNMENT
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STEP II-A: TOPOGRAPHY CODE ASSIGNMENT"
        )

        print(
            "=" * 70
        )

        topography_code = call_llama_subprocess(
            summary,
            topography_instructions,
        )

        if "Error" in topography_code:

            print(
                f"Topography assignment failed: "
                f"{topography_code}"
            )

            continue

        print(
            "\nAssigned Topography:\n"
        )

        print(
            topography_code
        )


        # ----------------------------------------------------
        # STEP II-B
        # MORPHOLOGY CODE ASSIGNMENT
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STEP II-B: MORPHOLOGY CODE ASSIGNMENT"
        )

        print(
            "=" * 70
        )

        morphology_code = call_llama_subprocess(
            summary,
            morphology_instructions,
        )

        if "Error" in morphology_code:

            print(
                f"Morphology assignment failed: "
                f"{morphology_code}"
            )

            continue

        print(
            "\nAssigned Morphology:\n"
        )

        print(
            morphology_code
        )


        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        gc.collect()

        time.sleep(1)

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PROCESSING COMPLETED"
        )

        print(
            "=" * 70
        )


if __name__ == "__main__":
    main()
