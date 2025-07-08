# SAFER
Code repository for paper ['SAFER: A Calibrated Risk-Aware Multimodal Recommendation Model for Dynamic Treatment Regimes'](https://arxiv.org/pdf/2506.06649)

# Dataset
You need to get acess MIMIC III dataset (in our reimplemtation, ver 1.4 was used but the original nature medicine paper used ver 1.3) from (https://mimic.physionet.org/). To get the dataset, you need to satisfy requirements from the webiste (take an online course and get approval from the manager). The MIMIC dataset is about 6G (compressed).

MIMIC Access. Create a Google Cloud account that you will use to access the MIMIC-IV data through BigQuery. Get access to MIMIC-IV by going to [PhysioNet](https://mimic.physionet.org/). Then select "Request access using Google BigQuery".

# Preprocess
Please follow [AI clinician](https://github.com/uribyul/py_ai_clinician) first, and then use the provided ipynb file to do the further processing.
