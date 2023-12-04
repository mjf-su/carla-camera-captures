import os
import numpy as np

# --- transform trimmed responses into a VQADataset compatible format. This file is only used to create the correct image paths relative to the data folder to be read in by colab ---

folder = os.path.dirname(__file__)
scenarios = ["anomaly_stop_sign"] # scenarios represented in the dataset

anom_response_file = "trim_anom_vlm_responses.npz"
anom_dataset_file = "trim_anom_VQADataset.npz"
nom_response_file = "trim_nom_vlm_responses.npz"
nom_dataset_file = "trim_nom_VQADataset.npz"

for scenario in scenarios:
    scenario_folder = os.path.join(folder, "data", scenario) # scenario type

    for exp in os.listdir(scenario_folder):
        if "exp" not in exp:
            continue

        anom_questions = []
        anom_answers = []
        anom_filenames = []

        nom_questions = []
        nom_answers = []
        nom_filenames = []

        exp_folder = os.path.join(scenario_folder, exp) # specific experiment
        anom_responses = np.load(os.path.join(exp_folder, anom_response_file))
        nom_responses = np.load(os.path.join(exp_folder, nom_response_file))

        for anom_response, anom_image in zip(anom_responses["vlm_responses"], anom_responses["image_names"]):
            anom_questions.append(anom_responses["vlm_prompt"])
            anom_answers.append(anom_response)
            anom_filenames.append(scenario + "/" + exp + "/image_data/" + anom_image) # prepare image paths for colab
        
        for nom_response, nom_image in zip(nom_responses["vlm_responses"], nom_responses["image_names"]):
            nom_questions.append(nom_responses["vlm_prompt"])
            nom_answers.append(nom_response)
            nom_filenames.append(scenario + "/" + exp + "/image_data/" + nom_image) # prepare image paths for colab
        
        np.savez(os.path.join(exp_folder, anom_dataset_file), questions = anom_questions, answers = anom_answers, filenames = anom_filenames)
        np.savez(os.path.join(exp_folder, nom_dataset_file), questions = nom_questions, answers = nom_answers, filenames = nom_filenames)
