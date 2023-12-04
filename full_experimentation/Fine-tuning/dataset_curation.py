import os
import numpy as np

# --- transform trimmed responses into a VQADataset compatible format. This file is only used to create the correct image paths relative to the data folder to be read in by colab ---

folder = os.path.dirname(__file__)
scenarios = ["nominal_traffic_light", "nominal_stop_sign"] # scenarios represented in the dataset
response_file = "trim_vlm_responses.npz" # trimmed responses already exclude incorrect classifications and incorrect formats
dataset_file = "trim_VQADataset.npz"

for scenario in scenarios:
    scenario_folder = os.path.join(folder, "data", scenario) # scenario type

    for exp in os.listdir(scenario_folder):
        if "exp" not in exp:
            continue

        questions = []
        answers = []
        filenames = []

        exp_folder = os.path.join(scenario_folder, exp) # specific experiment
        exp_responses = np.load(os.path.join(exp_folder, response_file))

        for i, image in enumerate(exp_responses["image_names"]):
            questions.append(exp_responses["vlm_prompt"])
            answers.append(exp_responses["vlm_responses"][i])
            filenames.append(scenario + "/" + exp + "/image_data/" + exp_responses["image_names"][i]) # prepare image paths for colab
        np.savez(os.path.join(exp_folder, dataset_file), questions = questions, answers = answers, filenames = filenames)




