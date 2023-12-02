import os
import json
import numpy as np

folder = os.path.dirname(__file__)
scenarios = ["nominal_traffic_light", "nominal_stop_sign"] # scenarios represented in the dataset
response_file = "trim_vlm_responses.npz"
dataset_file = "trim_VQADataset.npz"

for scenario in scenarios:
    scenario_folder = os.path.join(folder, "data", scenario) # scenario type

    with open(os.path.join(scenario_folder, "incorrect_labels.json"), 'r') as js:
        incorrect_labels = json.load(js)
    for exp in os.listdir(scenario_folder):
        if "exp" not in exp:
            continue

        questions = []
        answers = []
        filenames = []

        exp_folder = os.path.join(scenario_folder, exp) # specific experiment
        exp_responses = np.load(os.path.join(exp_folder, response_file))

        for i, image in enumerate(exp_responses["image_names"]):
            if image in incorrect_labels[exp]:
                continue # skip errenous response
            else:
                questions.append(exp_responses["vlm_prompt"])
                answers.append(exp_responses["vlm_responses"][i])
                filenames.append(scenario + "/" + exp + "/image_data/" + exp_responses["image_names"][i])
        np.savez(os.path.join(exp_folder, dataset_file), questions = questions, answers = answers, filenames = filenames)




