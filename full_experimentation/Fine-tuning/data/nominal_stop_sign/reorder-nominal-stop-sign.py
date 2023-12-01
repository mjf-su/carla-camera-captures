import os
import numpy as np

current_response_file = "vlm_responses.npz" # in exp folder
new_response_file = "vlm_responses.npz" # in exp folder

scenario_folder = os.path.dirname(__file__)
for exp in os.listdir(scenario_folder):
    if "exp" not in exp:
        continue

    exp_folder = os.path.join(scenario_folder, exp)
    current_responses = np.load(os.path.join(exp_folder, current_response_file))

    sort = np.argsort(current_responses["image_names"])
    np.savez(os.path.join(exp_folder, new_response_file), vlm_responses = current_responses["vlm_responses"][sort], image_names = current_responses["image_names"][sort], vlm_prompt = current_responses["vlm_prompt"])