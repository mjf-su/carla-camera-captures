import os
import numpy as np

current_response_file = "gpt4v_responses_newFormat.npz" # in image data folder of exp folder
new_response_file = "vlm_responses.npz" # in exp folder
frame_number_order = 3

scenario_folder = os.path.dirname(__file__)
for exp in os.listdir(scenario_folder):
    if "exp" not in exp:
        continue

    exp_folder = os.path.join(scenario_folder, exp)
    image_folder = os.path.join(exp_folder, "image_data")
    current_responses = np.load(os.path.join(image_folder, current_response_file))

    new_image_names = []
    for image in current_responses["image_names"]:
        frame_number = image.split("frame")[1].split(".jpg")[0]

        for _ in np.arange(frame_number_order - len(frame_number)):
                frame_number = "0" + frame_number
        filename = "frame" + frame_number + ".jpg"
        new_image_names.append(filename)

    assert len(new_image_names) == current_responses["gpt4v_responses"].shape[0]
    np.savez(os.path.join(exp_folder, new_response_file), vlm_responses = current_responses["gpt4v_responses"], image_names = new_image_names, vlm_prompt = current_responses["vlm_prompt"])