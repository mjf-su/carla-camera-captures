import numpy as np
from PIL import Image
import os

project_folder = os.path.dirname(__file__)
scenarios = ["nominal_stop_sign", "nominal_traffic_light", "anomaly_traffic_light", "anomaly_stop_sign", "anomaly_ood"]
frame_number_order = 3

for scenario in scenarios:
    print("--- " + scenario + " ---")
    scenario_folder = os.path.join(project_folder, "data", scenario)

    for exp in os.listdir(scenario_folder):
        if "exp" not in exp:
            continue

        exp_folder = os.path.join(scenario_folder, exp)
        image_folder = os.path.join(exp_folder, "image_data")
        vehicle_data = np.load(os.path.join(exp_folder, "vehicle_data.npz"))

        for image in os.listdir(image_folder):
            if ".jpg" not in image:
                continue
            else:
                os.remove(os.path.join(image_folder, image))


        obs = vehicle_data["observations"]
        filenames = []
        for frame in np.arange(obs.shape[0]):
            image = Image.fromarray(obs[frame])
            filename = str(frame)
            for _ in np.arange(frame_number_order - len(filename)):
                filename = "0" + filename
            filename = "frame" + filename + ".jpg"

            image.save(os.path.join(image_folder, filename))
            filenames.append(filename)
        
        np.savez(os.path.join(exp_folder, "vehicle_data.npz"), 
                fsm_states = vehicle_data["fsm_states"], 
                av_states = vehicle_data["av_states"],
                observations = vehicle_data["observations"],
                anomaly_detected = vehicle_data["anomaly_detected"],
                filenames = filenames)