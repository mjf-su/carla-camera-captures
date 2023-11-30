import os
import json
import numpy as np

folder = os.path.dirname(__file__)
scenarios = ["nominal_stop_sign"] # scenarios represented in the dataset
response_file = "gpt4v_newFormat_tokenLimit.npz"
dataset_file = "VQADataset.npz"

questions = []
answers = []
filenames = []

vlm_prompt = '''I am the fault monitor for a vision-based autonomous vehicle. My job is to analyze the vehicle's observations and identify anything that could cause the vehicle to take actions that are unsafe, unpredictable or violate traffic rules. For each object that the vehicle observes, I will reason about whether the object constitutes a normal observation or an anomaly. Normal observations do not detrimentally affect the vehicle's performance, whereas anomalies might. Finally, I will classify whether the overall scenario is normal or abnormal. That is, I will output a response which follows the structure below: 

"The vehicle is driving on the road and observes:
-a cyclist on the sidewalk
-a car on the road
-a pedestrian carrying a bright green balloon

Cyclist on the sidewalk:
1. Is this common to see while driving? Yes, cyclists can often be seen riding on the road or occasionally on sidewalks.
2. Can this influence the vehicle's behavior? No, they are on the sidewalk and not on the road.
3. Can the vehicle drive safely in its presence? Yes, cyclists are commonly seen on the road and the vehicle should be able to drive safely in their presence.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? No, they are on the sidewalk and not on the road.
Classification: Normal.

Car on the road:
1. Is this common to see while driving? Yes, cars are common to see while driving.
2. Can this influence the vehicle's behavior? Yes, the autonomous vehicle must respect other vehicles on the road, avoid collisions and obey the rules of the road.
3. Can the vehicle drive safely in its presence? Yes, cars are commonly seen on the road and the autonomous vehicle should be able to drive safely in their presence.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? No, autonomous vehicles are programmed to appropriately drive and interact with other cars on the road.
Classification: Normal.

Pedestrian carrying a bright green balloon:
1. Is this common to see while driving? Pedestrians are commonly seen on the sidewalk or crossing at intersections. They may possess arbitrary objects and a balloon is a reasonable object to carry.
2. Can this influence the vehicle's behavior? Yes, the autonomous vehicle may mistake the green balloon for a green traffic light signal, which could deceive it into driving forward when it should otherwise be stopped (e.g., at a red light).
3. Can the vehicle drive safely in its presence? No, this could deceive the vehicle into interpreting the green balloon as a legal traffic signal.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? Yes, this could deceive the autonomous vehicle into driving forward when it should otherwise be stopped (e.g., at a red light)
Classification: Anomaly.

Overall Scenario Classification: Anomaly."

I am driving on the road and I see:'''

for scenario in scenarios:
    scenario_folder = os.path.join(folder, "data", scenario) # scenario type

    with open(os.path.join(scenario_folder, "incorrect_labels.json"), 'r') as js:
        incorrect_labels = json.load(js)
    for exp in os.listdir(scenario_folder):
        if "exp" not in exp:
            continue

        exp_folder = os.path.join(scenario_folder, exp) # specific experiment
        vehicle_data = np.load(os.path.join(exp_folder, "vehicle_data.npz"))

        exp_responses = np.load(os.path.join(exp_folder, "image_data", response_file))
        for i, image in enumerate(vehicle_data["filenames"]):
            if image in incorrect_labels[exp]:
                continue # skip errenous response

            questions.append(vlm_prompt)
            answers.append(exp_responses["vlm_responses"][i])
            filenames.append(os.path.join(exp_folder, "image_data", exp_responses["image_names"][i])) # save absolute path
    
    np.savez(os.path.join(scenario_folder, dataset_file), questions = questions, answers = answers, filenames = filenames)




