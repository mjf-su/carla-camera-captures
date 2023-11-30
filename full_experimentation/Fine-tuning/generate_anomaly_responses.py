import os
import numpy as np
import base64
import time
import requests

"""--- This file generates the ground truth classifications (using GPT4V) of nominal scenes to fine-tune a local VLM ---"""

project_folder = os.path.dirname(__file__)
target = "anomaly_stop_sign"
scenario_folder = os.path.join(project_folder, "data", target)

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

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def return_api_key(count): # count is number of queries from program
    if count < 100:
        return "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"
    elif count < 200:
        return "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"
    elif count < 300:
        return "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"
    elif count < 400:
        return "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"
    else:
        print("Error! Too many calls to GPT4V API today.")
        return "limit"

count = 0
for exp in os.listdir(scenario_folder):
    if "exp" not in exp:
        continue
    
    vlm_responses = []
    images = []
    exp_folder = os.path.join(scenario_folder, exp)
    image_folder = os.path.join(exp_folder, "image_data")
    vehicle_data = np.load(os.path.join(exp_folder, "vehicle_data.npz"))

    response_file = os.path.join(image_folder, "gpt4v_responses_newFormat.npz")
    anomaly_detected = vehicle_data["anomaly_detected"]
    filenames = vehicle_data["filenames"]
    for image in filenames[anomaly_detected]:

        # Getting the base64 string
        base64_image = encode_image(os.path.join(image_folder, image))
        api_key = return_api_key(count)

        if api_key == "limit":
            break
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": vlm_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                    ]
                }
                ],
                "max_tokens": 3000
            }

            done = False
            while not done:
                try:
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    answer = response.json()["choices"][0]["message"]["content"]
                    vlm_responses.append(answer)
                    images.append(image)
                    done = True
                    time.sleep(30) # stay under 10k TPM limit
                    
                    print("Classified on " + image + " in " + exp)
                except:
                    print("OpenAI server error... Let's try again.")
                    time.sleep(2)
            count += 1
    np.savez(response_file, gpt4v_responses = vlm_responses, image_names = images, vlm_prompt = vlm_prompt)