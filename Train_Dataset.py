import cv2
import os
import numpy as np

# Path to your dataset
dataset_path = r"T:\TARUN\EIE\Final Yr Proj\Dataset"

# Initialize LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_dict = {}   # name -> numeric ID
id_dict = {}      # numeric ID -> name
current_id = 0

# Loop through dataset folders
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue
    
    # Assign numeric label
    label_dict[person_name] = current_id
    id_dict[current_id] = person_name
    
    # Loop through each image in this person's folder
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Skipping {img_path}, cannot read")
            continue
        
        # Resize to 200x200 for consistency
        img = cv2.resize(img, (200, 200))
        
        faces.append(img)
        labels.append(current_id)
    
    current_id += 1

# Train LBPH recognizer
print("Training LBPH recognizer...")
recognizer.train(faces, np.array(labels))

# Save model
model_path = "lbph_model.yml"
recognizer.save(model_path)

print(f"Model saved as {model_path}")
print("Label mapping:", id_dict)
