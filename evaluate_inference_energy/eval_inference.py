import os
import torch
import numpy as np
import cv2
import pandas as pd
from predictor import Predictor
from zeus.monitor import ZeusMonitor


# Define directories
checkpoints_dir = 'checkpoints/'
data_dir = 'data/'
output_dir = 'output/'
os.makedirs(output_dir, exist_ok=True)

label_path = 'data/voc.txt'
class_names = [name.strip() for name in open(label_path).readlines()]

# Generate a unique color for each class
np.random.seed(42)
class_colors = {cls: tuple(np.random.randint(0, 255, 3).tolist()) for cls in class_names}
model_files = [os.path.join(checkpoints_dir, file) for file in os.listdir(checkpoints_dir) if file.endswith('.pt')]
image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]

monitor = ZeusMonitor()

for model_file in model_files:
    print(f"Evaluating model: {model_file}")
    results_path = os.path.join(output_dir, model_file[12:-3])
    os.makedirs(results_path, exist_ok=True)
    
    model = torch.jit.load(model_file).eval()
    
    # Warm up
    input_img = torch.randn([1, 3, 300, 300])
    for i in range(10):
        with torch.no_grad():
            _ = model(input_img)

    results = []

    model_predictor = Predictor(model, 300, np.array([127, 127, 127]), 128.0, iou_threshold=0.3,
                                     candidate_size=200, sigma=0.5, device='cpu')
    
    for image_file in image_files:
        print(f"Processing image: {image_file}")
        
        orig_image = cv2.imread(image_file)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        inference_times_no_nms = []
        energy_consumptions = []

        for _ in range(10):
            monitor.begin_window("inference")
            boxes, labels, probs = model_predictor.predict(image, 10, 0.4)
            measurement = monitor.end_window("inference")
            inference_times_no_nms.append(measurement.time)
            energy_consumptions.append(measurement.total_energy)
        
        avg_inference_time = np.mean(inference_times_no_nms)
        avg_energy_consumption = np.mean(energy_consumptions)
        results.append({'Image': image_file, 
                        'Avg_Inference_Time': avg_inference_time, 
                        'Avg Energy': avg_energy_consumption})

        for i in range(boxes.size(0)):
            box = boxes[i, :]
            class_label = class_names[labels[i]]
            color = class_colors[class_label]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            label = f"{class_label}: {probs[i]:.2f}"
            cv2.putText(image, label,
                        (int(box[0])+20, int(box[1])+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 2)
        out_img_path = os.path.join(results_path, image_file[5:])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_img_path, image)
    # Save results to CSV
    model_name = os.path.basename(model_file).replace('.pt', '')
    output_csv = os.path.join(results_path, f'{model_name}_inference_times.csv')
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")
