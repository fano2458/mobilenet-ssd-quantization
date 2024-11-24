import torch
import numpy as np
import cv2
from predictor import Predictor


model = torch.jit.load("path_to_scripted_model").eval()

orig_image = cv2.imread("path_to_an_image")
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

model_predictor = Predictor(model, 300, np.array([127, 127, 127]), 128.0, iou_threshold=0.45, candidate_size=200, sigma=0.5, device='cpu')

boxes, labels, probs = model_predictor.predict(image, 10, 0.4)
