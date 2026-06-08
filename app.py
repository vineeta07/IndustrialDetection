import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification

print("Loading AI Models... Please wait (it will automatically download the ViT model the first time).")

# --- 1. Load YOLO (Draws Bounding Boxes) ---
yolo_model = YOLO('best.pt') 

# --- 2. Load ViT (Classifies whole image) ---
vit_model_path = "dima806/surface_crack_image_detection"
vit_processor = ViTImageProcessor.from_pretrained(vit_model_path)
vit_model = ViTForImageClassification.from_pretrained(vit_model_path)

# --- 3. Camera Setup ---
# Using cv2.CAP_DSHOW (DirectShow) fixes the 'can't grab frame' MSMF error on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nModels loaded! Webcam is running! Press 'q' to close the window.")

frame_count = 0
last_annotated_frame = None

while True:
    ret, frame = cap.read()
    if not ret: break

    # Process every 2nd frame to keep it running smoothly on your CPU
    if frame_count % 2 == 0:
        
        # --- A. Run YOLO (Gets Bounding Boxes) ---
        # Added conf=0.15 to make it more aggressive at finding cracks
        results = yolo_model(frame, imgsz=640, conf=0.15, stream=True, verbose=False)
        annotated_frame = frame.copy()
        for r in results:
            annotated_frame = r.plot() # Draws the boxes
            
        # --- B. Run ViT (Gets the Global Text Label) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = vit_processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = vit_model(**inputs)
            
        predicted_class_idx = outputs.logits.argmax(-1).item()
        vit_label = vit_model.config.id2label[predicted_class_idx]
        
        # Color coding: Red if Crack/Positive, Green if Negative
        text_color = (0, 0, 255) if ("crack" in vit_label.lower() or "positive" in vit_label.lower()) else (0, 255, 0)

        # --- C. Combine Both ---
        # Draw the global label text at the top
        cv2.putText(annotated_frame, f"Crack Detection: {vit_label.upper()}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                    
        last_annotated_frame = annotated_frame

    frame_count += 1

    # Show the combined video frame
    if last_annotated_frame is not None:
        cv2.imshow("WeldSight Dual-AI Vision", last_annotated_frame)
    else:
        cv2.imshow("WeldSight Dual-AI Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()