import torch
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import gradio as gr
from ultralytics import YOLO
import os

# Force PyTorch to use CPU
device = torch.device("cpu")

# Configure Google AI API
# genai.configure(api_key='AIzaSyATgL92t9qBCe4eqFX1cSfyfkzKMooQM48')
# model = genai.GenerativeModel('gemini-2.0-flash')

genai.configure(api_key='ADD YOUR API KEY')

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Test Gemini API connection
try:
    test_response = model.generate_content("Say Hello!")
    print("Test Response from Gemini:", test_response.text)
except Exception as e:
    print("Error: Gemini API connection failed! Check API key.")
    print(e)
    exit()

class FoodDetectionSystem:
    def __init__(self, model_path):
        self.model = YOLO('best.pt')  # Load trained YOLOv8 model
        self.model.to(device)  # Ensure model runs on CPU

    def detect_food(self, image_path):
        """Detect food items in the image"""
        results = self.model(image_path)  # Run inference on image
        detected_items = []
        
        for result in results:
            boxes = result.boxes.cpu()  # Ensure bounding boxes are on CPU
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:  # Confidence threshold
                    detected_items.append(result.names[class_id])
        
        return list(set(detected_items))  # Remove duplicates

def generate_recipe(ingredients, calorie_requirement):
    """Generate recipe using Gemini AI"""
    prompt = f"""
    Create a healthy recipe using some or all of these ingredients: {', '.join(ingredients)}.
    The recipe should be approximately {calorie_requirement} calories suggest 2 recipies.
    Please provide:
    Recipi Number 
    1. Recipe name
    2. Ingredients list with quantities
    3. Step-by-step instructions
    4. Approximate calorie count per serving
    """
    
    response = model.generate_content(prompt)
    return response.text

def process_image_and_generate_recipe(image, calorie_requirement):
    """Main function to process image and generate recipe"""
    try:
        # Save uploaded image temporarily
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        
        print("Image saved at:", temp_path)
        print("food detection start")
        # Initialize and use food detection system
        detector = FoodDetectionSystem('path_to_your_trained_model.pt')
        detected_foods = detector.detect_food(temp_path)
        print(detected_foods)
        if not detected_foods:
            return "No food items detected in the image. Please try another image."
        
        # Generate recipe
        recipe = generate_recipe(detected_foods, calorie_requirement)
        
        # Clean up
        os.remove(temp_path)
        
        return f"Detected Foods: {', '.join(detected_foods)}\n\nGenerated Recipe:\n{recipe}"
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio interface
def create_interface():
    iface = gr.Interface(
        fn=process_image_and_generate_recipe,
        inputs=[
            gr.Image(type="pil", label="Upload Food Image"),
            gr.Number(label="Desired Calorie Count", value=500)
        ],
        outputs=gr.Textbox(label="Results"),
        title="Food Detection and Recipe Generator",
        description="Upload a food image to detect ingredients and generate a recipe based on your calorie requirements."
    )
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()
