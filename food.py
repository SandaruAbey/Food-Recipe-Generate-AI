import torch
import cv2
import numpy as np
import os
from PIL import Image
import google.generativeai as genai
from ultralytics import YOLO

print(torch.__version__) 
print(torch.cuda.is_available()) 

# Force PyTorch to use CPU
device = torch.device("cpu")

# Configure Google Gemini AI API
genai.configure(api_key='AIzaSyATgL92t9qBCe4eqFX1cSfyfkzKMooQM48')

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
        """Initialize YOLO model for food detection"""
        self.model = YOLO(model_path)  # Load trained YOLOv8 model
        self.model.to(device)  # Ensure model runs on CPU

    def detect_food(self, image_path):
        """Detect food items in the image and return detected items with bounding boxes."""
        results = self.model(image_path)  # Run inference on image
        detected_items = []
        image = cv2.imread(image_path)

        for result in results:
            print("Available Class Names:", result.names)  # Debugging print statement
            boxes = result.boxes.cpu().numpy()  # Ensure bounding boxes are on CPU
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:  # Confidence threshold
                    detected_food = result.names.get(class_id, "Unknown")  # Fix incorrect name mapping
                    detected_items.append(detected_food)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(image, detected_food, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display image with bounding boxes
        if COLAB_MODE:
            cv2_imshow(image)  # Use cv2_imshow in Google Colab
        else:
            cv2.imshow('Detected Food Items', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return list(set(detected_items))  # Remove duplicates


def generate_recipe(ingredients, calorie_requirement):
    """Generate recipe using Gemini AI"""

    if not ingredients:
        print("Error: No ingredients detected!")
        return "No ingredients detected!"

    print(f"Sending to Gemini - Ingredients: {', '.join(ingredients)}")
    print(f"Calorie Requirement: {calorie_requirement}")

    prompt = f"""
    Create a healthy recipe using some or all of these ingredients: {', '.join(ingredients)}.
    The recipe should be approximately {calorie_requirement} calories.
    Please provide:
    1. Recipe name
    2. Ingredients list with quantities
    3. Step-by-step instructions
    4. Approximate calorie count per serving
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Error: Gemini API failed to generate a response.")
        print(e)
        return "Recipe generation failed."


def main():
    """Main function to process image and generate recipe"""
    image_path = input("Enter image path: ")
    calorie_requirement = input("Enter desired calorie count: ")

    # Validate calorie input
    try:
        calorie_requirement = int(calorie_requirement)
    except ValueError:
        print("Error: Invalid calorie count. Please enter a number.")
        return

    # Check if the image file exists
    if not os.path.exists(image_path):
        print("Error: Image file does not exist.")
        return

    # Initialize food detection system
    detector = FoodDetectionSystem('best.pt')
    detected_foods = detector.detect_food(image_path)

    # Check if any food was detected
    print("Detected Foods:", detected_foods)
    if not detected_foods:
        print("No food items detected in the image. Please try another image.")
        return

    # Generate recipe using detected food items
    recipe = generate_recipe(detected_foods, calorie_requirement)
    print("\nGenerated Recipe:\n", recipe)


if __name__ == "__main__":
    main()