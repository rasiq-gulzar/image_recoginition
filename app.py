from flask import Flask, render_template, request, send_file
import base64
from io import BytesIO
import numpy as np
from keras.preprocessing import image
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = 'numbers.pkl'  # Update this path to your model file
try:
    with open(model_path, 'rb') as file:
        classifier = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Model file '{model_path}' not found. Please ensure the file exists.")
    classifier = None

# Result mapping for digit classes
ResultMap = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
             5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

def preprocess_image(img_file):
    """
    Preprocess the uploaded image for prediction
    """
    try:
        # Save the uploaded file temporarily
        temp_path = 'temp_image.png'
        img_file.save(temp_path)
        
        # Load and preprocess the image
        test_image = image.load_img(temp_path, target_size=(64, 64))
        test_image_array = image.img_to_array(test_image)
        print(f"Image array shape: {test_image_array.shape}")
        
        # Reshape for prediction (add batch dimension)
        print('### Reshaping the image array as one single sample for prediction ###')
        test_image_array_exp_dim = np.expand_dims(test_image_array, axis=0)
        print(f"Expanded array shape: {test_image_array_exp_dim.shape}")
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return test_image_array_exp_dim
    
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_digit(processed_image):
    """
    Predict the digit from the processed image
    """
    if classifier is None:
        return "Model not loaded"
    
    try:
        # Make prediction
        result = classifier.predict(processed_image)
        predicted_class = np.argmax(result)
        predicted_digit = ResultMap[predicted_class]
        
        print(f'Prediction: This is {predicted_digit}')
        return predicted_digit
    
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return "Error in prediction"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    
    elif request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No image uploaded")
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return render_template('index.html', error="No image selected")
        
        try:
            # Preprocess the image
            processed_image = preprocess_image(image_file)
            
            if processed_image is not None:
                # Make prediction
                prediction = predict_digit(processed_image)
                return render_template('index.html', 
                                     image=image_file, 
                                     prediction=prediction)
            else:
                return render_template('index.html', 
                                     error="Error processing image")
        
        except Exception as e:
            return render_template('index.html', 
                                 error=f"Error: {str(e)}")
    
    else:
        return "Method Not Allowed", 405

# Optional: Add a test route for debugging
@app.route('/test')
def test_model():
    """
    Test route to verify model loading
    """
    if classifier is None:
        return "Model not loaded"
    else:
        return "Model loaded successfully"

if __name__ == '__main__':
    app.run(debug=True)