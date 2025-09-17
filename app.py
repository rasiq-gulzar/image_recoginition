import streamlit as st
import numpy as np
from keras.preprocessing import image
import pickle
import os
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="🔢 Digit Recognition App",
    page_icon="🔢",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-result {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .stFileUploader > div > div > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'classifier.pkl'
    try:
        with open(model_path, 'rb') as file:
            classifier = pickle.load(file)
        st.success("✅ Model loaded successfully!")
        return classifier
    except FileNotFoundError:
        st.error(f"❌ Model file '{model_path}' not found. Please ensure the file exists.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Result mapping for digit classes
ResultMap = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

def preprocess_image(uploaded_file):
    """Preprocess the uploaded image for prediction"""
    try:
        # Convert uploaded file to PIL Image
        pil_image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Save temporarily for keras preprocessing
        temp_path = 'temp_image.png'
        pil_image.save(temp_path)
        
        # Load and preprocess with keras
        test_image = image.load_img(temp_path, target_size=(64, 64))
        test_image_array = image.img_to_array(test_image)
        
        # Reshape for prediction (add batch dimension)
        test_image_array_exp_dim = np.expand_dims(test_image_array, axis=0)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return test_image_array_exp_dim, pil_image
    
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None, None

def predict_digit(processed_image, classifier):
    """Predict the digit from the processed image"""
    if classifier is None:
        return "Model not loaded", 0.0
    
    try:
        # Make prediction
        result = classifier.predict(processed_image)
        predicted_class = np.argmax(result)
        predicted_digit = ResultMap[predicted_class]
        
        # Calculate confidence (probability of predicted class)
        confidence = float(np.max(result) * 100)
        
        return predicted_digit, confidence
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return "Error in prediction", 0.0

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🔢 Digit Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an image of a handwritten digit (0-9) and get instant AI-powered recognition!</p>', unsafe_allow_html=True)
    
    # Load model
    classifier = load_model()
    
    # Instructions
    with st.expander("📋 Instructions", expanded=False):
        st.write("""
        - Upload a clear image of a single handwritten digit
        - Supported formats: JPG, PNG, GIF
        - Best results with dark digits on light backgrounds  
        - Image will be automatically resized to 64x64 pixels
        - The AI model will predict the digit and show confidence score
        """)
    
    # File uploader
    st.markdown("### 📁 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'gif'],
        help="Upload a clear image of a handwritten digit"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Process and predict
        if st.button("🚀 Predict Digit", type="primary", use_container_width=True):
            if classifier is not None:
                with st.spinner("🔄 Processing your image..."):
                    # Preprocess image
                    processed_image, pil_image = preprocess_image(uploaded_file)
                    
                    if processed_image is not None:
                        # Make prediction
                        prediction, confidence = predict_digit(processed_image, classifier)
                        
                        if prediction != "Error in prediction" and prediction != "Model not loaded":
                            # Display result
                            st.markdown(f"""
                            <div class="result-box">
                                <h3>🎯 Prediction Result</h3>
                                <div class="prediction-result">{prediction}</div>
                                <p>The AI model predicts this digit is: <strong>{prediction}</strong></p>
                                <p class="confidence-score">Confidence: {confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show processed image info
                            st.success(f"✅ Successfully processed and predicted: **{prediction}** with {confidence:.1f}% confidence")
                            
                        else:
                            st.error(f"❌ {prediction}")
                    else:
                        st.error("❌ Failed to process the image. Please try a different image.")
            else:
                st.error("❌ Model not available. Please check if the model file exists.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with ❤️ using Streamlit</p>
        <p>Upload a digit image to see the magic! ✨</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
