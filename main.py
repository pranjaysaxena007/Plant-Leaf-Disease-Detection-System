import streamlit as st
import tensorflow as tf
import numpy as np

# #Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


#SIDEBAR
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",['Home','About','Disease Detection'])

#Home Page
if(app_mode == 'Home'):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path ='hpimg.jpg'
    st.image(image_path,use_column_width=True)
    st.markdown(<div style="text-align: justify";
    """Welcome to the Plant Disease Recognition System! 🌿🔍
    
In modern agriculture, plant health is crucial to ensuring high yields and sustainable farming practices. However, plant diseases remain a significant challenge, often leading to substantial losses if not detected and managed promptly. To address this, we present an advanced Plant Disease Detection System, designed to be both user-friendly and highly accurate.

This system leverages state-of-the-art technologies, including image processing and machine learning algorithms, to rapidly and efficiently identify diseases at early stages. With a simple, intuitive interface, it caters to farmers, agronomists, and researchers, allowing them to upload plant images and receive instant, reliable diagnoses. The system is optimized for high accuracy, reducing the margin for error, and ensuring efficient disease management to safeguard crops and improve agricultural productivity.

By combining ease of use with cutting-edge precision, this Plant Disease Detection System is set to revolutionize plant health monitoring, enabling smarter, faster, and more effective responses to plant diseases.
### How It Works
1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with any suspected diseases.
2. **Analysis:** Our system will process the image using advanced algorithms to identify potential disease.
3. **Results:** View the results and recommendations for further action.

### Why Choose Us?
**Why Choose This Plant Disease Detection App?**

1. **High Accuracy**: The app utilizes advanced machine learning models to provide highly accurate disease detection, ensuring reliable diagnoses for early intervention.

2. **User-Friendly Interface**: Designed for all users, from farmers to researchers, the app features a simple, intuitive interface that makes disease detection quick and easy.

3. **Fast and Efficient**: Detect plant diseases within seconds, allowing users to take immediate action to protect their crops and minimize losses.

4. **Early Detection Capability**: With its advanced technology, the app can identify diseases in the early stages, giving users the chance to implement treatment before significant damage occurs.

5. **Wide Range of Disease Identification**: The app supports the identification of numerous plant diseases across various crops, making it versatile and useful for different types of agriculture.

6. **Real-Time Results**: Instant feedback allows users to make timely decisions, saving time and resources compared to traditional diagnostic methods.

7. **Portable and Accessible**: Available on mobile devices, the app can be used anywhere, even in the field, making it a practical tool for on-the-go farmers and agronomists.

8. **Cost-Effective Solution**: By reducing the need for expert consultations or laboratory testing, the app provides a cost-effective way to monitor and manage plant health.

9. **Continuous Learning and Updates**: The app is regularly updated with new disease data and improved algorithms to keep up with the latest agricultural trends and challenges.

10. **Sustainability and Productivity**: By helping farmers detect and manage diseases efficiently, the app supports sustainable farming practices and contributes to higher crop yields.
### Get Started
Navigate to the **Disease Recognition** page in the sidebar to upload an image and witness the capabilities of our Plant Disease Recognition System! 
### About Us
Learn more about the project, and our goals on the **About** page."""
    </div>, unsafe_allow_html=True
    )

#About Project
elif(app_mode == 'About'):
    st.header('About')
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 images)
    3. Test (33 images)    
    """)

#Prediction Page
elif(app_mode == 'Disease Detection'):
    st.header("Disease Detection")
    test_image = st.file_uploader('Choose an image:')
    #Show Image Button
    try:
        if(st.button('Show Image')):
            st.image(test_image,use_column_width=True)
    except:
        st.write("Please INSERT an Image for Viewing")    
    #Predict button
    if(st.button('Predict')):
        try:
            
            with st.spinner("Wait for it..."):
                st.markdown("""#### Our prediction""")
                result_index = model_prediction(test_image)
                # Define Class
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
                st.success(f'''Model is Predicting it's a  ""{class_name[result_index]}"" ''')
                st.snow()
        except:
            st.write("Please INSERT an Image for Prediction")
            

