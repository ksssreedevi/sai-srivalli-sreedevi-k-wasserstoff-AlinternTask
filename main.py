import os
import time
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from models.segmentation_model import segment_image
from models.identification_model import identify_objects
from models.text_extraction import extract_text
from models.summerisation_model import summarize_attributes
from utils.visualize_segments import visualize_segments
from utils.postprocessing import save_objects
from utils.data_mapping import map_data, generate_output

matplotlib.use('Agg')

def process_image(image_path, output_dir='output'):
    """Process the uploaded image by segmenting, identifying, extracting, summarizing, and mapping data."""
    start_time = time.time()
    
    predictions = segment_image(image_path)
    visualize_segments(image_path, predictions, output_dir)
    
    object_data = save_objects(image_path, predictions, output_dir=output_dir)
    object_data = identify_objects(object_data)
    
    object_data = extract_text(object_data)
    object_data = summarize_attributes(object_data)
    
    mapped_data = map_data(object_data)
    generate_output(mapped_data)
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    
    return mapped_data

def main():
    st.title("Image Processing with Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        input_image_path = os.path.join('data/input_images', uploaded_file.name)
        with open(input_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        with st.spinner("Processing the image..."):
            output_dir = 'data/segmented_objects'
            mapped_data = process_image(input_image_path, output_dir)

        st.subheader("Segmentation Visualization")
        segmented_image_path = os.path.join(output_dir, 'visualized_image.png')
        if os.path.exists(segmented_image_path):
            img = plt.imread(segmented_image_path)
            st.image(img, caption='Segmented Image', use_column_width=True)
        
        st.subheader("Extracted Data")
        if mapped_data:
            df = pd.DataFrame(mapped_data.get('objects', []))
            st.dataframe(df)
    else:
        st.info("Please upload an image to process.")

if __name__ == "__main__":
    main()
