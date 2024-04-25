import streamlit as st
from PIL import Image
import io
import google.generativeai as genai
from api_key import api_key
import re
import folium 
from streamlit_folium import folium_static  # Importing folium_static

#import pandas as pd

# Configure genai with api key
genai.configure(api_key=api_key)

# Updated system prompt
system_prompt = """
You are a foundation model equipped with a temporal Vision transformer, utilizing a self-supervised encoder developed with a Vision Transformer (ViT) architecture and Masked Autoencoder (MAE) learning strategy, employing an MSE loss function.

You will receive two satellite images of the same location as input, depicting the location before and after the disaster, in the same order.

Your task is to analyze these images and provide a comprehensive damage assessment report, including:

1. **Region Identification:** Identify and categorize areas affected by the disaster into four regions:
    - **Destroyed:** Areas with complete structural collapse and severe damage.
    - **Major damage:** Areas with significant structural damage and potential hazards.
    - **Minor damage:** Areas with visible damage but overall structural integrity.
    - **Undamaged:** Areas with no apparent damage.
    For each region, provide the approximate coordinates in the format [Latitude, Longitude]. You may identify multiple non-contiguous areas for each category.

2. **Damage Description:** Briefly describe the type of damage observed in each region (e.g., collapsed buildings, flooded areas, debris).

3. **Accessibility Analysis (Optional):** If possible, assess the accessibility of each region based on visible road conditions and potential blockages (e.g., debris, flooding).

4. **Prioritization Recommendations:** Based on the damage assessment and accessibility analysis, suggest a prioritized order for rescue and relief efforts, considering factors like severity of damage and potential for survivors.

**Example Output:**
**Destroyed:**
- [34.0522, -118.2437] - Extensive building collapse and debris.
- [34.0488, -118.2511] - Large fire damage and structural instability.
**Major damage:** 
- [34.0555, -118.2399] - Partial building collapse and road blockages.
**Minor damage:**
- [34.0612, -118.2456] - Roof damage and scattered debris.
**Undamaged:**
- [34.0701, -118.2380] - No visible damage, roads appear clear.
Give similar output for the provided region
**Prioritization Recommendations:**

1. Focus initial efforts on the Destroyed regions, particularly the area with potential fire damage due to increased risk.
2. Next, address the Major damage region, clearing road blockages to enable access and search for survivors.
3. Assess the Minor damage region for any hidden dangers or individuals requiring assistance.
4. The Undamaged region can serve as a staging area for rescue operations and temporary shelter.
"""

# Set up the model
generation_config = {
    "temperature": 0.6,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(
    model_name="gemini-1.0-pro-vision-latest",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Set the page configuration
st.set_page_config(page_title="Disaster Vision AI", page_icon=":robot", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    body {
        color: #333333;
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo and title
col1, col2 = st.columns([1, 3])
with col1:
    st.image(r"https://github.com/radhikaa-gupta/RescueAI/blob/main/assets/logo.jpg?raw=true", width=325)

with col2:
    st.title("Mapping the Path to Recovery: Assessing Disaster Damage from Above")

# Function to crop and resize image
def process_image(uploaded_file):
    try:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.crop((0, 0, min(image.size), min(image.size)))
        image = image.resize((256, 256))
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# File uploaders
st.header("Upload Images")
uploaded_file1 = st.file_uploader("Upload 'Before' Image", type=["jpeg", "png", "jpg"])
if uploaded_file1:
    uploaded_file1 = process_image(uploaded_file1)
    st.image(uploaded_file1, width=220, caption="Before Disaster")

uploaded_file2 = st.file_uploader("Upload 'After' Image", type=["jpeg", "png", "jpg"])
if uploaded_file2:
    uploaded_file2 = process_image(uploaded_file2)
    st.image(uploaded_file2, width=220, caption="After Disaster")

st.markdown("---")

# Function to extract coordinates from text
def extract_coordinates(text):
    pattern = r"\[(-?\d+\.\d+), (-?\d+\.\d+)\]"
    matches = re.findall(pattern, text)
    if matches:
        return [tuple(map(float, match)) for match in matches]
    else:
        return None

damage_map = None

# Generate analysis button
submit_button = st.button("Generate Analysis")
if submit_button:
    st.spinner("Generating analysis...")
    st.markdown("---")
    
    image_parts = []
    processed_images = [uploaded_file1, uploaded_file2]
    for image in processed_images:
        if image is not None:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            image_parts.append({
                "mime_type": "image/jpeg",
                "data": img_byte_arr
            })
        else:
            st.error("Error: Please upload all three images.")
            break

    prompt_parts = [
        image_parts[0],
        image_parts[1],
        system_prompt
    ]

    response = model.generate_content(prompt_parts)
    st.write("Model Response:")
    st.write(response.text)

    coordinates = extract_coordinates(response.text)
    if coordinates:
        center_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
        center_lon = sum(coord[1] for coord in coordinates) / len(coordinates)
        damage_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr="Google"
        )
    
    region_categories = {}
    lines = response.text.split("\n")
    for line in lines:
        if line.startswith("1)") or line.startswith("2)") or line.startswith("3)") or line.startswith("4)"):
            parts = line.split(")")
            category = parts[0].strip()
            description = parts[1].strip()
            coordinates = extract_coordinates(description)
            region_categories[category] = {"description": description, "coordinates": coordinates}

    color_map = {
        "1) Destroyed": "red",
        "2) Major damage": "orange",
        "3) Minor damage": "yellow",
        "4) Undamaged": "green"
    }
    
    # Adding markers and shading regions on the map
    for category, data in region_categories.items():
        description = data["description"]
        coordinates_list = data["coordinates"]
        
        # Add markers
        for coords in coordinates_list:
            lat, lon = coords
            marker = folium.Marker(
                location=[lat, lon],
                popup=f"Category: {category}",
                icon=folium.Icon(color=color_map[category])
            )
            marker.add_to(damage_map)

        # Shade regions
        folium.Polygon(
            locations=coordinates_list,
            color=color_map[category],
            fill=True,
            fill_color=color_map[category],
            fill_opacity=0.4,
            popup=f"Category: {category}"
        ).add_to(damage_map)

    # Displaying the map with a heading
    st.header(":world_map: :round_pushpin: Damage Assessment Map")
    folium_static(damage_map)

# About, How it Works, and Contact Us sections
st.markdown("---")
st.subheader(":information_source: )About the Project")
st.write("""
This project aims to provide a quick and accurate assessment of disaster-affected areas using satellite imagery and AI.
The system analyzes 'Before' and 'After' disaster images to identify affected regions, assess damage severity, and prioritize rescue efforts.
""")

st.markdown("---")
st.subheader(":gear: How It Works")
st.write("""
1. **Upload Images**: Upload satellite images depicting the location before, during, and after the disaster.
2. **Generate Analysis**: Click the button to analyze the uploaded images and get a comprehensive damage assessment report.
3. **View Results**: Check the map and region categories to understand the affected areas and their severity.
""")

st.markdown("---")
st.subheader(":email: Contact Us")
st.write("""
For more information or support, please contact us at [radhikagupta.contact@gmail.com](mailto:radhikagupta.contact@gmail.com).
""")
