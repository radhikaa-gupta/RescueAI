# Mapping the Path to Recovery: Assessing Disaster Damage from Above
![image](https://github.com/radhikaa-gupta/RescueAI/assets/123308047/d5467a5c-ef01-4bb1-9e14-f3ece2f7edd4)

## Overview

This project aims to provide a quick and accurate assessment of disaster-affected areas using satellite imagery and AI. The system analyzes 'Before' and 'After' disaster images to identify affected regions, assess damage severity, and prioritize rescue efforts.

## Features

- **Image Upload**: Upload satellite images depicting the location before, during, and after the disaster.
- **Damage Assessment**: Analyze uploaded images to generate a comprehensive damage assessment report.
- **Interactive Map**: Visualize affected areas and their severity on an interactive map.

## Screenshots
<p float="left">
  <img src="https://github.com/radhikaa-gupta/RescueAI/assets/123308047/f9cab148-3924-44d0-a31a-74d9b020c4ce" width="200" />
  <img src="https://github.com/radhikaa-gupta/RescueAI/assets/123308047/234ad2ca-bf7f-4535-8870-f0f40bd54631" width="200" /> 
  <img src="https://github.com/radhikaa-gupta/RescueAI/assets/123308047/c1304df5-1d9b-4ac8-bc4b-b2054ad28a11" width="200" />
</p>



## Prerequisites

- Python 3.x
- API key for Google GenerativeAI

## Installation

```bash
git clone https://github.com/radhikaa-gupta/RescueAI
cd RescueAI
pip install -r requirements.txt
streamlit run rescueAi.py
```

## Setup

Obtain an API key for Google GenerativeAI and save it in a file named \`api_key.py\`:

```python
# api_key.py

api_key = "your_api_key_here"
```

## Usage

1. Upload 'Before' and 'After' disaster images using the file uploaders.
3. Click on the "Generate Analysis" button to analyze the images.
4. View the interactive map and damage assessment report generated by the AI.

## Contact

For more information or support, please contact [radhikagupta.contact@gmail.com](mailto:radhikagupta.contact@gmail.com).
