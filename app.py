import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import io
import matplotlib.pyplot as plt
import uuid
import json
import datetime
import base64
from io import BytesIO
import os
from model import CVLANet, load_model
from utils import (
    preprocess_image, 
    get_disease_info, 
    create_prediction_chart,
    get_medical_help_content,
    display_help_bubble,
    create_glossary_term,
    create_medical_insights_sidebar
)

# Page configuration
st.set_page_config(
    page_title="Skin Disease Prediction",
    page_icon="🔬",
    layout="wide"
)

# Application title and description
st.title("Skin Disease Prediction System")
st.markdown("""
This application uses a Convolutional Vision Long Attention Network (CVLAN) to predict skin diseases from uploaded images.
The system employs sophisticated image processing techniques with advanced computer vision features:

- **Advanced Feature Extraction**: Color, texture, and shape analysis
- **GLCM Texture Analysis**: Gray-Level Co-occurrence Matrix for texture assessment
- **Attention Mechanisms**: Focused processing on important image regions
- **Adaptive Feature Weighting**: Different weights for different disease patterns
- **Real-time Prediction**: Instant analysis of uploaded images

The model can identify the following conditions:
- Hyperpigmentation
- Acne
- Nail Psoriasis
- Vitiligo
- SJS-TEN (Stevens-Johnson Syndrome and Toxic Epidermal Necrolysis)
""")

# Initialize session state for storing prediction results
if 'predictions_dict' not in st.session_state:
    st.session_state.predictions_dict = {}
if 'images_dict' not in st.session_state:
    st.session_state.images_dict = {}
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'comparison_images' not in st.session_state:
    st.session_state.comparison_images = {
        'before': {'image': None, 'filename': None, 'enhanced': None},
        'after': {'image': None, 'filename': None, 'enhanced': None}
    }
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_analyses': 0,
        'condition_counts': {condition: 0 for condition in ['Hyperpigmentation', 'Acne', 'Nail Psoriasis', 'Vitiligo', 'SJS-TEN']},
        'last_analysis_time': None
    }
if 'filtered_image' not in st.session_state:
    st.session_state.filtered_image = None

# Load the model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Define the class names
class_names = ['Hyperpigmentation', 'Acne', 'Nail Psoriasis', 'Vitiligo', 'SJS-TEN']

# Create tabs for all application features
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Multi-Image Analysis", 
    "Image Comparison", 
    "History & Reports",
    "Statistics",
    "Advanced Features"
])

# ----- MULTI-IMAGE ANALYSIS ------
with tab1:
    st.subheader("Upload multiple images for batch analysis")
    
    # Add contextual help for batch analysis
    display_help_bubble(
        "batch_image_analysis", 
        "procedures", 
        icon="📷", 
        location="main"
    )
    
    # Create a unique id for this file uploader
    multi_files_id = "multi_image_uploader"
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key=multi_files_id
    )
    
    if uploaded_files:
        # Display a message with the number of images
        st.write(f"Total images uploaded: {len(uploaded_files)}")
        
        # Display thumbnails in a grid (3 columns)
        cols = st.columns(3)
        
        # Button to analyze all images
        if st.button("Analyze All Images", key="analyze_all"):
            # Clear previous multi-image predictions
            for key in list(st.session_state.predictions_dict.keys()):
                if key.startswith("multi_"):
                    st.session_state.predictions_dict.pop(key, None)
                    st.session_state.images_dict.pop(key, None)
            
            # Process each image with a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing image {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Read the file
                    image_bytes = uploaded_file.getvalue()
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert PIL image to numpy array
                    img_array = np.array(pil_image)
                    
                    # Generate unique ID for this image
                    image_id = f"multi_{uuid.uuid4().hex}"
                    
                    # Store in session state
                    st.session_state.images_dict[image_id] = {
                        "original": img_array,
                        "pil_image": pil_image,
                        "filename": uploaded_file.name
                    }
                    
                    # Preprocess the image
                    processed_image = preprocess_image(img_array)
                    
                    # Add batch dimension for model input
                    input_image = np.expand_dims(processed_image, axis=0)
                    
                    # Get prediction from model
                    predictions = model.predict(input_image)[0]
                    
                    # Store predictions in session state
                    st.session_state.predictions_dict[image_id] = predictions
                    
                    # Update stats
                    st.session_state.stats['total_analyses'] += 1
                    pred_class = np.argmax(predictions)
                    pred_class_name = class_names[pred_class]
                    st.session_state.stats['condition_counts'][pred_class_name] += 1
                    st.session_state.stats['last_analysis_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Add to history
                    confidence = predictions[pred_class] * 100
                    history_entry = {
                        "id": image_id,
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded_file.name,
                        "prediction": pred_class_name,
                        "confidence": confidence
                    }
                    st.session_state.history.append(history_entry)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"Error processing image {uploaded_file.name}: {e}")
            
            status_text.text("All images processed successfully!")
            st.session_state.current_tab = "multi"
            st.rerun()
        
        # Display thumbnails
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 3]:
                try:
                    # Display thumbnail
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, width=150)
                except Exception as e:
                    st.error(f"Error displaying thumbnail: {e}")
    
    # Display multi-image prediction results
    if st.session_state.current_tab == "multi":
        # Get all multi image predictions
        multi_results = [(k, v) for k, v in st.session_state.predictions_dict.items() if k.startswith("multi_")]
        
        if multi_results:
            st.subheader("Batch Analysis Results")
            
            # Create a table with predictions
            results_df_data = []
            
            for image_id, predictions in multi_results:
                if image_id in st.session_state.images_dict:
                    # Get the predicted class
                    pred_class = np.argmax(predictions)
                    pred_class_name = class_names[pred_class]
                    confidence = predictions[pred_class] * 100
                    
                    # Add to results
                    results_df_data.append({
                        "Filename": st.session_state.images_dict[image_id]["filename"],
                        "Predicted Condition": pred_class_name,
                        "Confidence": f"{confidence:.2f}%",
                        "Image ID": image_id
                    })
            
            # Display results in expanders, one per image
            for result in results_df_data:
                image_id = result["Image ID"]
                with st.expander(f"{result['Filename']} - {result['Predicted Condition']} ({result['Confidence']})"):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Display the original image
                        st.image(
                            st.session_state.images_dict[image_id]["original"], 
                            caption=f"Image: {result['Filename']}", 
                            use_container_width=True
                        )
                    
                    with col2:
                        # Get the predictions
                        predictions = st.session_state.predictions_dict[image_id]
                        
                        # Create and display chart
                        fig = create_prediction_chart(predictions, class_names)
                        st.pyplot(fig)
                        
                        # Get disease info
                        pred_class_name = result["Predicted Condition"]
                        disease_info = get_disease_info(pred_class_name)
                        
                        # Show brief description
                        st.markdown(f"**About {pred_class_name}:**")
                        st.write(disease_info["description"][:200] + "...")

# ----- IMAGE COMPARISON ------
with tab2:
    st.subheader("Image Comparison Tool")
    st.write("Upload before and after images to compare treatment progress")
    
    # Add contextual help for image comparison
    display_help_bubble(
        "image_comparison", 
        "procedures", 
        icon="⚖️", 
        location="main"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Before Treatment")
        before_image = st.file_uploader("Upload 'before' image", type=["jpg", "jpeg", "png"], key="before_uploader")
        if before_image:
            try:
                before_img = Image.open(before_image)
                st.session_state.comparison_images['before'] = {
                    'image': np.array(before_img),
                    'filename': before_image.name
                }
                st.image(before_img, caption="Before Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    with col2:
        st.write("### After Treatment")
        after_image = st.file_uploader("Upload 'after' image", type=["jpg", "jpeg", "png"], key="after_uploader")
        if after_image:
            try:
                after_img = Image.open(after_image)
                st.session_state.comparison_images['after'] = {
                    'image': np.array(after_img),
                    'filename': after_image.name
                }
                st.image(after_img, caption="After Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    # Image enhancement options
    if st.session_state.comparison_images['before'] is not None and st.session_state.comparison_images['after'] is not None:
        st.subheader("Image Enhancement")
        enhancement_options = st.multiselect(
            "Select enhancement options to apply:",
            ["Brightness", "Contrast", "Sharpness", "Color Balance", "Edge Detection"],
            key="enhance_options"
        )
        
        if enhancement_options:
            enhance_col1, enhance_col2 = st.columns(2)
            
            # Apply enhancements to before image
            if "Brightness" in enhancement_options:
                with enhance_col1:
                    before_brightness = st.slider("Before Image Brightness", 0.5, 2.0, 1.0, 0.1, key="before_brightness")
                    before_img_pil = Image.fromarray(st.session_state.comparison_images['before']['image'])
                    before_enhanced = ImageEnhance.Brightness(before_img_pil).enhance(before_brightness)
                    st.session_state.comparison_images['before']['enhanced'] = np.array(before_enhanced)
            
            if "Contrast" in enhancement_options:
                with enhance_col1:
                    before_contrast = st.slider("Before Image Contrast", 0.5, 2.0, 1.0, 0.1, key="before_contrast")
                    before_img_pil = Image.fromarray(st.session_state.comparison_images['before'].get('enhanced', 
                                                   st.session_state.comparison_images['before']['image']))
                    before_enhanced = ImageEnhance.Contrast(before_img_pil).enhance(before_contrast)
                    st.session_state.comparison_images['before']['enhanced'] = np.array(before_enhanced)
            
            # Apply enhancements to after image
            if "Brightness" in enhancement_options:
                with enhance_col2:
                    after_brightness = st.slider("After Image Brightness", 0.5, 2.0, 1.0, 0.1, key="after_brightness")
                    after_img_pil = Image.fromarray(st.session_state.comparison_images['after']['image'])
                    after_enhanced = ImageEnhance.Brightness(after_img_pil).enhance(after_brightness)
                    st.session_state.comparison_images['after']['enhanced'] = np.array(after_enhanced)
            
            if "Contrast" in enhancement_options:
                with enhance_col2:
                    after_contrast = st.slider("After Image Contrast", 0.5, 2.0, 1.0, 0.1, key="after_contrast")
                    after_img_pil = Image.fromarray(st.session_state.comparison_images['after'].get('enhanced', 
                                                  st.session_state.comparison_images['after']['image']))
                    after_enhanced = ImageEnhance.Contrast(after_img_pil).enhance(after_contrast)
                    st.session_state.comparison_images['after']['enhanced'] = np.array(after_enhanced)
            
            # Apply sharpness
            if "Sharpness" in enhancement_options:
                with enhance_col1:
                    before_sharpness = st.slider("Before Image Sharpness", 0.5, 3.0, 1.0, 0.1, key="before_sharpness")
                    before_img_pil = Image.fromarray(st.session_state.comparison_images['before'].get('enhanced', 
                                                   st.session_state.comparison_images['before']['image']))
                    before_enhanced = ImageEnhance.Sharpness(before_img_pil).enhance(before_sharpness)
                    st.session_state.comparison_images['before']['enhanced'] = np.array(before_enhanced)
                
                with enhance_col2:
                    after_sharpness = st.slider("After Image Sharpness", 0.5, 3.0, 1.0, 0.1, key="after_sharpness")
                    after_img_pil = Image.fromarray(st.session_state.comparison_images['after'].get('enhanced', 
                                                  st.session_state.comparison_images['after']['image']))
                    after_enhanced = ImageEnhance.Sharpness(after_img_pil).enhance(after_sharpness)
                    st.session_state.comparison_images['after']['enhanced'] = np.array(after_enhanced)
            
            # Apply color balance
            if "Color Balance" in enhancement_options:
                with enhance_col1:
                    before_color = st.slider("Before Image Color", 0.5, 2.0, 1.0, 0.1, key="before_color")
                    before_img_pil = Image.fromarray(st.session_state.comparison_images['before'].get('enhanced', 
                                                   st.session_state.comparison_images['before']['image']))
                    before_enhanced = ImageEnhance.Color(before_img_pil).enhance(before_color)
                    st.session_state.comparison_images['before']['enhanced'] = np.array(before_enhanced)
                
                with enhance_col2:
                    after_color = st.slider("After Image Color", 0.5, 2.0, 1.0, 0.1, key="after_color")
                    after_img_pil = Image.fromarray(st.session_state.comparison_images['after'].get('enhanced', 
                                                  st.session_state.comparison_images['after']['image']))
                    after_enhanced = ImageEnhance.Color(after_img_pil).enhance(after_color)
                    st.session_state.comparison_images['after']['enhanced'] = np.array(after_enhanced)
            
            # Apply edge detection
            if "Edge Detection" in enhancement_options:
                with enhance_col1:
                    before_img_pil = Image.fromarray(st.session_state.comparison_images['before'].get('enhanced', 
                                                   st.session_state.comparison_images['before']['image']))
                    before_edges = before_img_pil.filter(ImageFilter.FIND_EDGES)
                    st.session_state.comparison_images['before']['enhanced'] = np.array(before_edges)
                
                with enhance_col2:
                    after_img_pil = Image.fromarray(st.session_state.comparison_images['after'].get('enhanced', 
                                                  st.session_state.comparison_images['after']['image']))
                    after_edges = after_img_pil.filter(ImageFilter.FIND_EDGES)
                    st.session_state.comparison_images['after']['enhanced'] = np.array(after_edges)
            
            # Show enhanced images side by side
            st.subheader("Enhanced Images")
            compare_col1, compare_col2 = st.columns(2)
            
            with compare_col1:
                before_display = st.session_state.comparison_images['before'].get('enhanced', 
                                                                             st.session_state.comparison_images['before']['image'])
                st.image(before_display, caption="Enhanced Before Image", use_container_width=True)
            
            with compare_col2:
                after_display = st.session_state.comparison_images['after'].get('enhanced', 
                                                                           st.session_state.comparison_images['after']['image'])
                st.image(after_display, caption="Enhanced After Image", use_container_width=True)
            
            # Generate analysis button
            if st.button("Analyze Changes", key="analyze_changes"):
                st.subheader("Change Analysis")
                st.write("Analyzing visual differences between before and after images...")
                
                # Calculate a sample difference percentage (this is a simplified example)
                before_img = cv2.cvtColor(st.session_state.comparison_images['before']['image'], cv2.COLOR_RGB2GRAY)
                after_img = cv2.cvtColor(st.session_state.comparison_images['after']['image'], cv2.COLOR_RGB2GRAY)
                
                # Resize to same dimensions if needed
                if before_img.shape != after_img.shape:
                    after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))
                
                # Calculate absolute difference (simplified)
                diff = cv2.absdiff(before_img, after_img)
                
                # Calculate a pseudo-improvement score (higher is better)
                non_zero = np.count_nonzero(diff)
                total_pixels = diff.size
                change_percentage = (non_zero / total_pixels) * 100
                
                st.write(f"Detected changes: Approximately {change_percentage:.2f}% of the image area shows changes")
                
                # Visual progress bar
                st.progress(min(change_percentage / 100, 1.0))
                
                # Add notes section
                st.text_area("Add notes about observed changes:", key="comparison_notes", height=100)
                
                # Save to history button
                if st.button("Save Comparison to History"):
                    new_comparison = {
                        "id": f"comparison_{uuid.uuid4().hex}",
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "before_filename": st.session_state.comparison_images['before']['filename'],
                        "after_filename": st.session_state.comparison_images['after']['filename'],
                        "change_percentage": change_percentage,
                        "notes": st.session_state.get("comparison_notes", "")
                    }
                    
                    st.session_state.history.append(new_comparison)
                    st.success("Comparison saved to history!")

# ----- HISTORY & REPORTS ------
with tab3:
    st.subheader("Analysis History & Reports")
    
    # Add contextual help for history section
    display_help_bubble(
        "history_reports", 
        "general", 
        icon="📋", 
        location="main"
    )
    
    # Display history of analyses
    if not st.session_state.history:
        st.info("No analysis history available yet. Use the prediction tools to generate some analysis results.")
    else:
        st.write(f"**Total analyses saved: {len(st.session_state.history)}**")
        
        # Sort history by date (newest first)
        sorted_history = sorted(st.session_state.history, key=lambda x: x["date"], reverse=True)
        
        for idx, entry in enumerate(sorted_history):
            with st.expander(f"Analysis {idx+1}: {entry['date']}"):
                if "change_percentage" in entry:
                    # This is a comparison entry
                    st.write(f"**Comparison Analysis**")
                    st.write(f"Before image: {entry['before_filename']}")
                    st.write(f"After image: {entry['after_filename']}")
                    st.write(f"Detected changes: {entry['change_percentage']:.2f}%")
                    
                    if entry.get("notes"):
                        st.write("**Notes:**")
                        st.write(entry["notes"])
                else:
                    # This is a prediction entry
                    st.write(f"**Disease Prediction**")
                    st.write(f"Image: {entry['filename']}")
                    st.write(f"Predicted condition: {entry['prediction']}")
                    st.write(f"Confidence: {entry['confidence']:.2f}%")
                
                # Generate report button
                if st.button("Generate PDF Report", key=f"gen_report_{idx}"):
                    st.write("Preparing downloadable PDF report...")
                    
                    # Create HTML-like content for the report
                    report_content = f"""
                    <h1>Skin Disease Analysis Report</h1>
                    <p>Date: {entry['date']}</p>
                    """
                    
                    if "change_percentage" in entry:
                        report_content += f"""
                        <h2>Image Comparison Analysis</h2>
                        <p>Before image: {entry['before_filename']}</p>
                        <p>After image: {entry['after_filename']}</p>
                        <p>Detected changes: {entry['change_percentage']:.2f}%</p>
                        """
                        if entry.get("notes"):
                            report_content += f"<p>Notes: {entry['notes']}</p>"
                    else:
                        report_content += f"""
                        <h2>Disease Prediction Analysis</h2>
                        <p>Image: {entry['filename']}</p>
                        <p>Predicted condition: {entry['prediction']}</p>
                        <p>Confidence: {entry['confidence']:.2f}%</p>
                        """
                    
                    report_content += """
                    <h3>Medical Disclaimer</h3>
                    <p>This tool is meant for informational purposes only and should not replace
                    professional medical advice. Please consult with a healthcare provider for proper 
                    diagnosis and treatment.</p>
                    """
                    
                    # For demonstration purposes only - normally we'd use a PDF library
                    # to generate an actual PDF file
                    
                    def get_download_link(content, filename):
                        b64 = base64.b64encode(content.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Report</a>'
                        return href
                    
                    # Create download link
                    report_link = get_download_link(report_content, f"skin_analysis_report_{idx}.html")
                    st.markdown(report_link, unsafe_allow_html=True)
                    
                # Delete entry button
                if st.button("Delete Entry", key=f"del_entry_{idx}"):
                    st.session_state.history.remove(entry)
                    st.success("Entry deleted from history!")
                    st.rerun()
        
        # Clear history button
        if st.button("Clear All History"):
            st.session_state.history = []
            st.success("History cleared!")
            st.rerun()

# ----- STATISTICS ------
with tab4:
    st.subheader("Usage Statistics")
    
    # Add contextual help for statistics section
    display_help_bubble(
        "statistics", 
        "general", 
        icon="📊", 
        location="main"
    )
    
    # Extract stats from history
    total_analyses = len(st.session_state.history)
    
    # Count conditions predicted
    condition_counts = {condition: 0 for condition in class_names}
    for entry in st.session_state.history:
        if "prediction" in entry:
            condition = entry["prediction"]
            if condition in condition_counts:
                condition_counts[condition] += 1
    
    # Display overall stats
    st.write(f"**Total analyses performed: {total_analyses}**")
    
    # Create visualization of condition distribution
    if any(condition_counts.values()):
        st.subheader("Disease Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        conditions = list(condition_counts.keys())
        counts = list(condition_counts.values())
        
        ax.bar(conditions, counts)
        ax.set_xlabel('Skin Condition')
        ax.set_ylabel('Number of Cases')
        ax.set_title('Distribution of Detected Skin Conditions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Include system stats
    st.subheader("System Statistics")
    st.write(f"Application started: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    
    # Usage pattern by time (mocked for demo)
    st.subheader("Usage Pattern")
    
    # Create mock data
    dates = [datetime.datetime.now() - datetime.timedelta(days=x) for x in range(7)]
    usage_counts = [4, 7, 2, 8, 5, 3, 6]  # Example data
    
    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([d.strftime('%m-%d') for d in dates], usage_counts)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Analyses')
    ax.set_title('Usage Pattern (Last 7 Days)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Tips section based on statistics
    st.subheader("Tips & Insights")
    most_common = max(condition_counts.items(), key=lambda x: x[1])[0] if any(condition_counts.values()) else None
    
    if most_common:
        st.info(f"""
        The most commonly detected condition is **{most_common}**. 
        
        Common treatments for {most_common} include:
        - {get_disease_info(most_common)['treatment'].split('.')[0]}.
        
        For more detailed information, please consult a healthcare professional.
        """)
    
    # Export statistics button
    if st.button("Export Statistics"):
        # Create a JSON string of the statistics
        stats_data = {
            "total_analyses": total_analyses,
            "condition_counts": condition_counts,
            "generated_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        stats_json = json.dumps(stats_data, indent=4)
        
        # Create download link
        b64 = base64.b64encode(stats_json.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="skin_analysis_stats.json">Download Statistics as JSON</a>'
        st.markdown(href, unsafe_allow_html=True)

# ----- ADVANCED FEATURES ------
with tab5:
    st.subheader("Advanced Analysis Features")
    
    # Add help bubble at the top of advanced features
    display_help_bubble(
        "advanced_features", 
        "advanced_features", 
        icon="🔬", 
        location="main"
    )
    
    # Feature 1: Educational Medical Glossary
    with st.expander("Medical Educational Resources", expanded=True):
        st.write("""
        ### Interactive Medical Glossary
        
        This section provides educational resources to help you understand dermatological terms
        and concepts. Hover over highlighted terms to see definitions.
        """)
        
        # Create two columns
        glossary_col1, glossary_col2 = st.columns(2)
        
        with glossary_col1:
            st.markdown("#### Skin Structure Terms")
            st.markdown(f"""
            The {create_glossary_term('epidermis', 'anatomy')} is the outermost layer of skin, 
            which provides a waterproof barrier and creates our skin tone.
            
            Below this lies the {create_glossary_term('dermis', 'anatomy')}, containing tough 
            connective tissue, hair follicles, and sweat glands.
            
            {create_glossary_term('Melanocytes', 'anatomy')} are the cells that produce 
            {create_glossary_term('melanin', 'conditions')}, the pigment responsible for 
            skin color.
            
            The {create_glossary_term('stratum corneum', 'anatomy')} is the outermost layer of 
            the epidermis consisting of dead cells that shed approximately every 2 weeks.
            """, unsafe_allow_html=True)
            
        with glossary_col2:
            st.markdown("#### Dermatological Condition Terms")
            st.markdown(f"""
            {create_glossary_term('Erythema', 'symptoms')} refers to redness of the skin caused 
            by increased blood flow.
            
            {create_glossary_term('Hyperpigmentation', 'conditions')} occurs when patches of 
            skin become darker due to excess melanin production.
            
            {create_glossary_term('Pruritus', 'symptoms')} is the medical term for itching, 
            a common symptom in many skin conditions.
            
            {create_glossary_term('Vesicles', 'symptoms')} are small, fluid-filled blisters 
            that can appear in various skin conditions like eczema or herpes.
            """, unsafe_allow_html=True)
        
        # Display a sample custom medical insights sidebar
        if st.button("Show Sample Medical Insights", key="show_insights"):
            custom_insights = {
                "Understanding Medical Terms": """
                Medical terminology can be confusing. Here are simple definitions:
                - **Acute**: Short-term, severe onset
                - **Chronic**: Long-lasting, persistent
                - **Topical**: Applied to the skin surface
                - **Systemic**: Affecting the entire body
                """,
                "When To Consult A Dermatologist": """
                Consider seeing a dermatologist if you experience:
                - Rash that doesn't respond to over-the-counter treatment
                - Sudden skin changes with no obvious cause
                - Moles that change in appearance
                - Severe or recurring conditions that affect daily life
                """
            }
            create_medical_insights_sidebar(custom_insights=custom_insights)
            st.success("Medical insights added to the sidebar. Check it out!")
    
    # Feature 2: Texture Analysis
    with st.expander("Texture Analysis"):
        st.write("""
        This tool provides advanced texture analysis of skin conditions using Gray Level Co-occurrence Matrix (GLCM) 
        features. Upload an image to analyze its texture properties which can be helpful for diagnosis.
        """)
        
        texture_file = st.file_uploader("Upload image for texture analysis", type=["jpg", "jpeg", "png"], key="texture_uploader")
        
        if texture_file:
            try:
                texture_img = Image.open(texture_file)
                texture_array = np.array(texture_img)
                
                # Display the uploaded image
                st.image(texture_img, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Analyze Texture", key="analyze_texture"):
                    # Convert to grayscale for GLCM analysis
                    if len(texture_array.shape) == 3:
                        gray_img = cv2.cvtColor(texture_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray_img = texture_array
                    
                    # Calculate GLCM properties
                    glcm = model._compute_glcm(gray_img)
                    contrast = model._compute_contrast(glcm)
                    homogeneity = model._compute_homogeneity(glcm)
                    energy = model._compute_energy(glcm)
                    correlation = model._compute_correlation(glcm)
                    
                    # Create metrics layout
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric("Contrast", f"{contrast:.4f}", 
                                 delta_color="inverse",
                                 help="Measures the local variations in the GLCM. Higher values indicate more contrast.")
                        
                        st.metric("Energy", f"{energy:.4f}", 
                                 help="Measures the uniformity. Higher values indicate more uniform textures.")
                    
                    with metric_col2:
                        st.metric("Homogeneity", f"{homogeneity:.4f}", 
                                 help="Measures the closeness of the distribution of elements in the GLCM to its diagonal.")
                        
                        st.metric("Correlation", f"{correlation:.4f}", 
                                 help="Measures the joint probability occurrence of the specified pixel pairs.")
                    
                    # Display interpretation
                    st.subheader("Texture Interpretation")
                    
                    # Contrast interpretation
                    if contrast > 0.5:
                        st.write("**High Contrast**: The skin condition shows significant variation in texture, which may indicate an active inflammatory process.")
                    else:
                        st.write("**Low Contrast**: The skin appears to have uniform texture patterns, suggesting either normal skin or a chronic condition.")
                    
                    # Homogeneity interpretation
                    if homogeneity > 0.7:
                        st.write("**High Homogeneity**: The texture is very uniform, which could indicate healthy skin or a diffuse condition.")
                    else:
                        st.write("**Low Homogeneity**: The texture shows significant variations, which could indicate lesions, scaling, or other disruptions.")
                    
                    # Create a visual representation
                    st.subheader("Visual Texture Analysis")
                    
                    # Apply edge detection to visualize texture boundaries
                    edges = cv2.Canny(gray_img, 50, 150)
                    
                    # Show original vs edges side by side
                    display_col1, display_col2 = st.columns(2)
                    
                    with display_col1:
                        st.image(gray_img, caption="Original (Grayscale)", use_container_width=True)
                    
                    with display_col2:
                        st.image(edges, caption="Texture Boundaries", use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    # Feature 2: Region of Interest Analysis
    with st.expander("Region of Interest (ROI) Analysis"):
        st.write("""
        This tool allows you to select a specific region of an image for focused analysis. 
        This can be particularly useful for analyzing targeted areas of a skin condition.
        """)
        
        roi_file = st.file_uploader("Upload image for ROI analysis", type=["jpg", "jpeg", "png"], key="roi_uploader")
        
        if roi_file:
            try:
                roi_img = Image.open(roi_file)
                roi_array = np.array(roi_img)
                
                # Display the uploaded image
                st.image(roi_img, caption="Uploaded Image", use_container_width=True)
                
                # ROI selection options
                st.write("### Select Region of Interest")
                st.write("Use sliders to define the region of interest")
                
                # Get image dimensions
                height, width = roi_array.shape[:2]
                
                # Create sliders for ROI selection
                col1, col2 = st.columns(2)
                
                with col1:
                    x_start = st.slider("X Start", 0, width-10, int(width*0.25), key="x_start")
                    y_start = st.slider("Y Start", 0, height-10, int(height*0.25), key="y_start")
                
                with col2:
                    x_end = st.slider("X End", x_start+10, width, int(width*0.75), key="x_end")
                    y_end = st.slider("Y End", y_start+10, height, int(height*0.75), key="y_end")
                
                # Create a copy of the image with ROI highlighted
                highlighted_img = roi_array.copy()
                highlighted_img[y_start:y_end, x_start, :] = [255, 0, 0]  # Left edge
                highlighted_img[y_start:y_end, x_end-1, :] = [255, 0, 0]  # Right edge
                highlighted_img[y_start, x_start:x_end, :] = [255, 0, 0]  # Top edge
                highlighted_img[y_end-1, x_start:x_end, :] = [255, 0, 0]  # Bottom edge
                
                # Display image with ROI highlighted
                st.image(highlighted_img, caption="Selected ROI", use_container_width=True)
                
                if st.button("Analyze ROI", key="analyze_roi"):
                    # Extract the ROI
                    roi = roi_array[y_start:y_end, x_start:x_end]
                    
                    # Display the ROI
                    st.image(roi, caption="Extracted ROI", use_container_width=True)
                    
                    # Process the ROI for analysis
                    processed_roi = preprocess_image(roi)
                    input_roi = np.expand_dims(processed_roi, axis=0)
                    
                    # Make prediction on ROI
                    predictions = model.predict(input_roi)[0]
                    
                    # Display prediction results
                    st.subheader("ROI Analysis Results")
                    
                    # Get the predicted class
                    pred_class = np.argmax(predictions)
                    pred_class_name = class_names[pred_class]
                    confidence = predictions[pred_class] * 100
                    
                    # Display prediction and confidence
                    st.markdown(f"### Predicted Condition: **{pred_class_name}**")
                    st.markdown(f"### Confidence: **{confidence:.2f}%**")
                    
                    # Create and display chart
                    fig = create_prediction_chart(predictions, class_names)
                    st.pyplot(fig)
                    
                    # Compare with full image analysis
                    st.subheader("Comparison with Full Image Analysis")
                    
                    # Process the full image
                    processed_full = preprocess_image(roi_array)
                    input_full = np.expand_dims(processed_full, axis=0)
                    
                    # Make prediction on full image
                    full_predictions = model.predict(input_full)[0]
                    
                    # Get the predicted class for full image
                    full_pred_class = np.argmax(full_predictions)
                    full_pred_class_name = class_names[full_pred_class]
                    full_confidence = full_predictions[full_pred_class] * 100
                    
                    # Display full image prediction
                    st.markdown(f"Full Image Prediction: **{full_pred_class_name}** (Confidence: {full_confidence:.2f}%)")
                    
                    # Compare results
                    if pred_class == full_pred_class:
                        st.success("ROI analysis confirms the full image diagnosis.")
                    else:
                        st.warning("""
                        ROI analysis shows a different condition than the full image analysis. 
                        This could indicate a mixed condition or that the selected region has distinct characteristics.
                        """)
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    # Feature 3: Color Analysis
    with st.expander("Color Profile Analysis"):
        st.write("""
        This tool analyzes the color distribution in skin images, which can help identify patterns
        specific to certain skin conditions based on their typical color profiles.
        """)
        
        color_file = st.file_uploader("Upload image for color analysis", type=["jpg", "jpeg", "png"], key="color_uploader")
        
        if color_file:
            try:
                color_img = Image.open(color_file)
                color_array = np.array(color_img)
                
                # Display the uploaded image
                st.image(color_img, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Analyze Color Profile", key="analyze_color"):
                    # Extract the RGB channels
                    if len(color_array.shape) == 3 and color_array.shape[2] >= 3:
                        r_channel = color_array[:,:,0]
                        g_channel = color_array[:,:,1]
                        b_channel = color_array[:,:,2]
                        
                        # Calculate color statistics
                        r_mean = np.mean(r_channel)
                        g_mean = np.mean(g_channel)
                        b_mean = np.mean(b_channel)
                        
                        r_std = np.std(r_channel)
                        g_std = np.std(g_channel)
                        b_std = np.std(b_channel)
                        
                        # Display color metrics
                        st.subheader("Color Profile Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Red Channel", f"{r_mean:.2f}", 
                                    help="Average intensity of the red channel (0-255)")
                            st.metric("Red Variation", f"{r_std:.2f}", 
                                    help="Standard deviation of red values")
                        
                        with col2:
                            st.metric("Green Channel", f"{g_mean:.2f}", 
                                    help="Average intensity of the green channel (0-255)")
                            st.metric("Green Variation", f"{g_std:.2f}", 
                                    help="Standard deviation of green values")
                        
                        with col3:
                            st.metric("Blue Channel", f"{b_mean:.2f}", 
                                    help="Average intensity of the blue channel (0-255)")
                            st.metric("Blue Variation", f"{b_std:.2f}", 
                                    help="Standard deviation of blue values")
                        
                        # Calculate and display color ratios
                        st.subheader("Color Ratios")
                        
                        rg_ratio = r_mean / g_mean if g_mean > 0 else 0
                        rb_ratio = r_mean / b_mean if b_mean > 0 else 0
                        gb_ratio = g_mean / b_mean if b_mean > 0 else 0
                        
                        ratio_col1, ratio_col2, ratio_col3 = st.columns(3)
                        
                        with ratio_col1:
                            st.metric("R/G Ratio", f"{rg_ratio:.2f}", 
                                    help="Ratio of red to green channels. Higher values indicate more redness.")
                        
                        with ratio_col2:
                            st.metric("R/B Ratio", f"{rb_ratio:.2f}", 
                                    help="Ratio of red to blue channels. Higher values can indicate inflammation.")
                        
                        with ratio_col3:
                            st.metric("G/B Ratio", f"{gb_ratio:.2f}", 
                                    help="Ratio of green to blue channels.")
                        
                        # Color interpretation
                        st.subheader("Color Profile Interpretation")
                        
                        # Interpret redness
                        if rg_ratio > 1.2:
                            st.warning("**High Redness Detected**: The image shows elevated red channel intensity, which can indicate inflammation, erythema, or active lesions.")
                        else:
                            st.info("**Normal Redness Levels**: The image shows normal red channel intensity relative to other colors.")
                        
                        # Interpret color variation
                        if r_std > 50 or g_std > 50 or b_std > 50:
                            st.info("**High Color Variation**: The image shows significant color variation, which can indicate a mixed condition or multiple features.")
                        else:
                            st.info("**Uniform Color Profile**: The image shows relatively uniform coloration, which may indicate a single consistent condition.")
                        
                        # Display color channels separately
                        st.subheader("Color Channel Visualization")
                        
                        # Create blank images for each channel
                        r_image = np.zeros_like(color_array)
                        g_image = np.zeros_like(color_array)
                        b_image = np.zeros_like(color_array)
                        
                        # Set the channels
                        r_image[:,:,0] = r_channel
                        g_image[:,:,1] = g_channel
                        b_image[:,:,2] = b_channel
                        
                        # Display channels
                        channel_col1, channel_col2, channel_col3 = st.columns(3)
                        
                        with channel_col1:
                            st.image(r_image, caption="Red Channel", use_container_width=True)
                        
                        with channel_col2:
                            st.image(g_image, caption="Green Channel", use_container_width=True)
                        
                        with channel_col3:
                            st.image(b_image, caption="Blue Channel", use_container_width=True)
                    else:
                        st.error("Uploaded image must be a color image with RGB channels.")
            
            except Exception as e:
                st.error(f"Error processing image: {e}")

# Disclaimer (shown on all tabs)
st.warning("""
**Medical Disclaimer**: This tool is meant for informational purposes only and should not replace
professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.
""")

# Reset button to clear all predictions
if st.button("Clear All Predictions", key="clear_all"):
    st.session_state.predictions_dict = {}
    st.session_state.images_dict = {}
    st.session_state.current_tab = None
    st.rerun()

# Footer
st.markdown("---")
st.markdown("© 2025 Skin Disease Prediction System | Powered by Convolutional Vision Long Attention Network")
