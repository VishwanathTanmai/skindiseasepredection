import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an input image for the CVLAN model
    
    Args:
        image: Input image as numpy array
        target_size: Target size for model input
        
    Returns:
        Preprocessed image ready for model input
    """
    try:
        # Ensure image has 3 channels (RGB)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image to target size
        image = cv2.resize(image, target_size)
        
        # Normalize image to [0, 1] range
        image = image.astype(np.float32) / 255.0
        
        return image
        
    except Exception as e:
        # Fallback for deployment environments with limited OpenCV capabilities
        import numpy as np
        print(f"Warning: Using fallback image preprocessing: {e}")
        
        # Fallback preprocessing with numpy only
        if len(image.shape) == 2:  # Grayscale
            # Convert grayscale to RGB by duplicating the channel
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:  # RGBA
            # Remove alpha channel for RGBA images
            image = image[:, :, :3]
        
        # Simple resize using nearest neighbor interpolation
        h, w = target_size
        orig_h, orig_w = image.shape[0], image.shape[1]
        h_scale, w_scale = h / orig_h, w / orig_w
        
        resized = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(h):
            src_i = min(int(i / h_scale), orig_h - 1)
            for j in range(w):
                src_j = min(int(j / w_scale), orig_w - 1)
                resized[i, j] = image[src_i, src_j]
        
        # Normalize to [0, 1]
        if resized.max() > 1.0:
            resized = resized / 255.0
            
        return resized

def create_prediction_chart(predictions, class_names):
    """
    Create a bar chart visualization of prediction probabilities
    
    Args:
        predictions: Model prediction probabilities
        class_names: List of class names
        
    Returns:
        Matplotlib figure object with the visualization
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.barh(class_names, predictions * 100)
    
    # Color the bar for highest prediction differently
    max_idx = np.argmax(predictions)
    for i, bar in enumerate(bars):
        if i == max_idx:
            bar.set_color('#1f77b4')  # Highlight color
        else:
            bar.set_color('#d3d3d3')  # Light gray
    
    # Add percentage labels to the bars
    for i, v in enumerate(predictions):
        ax.text(v * 100 + 1, i, f'{v * 100:.1f}%', va='center')
    
    # Set chart title and labels
    ax.set_title('Prediction Confidence', fontsize=16)
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_xlim(0, 100)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Close any other figures to prevent memory leaks
    # This prevents the "More than 20 figures have been opened" warning
    plt.close('all')
    
    return fig

def get_medical_help_content():
    """
    Provides contextual medical help content for tooltips and help bubbles
    
    Returns:
        Dictionary containing medical help content for different topics
    """
    return {
        # Content for application section help bubbles
        "procedures": {
            "batch_image_analysis": "Upload multiple images at once to analyze several skin conditions simultaneously. This feature is ideal for tracking conditions over time or comparing different affected areas. The system will process each image individually and provide detailed results for each one.",
            "image_comparison": "Compare before and after treatment images to visualize improvements. This tool allows you to upload two images and apply various enhancement filters to better highlight changes. The system can calculate approximate change percentages between images.",
            "single_image_analysis": "Upload a single image for detailed analysis of a skin condition. The system will process the image and provide prediction probabilities for five different skin conditions along with educational information about the detected conditions."
        },
        
        "general": {
            "history_reports": "View and manage your previous analyses. This section stores all your previous image analyses and comparisons, allowing you to track conditions over time. You can generate downloadable reports from any analysis and delete entries you no longer need.",
            "statistics": "View aggregated statistics about your usage and analyses. This section shows distribution of detected conditions, usage patterns, and provides insights based on your most common conditions. You can also export statistics data for your records."
        },
        
        "advanced_features": {
            "advanced_features": "This section provides specialized analysis tools and educational resources for skin conditions. Features include texture analysis, region of interest (ROI) analysis, color profiling, and an interactive medical glossary with hover-over definitions for dermatological terms."
        },
        
        # Skin anatomy and structure terms
        "anatomy": {
            "epidermis": "The outermost layer of skin that serves as a protective barrier. It contains no blood vessels and is made up of 4-5 sublayers depending on location in the body.",
            "dermis": "The middle layer of skin beneath the epidermis. It contains blood vessels, nerve endings, hair follicles, and sweat glands.",
            "hypodermis": "The deepest layer of skin (also called subcutaneous tissue) that contains fat and connective tissue for insulation and padding.",
            "melanocytes": "Specialized cells in the epidermis that produce melanin, the pigment that gives skin its color and helps protect against UV radiation.",
            "keratinocytes": "The most common cell type in the epidermis that produces keratin, a protein that provides structure and waterproofing.",
            "sebaceous glands": "Oil-producing glands attached to hair follicles that secrete sebum to lubricate and waterproof the skin and hair.",
            "stratum corneum": "The outermost sublayer of the epidermis made up of dead cells that continuously shed and are replaced by new cells from below."
        },
        
        # Symptoms and clinical signs
        "symptoms": {
            "erythema": "Redness of the skin caused by increased blood flow to superficial capillaries. Often a sign of inflammation or infection.",
            "pruritus": "The medical term for itching, which can range from mild to severe and may be localized or generalized across the body.",
            "vesicles": "Small, fluid-filled blisters on the skin, typically less than 5mm in diameter.",
            "pustules": "Small, inflamed, pus-filled blisters similar to vesicles but containing purulent material.",
            "macules": "Flat, discolored spots on the skin that are neither raised nor depressed, like freckles or flat moles.",
            "papules": "Small, raised, solid bumps on the skin, usually less than 1cm in diameter.",
            "scales": "Thin, dry flakes of dead skin cells that may be white, silver, or gray in appearance."
        },
        
        # Dermatological conditions
        "conditions": {
            "melanin": "The pigment that gives human skin, hair, and eyes their color. Higher amounts of melanin result in darker skin, hair, and eye color.",
            "hyperpigmentation": "A condition where patches of skin become darker than surrounding areas due to excess melanin production. Can be caused by sun damage, inflammation, hormonal changes, or medications.",
            "vitiligo": "A chronic condition where the skin loses pigment cells (melanocytes), resulting in white patches that can appear anywhere on the body.",
            "psoriasis": "A chronic autoimmune condition that causes cells to build up rapidly on the skin's surface, forming itchy, scaly patches that can be painful.",
            "eczema": "A group of conditions that cause skin inflammation, itchiness, and rash-like symptoms. The most common type is atopic dermatitis.",
            "acne": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells, leading to whiteheads, blackheads, pimples, or deeper lumps.",
            "rosacea": "A chronic inflammatory skin condition primarily affecting the face, causing redness, visible blood vessels, and sometimes small pus-filled bumps."
        },
        
        # General dermatology terms
        "dermatology_terms": {
            "melanin": "The pigment that gives human skin, hair, and eyes their color. Higher amounts of melanin result in darker skin, hair, and eye color.",
            "erythema": "Redness of the skin caused by increased blood flow to superficial capillaries. Often a sign of inflammation or infection.",
            "lesion": "A broad term referring to any abnormal change in skin tissue, such as a wound, sore, rash, or growth.",
            "papule": "A small, raised, solid pimple or swelling on the skin that's usually less than 1 centimeter in diameter.",
            "macule": "A flat, discolored area of skin that's neither raised nor depressed, such as freckles or flat moles.",
            "nodule": "A solid, raised lesion that extends deeper into the dermis than a papule, usually greater than 1 centimeter in diameter.",
            "vesicle": "A small, fluid-filled blister on the skin.",
            "pustule": "A small, inflamed, pus-filled, blister-like lesion.",
            "wheal": "A temporary raised, itchy area of skin, often with a white center and a red halo, usually caused by an allergic reaction.",
            "scale": "Dry, thin pieces of skin that flake off; associated with many skin conditions like psoriasis, eczema, and fungal infections."
        },
        
        # Image analysis terms
        "image_analysis": {
            "texture_analysis": "The process of quantifying texture characteristics in an image, which can help identify patterns associated with specific skin conditions.",
            "GLCM": "Gray Level Co-occurrence Matrix - A statistical method for examining texture that considers the spatial relationship of pixels in the image.",
            "contrast": "In texture analysis, contrast measures local variations in the image. Higher values indicate more dynamic range and texture complexity.",
            "homogeneity": "Measures the closeness of the distribution of elements in the GLCM to its diagonal. Higher values indicate a more uniform texture.",
            "energy": "Represents the sum of squared elements in the GLCM, also known as uniformity. Higher values indicate fewer gray level transitions.",
            "correlation": "Measures the joint probability occurrence of specified pixel pairs. Higher values indicate more linear dependencies in the texture.",
            "color_profile": "The distribution and intensity of colors in an image, which can help identify conditions with characteristic color patterns.",
            "edge_detection": "Algorithmic technique to identify points in an image where brightness changes sharply, useful for finding boundaries of skin lesions."
        },
        
        # Diagnostic terms
        "diagnostic_terms": {
            "differential_diagnosis": "The process of weighing the probability of one disease versus that of other diseases that might account for a patient's symptoms.",
            "biopsy": "The removal of a small sample of tissue for examination under a microscope to determine the presence or extent of disease.",
            "dermatoscopy": "Examination of skin lesions with a dermatoscope, which provides illumination and magnification to better visualize features not visible to the naked eye.",
            "patch_test": "A procedure to determine whether a specific substance causes skin inflammation, used to diagnose contact dermatitis.",
            "wood's_lamp": "An examination tool that emits ultraviolet light, used to diagnose certain skin conditions by the characteristic fluorescence they produce.",
            "KOH_examination": "A laboratory test that uses potassium hydroxide to diagnose fungal infections of the skin, hair, or nails.",
            "tzanck_smear": "A diagnostic test for viral skin infections, particularly herpes and varicella.",
            "immunofluorescence": "A technique used to visualize specific proteins in tissues, helpful in diagnosing autoimmune skin diseases."
        },
        
        # Treatment terms
        "treatment_terms": {
            "topical": "Applied directly to a particular spot, usually on the skin.",
            "systemic": "Affecting the entire body, rather than a single organ or part. Systemic medications are taken orally or by injection.",
            "corticosteroids": "Anti-inflammatory medications that can be applied topically or taken systemically to treat various skin conditions.",
            "retinoids": "A class of compounds derived from vitamin A that regulate cell growth and are used to treat acne, psoriasis, and aging.",
            "immunosuppressants": "Medications that reduce the strength of the body's immune response, used in severe skin conditions with an autoimmune component.",
            "biologics": "Advanced drugs made from living organisms or their products, targeting specific parts of the immune system to treat severe skin diseases.",
            "phototherapy": "Treatment that uses different types of light (often narrowband UVB) to treat skin conditions like psoriasis, vitiligo, and eczema.",
            "laser_therapy": "The use of focused light therapy to treat skin conditions, including vascular lesions, pigmentation issues, and scarring.",
            "cryotherapy": "The use of extreme cold, usually liquid nitrogen, to destroy abnormal tissue."
        },
        
        # Prevention terms
        "prevention_terms": {
            "photoprotection": "The prevention of damage to the skin caused by UV radiation, primarily through sunscreens and protective clothing.",
            "SPF": "Sun Protection Factor - A measure of how well a sunscreen will protect skin from UVB rays, the kind that cause sunburn and contribute to skin cancer.",
            "broad_spectrum": "Refers to sunscreens that protect against both UVA and UVB rays.",
            "moisturizer": "Products that increase the skin's water content, helping to prevent dryness and maintain the skin barrier.",
            "emollient": "Substances that soften and smooth the skin by filling in spaces between skin flakes with droplets of oil.",
            "humectant": "Ingredients that attract water from the dermis into the epidermis or from the environment to the epidermis.",
            "occlusives": "Ingredients that create a physical barrier on the skin to reduce water loss.",
            "antioxidants": "Substances that inhibit oxidation and combat the effects of free radicals, which can damage skin cells."
        }
    }

def display_help_bubble(term, category, icon="ℹ️", location="sidebar"):
    """
    Displays a contextual help bubble/tooltip with medical information
    
    Args:
        term: The specific term to display help for
        category: Category in the medical help content dictionary
        icon: Icon to display before the term (default: information symbol)
        location: Where to display the help (sidebar or main)
    
    Returns:
        None, displays the help directly
    """
    help_content = get_medical_help_content()
    
    if category in help_content and term in help_content[category]:
        content = help_content[category][term]
        
        # Format the term for display
        formatted_term = f"{icon} {term.replace('_', ' ').title()}"
        
        # Display in the appropriate location
        if location == "sidebar":
            with st.sidebar:
                with st.expander(formatted_term):
                    st.markdown(content)
        else:
            with st.expander(formatted_term):
                st.markdown(content)
    else:
        if location == "sidebar":
            with st.sidebar:
                with st.expander(f"{icon} {term.replace('_', ' ').title()}"):
                    st.write("Information not available.")
        else:
            with st.expander(f"{icon} {term.replace('_', ' ').title()}"):
                st.write("Information not available.")

def create_medical_insights_sidebar(disease_name=None, custom_insights=None):
    """
    Creates a sidebar with relevant medical insights based on the detected condition
    or custom insights
    
    Args:
        disease_name: Name of the detected skin disease (optional)
        custom_insights: Dictionary of custom insights to display (optional)
        
    Returns:
        None, displays the sidebar directly
    """
    with st.sidebar:
        st.markdown("### 🔍 Medical Insights")
        
        if disease_name:
            disease_info = get_disease_info(disease_name)
            
            # Show common triggers if it's a hypersensitivity condition
            if disease_name in ["SJS-TEN"]:
                st.markdown("#### Common Medication Triggers:")
                st.markdown("""
                - Antibiotics (sulfonamides, penicillins)
                - Anticonvulsants (carbamazepine, lamotrigine)
                - NSAIDs (ibuprofen, naproxen)
                - Allopurinol (gout medication)
                - Nevirapine (HIV medication)
                """)
            
            # Show prevention tips for hyperpigmentation and acne
            if disease_name in ["Hyperpigmentation", "Acne"]:
                st.markdown("#### Prevention Tips:")
                
                if disease_name == "Hyperpigmentation":
                    st.markdown("""
                    - Use broad-spectrum sunscreen (SPF 30+)
                    - Avoid sun exposure during peak hours
                    - Wear protective clothing and hats
                    - Avoid picking or scratching skin lesions
                    - Treat inflammation promptly
                    """)
                elif disease_name == "Acne":
                    st.markdown("""
                    - Cleanse face twice daily with gentle cleanser
                    - Use non-comedogenic products
                    - Avoid touching your face
                    - Change pillowcases regularly
                    - Consider dietary triggers (dairy, high-glycemic foods)
                    """)
            
            # Show self-care tips for specific conditions
            st.markdown("#### Self-Care Strategies:")
            
            if disease_name == "Nail Psoriasis":
                st.markdown("""
                - Keep nails trimmed and clean
                - Use moisturizer on nails and cuticles
                - Avoid harsh chemicals and trauma to nails
                - Wear gloves for wet work or cleaning
                - Don't pick or manipulate affected nails
                """)
            elif disease_name == "Vitiligo":
                st.markdown("""
                - Use sunscreen on depigmented areas
                - Consider cosmetic camouflage products
                - Join support groups to address psychological impact
                - Maintain overall skin health
                - Consider vitamin D supplementation (consult your doctor)
                """)
            else:
                st.markdown("""
                - Follow prescribed treatment regimen
                - Keep affected areas clean and moisturized
                - Avoid potential irritants and triggers
                - Track symptoms and treatment efficacy
                - Consider stress management techniques
                """)
        
        # Display custom insights if provided
        if custom_insights:
            for title, content in custom_insights.items():
                st.markdown(f"#### {title}")
                st.markdown(content)
        
        # Always show general resources
        with st.expander("Resources for Patients"):
            st.markdown("""
            - American Academy of Dermatology: [aad.org](https://www.aad.org)
            - National Eczema Association: [nationaleczema.org](https://nationaleczema.org)
            - Skin Cancer Foundation: [skincancer.org](https://www.skincancer.org)
            - National Psoriasis Foundation: [psoriasis.org](https://www.psoriasis.org)
            - Vitiligo Support International: [vitiligosupport.org](https://www.vitiligosupport.org)
            """)

def create_glossary_term(term, category):
    """
    Creates an HTML for a glossary term with tooltip/hover effect
    
    Args:
        term: The term to highlight and define
        category: Category in the medical help content dictionary
        
    Returns:
        HTML string with hover tooltip effect
    """
    help_content = get_medical_help_content()
    
    if category in help_content and term.lower() in help_content[category]:
        content = help_content[category][term.lower()]
        # Create a span with title attribute for the tooltip
        return f'<span title="{content}" style="border-bottom: 1px dotted #007bff; cursor: help; color: #007bff;">{term}</span>'
    else:
        # Return the original term without tooltip if not found
        return term

def get_disease_info(disease_name):
    """
    Get information about a specific skin disease
    
    Args:
        disease_name: Name of the skin disease
        
    Returns:
        Dictionary containing disease information
    """
    disease_info = {
        "Hyperpigmentation": {
            "description": "Hyperpigmentation is a condition where patches of skin become darker than the surrounding skin due to excess melanin production. It can affect people of all skin types but is more common in those with darker skin tones.",
            "symptoms": "• Darkened patches of skin\n• Uneven skin tone\n• Brown, black, gray, or reddish spots\n• Most common on face, hands, and other areas exposed to the sun",
            "treatment": "• Topical treatments (hydroquinone, retinoids, vitamin C)\n• Chemical peels\n• Laser therapy\n• Intense pulsed light (IPL)\n• Sunscreen and sun protection\n• Prescription medications for certain types",
            "when_to_see_doctor": "See a doctor if you notice sudden or unexplained changes in your skin color, especially if the changes are accompanied by other symptoms like itching or pain. Also seek medical advice if over-the-counter treatments aren't effective."
        },
        "Acne": {
            "description": "Acne is a common skin condition that occurs when hair follicles become plugged with oil and dead skin cells. It often causes whiteheads, blackheads, or pimples and typically appears on the face, forehead, chest, upper back, and shoulders.",
            "symptoms": "• Whiteheads (closed plugged pores)\n• Blackheads (open plugged pores)\n• Papules (small red, tender bumps)\n• Pustules (papules with pus at the tips)\n• Nodules (large, solid, painful lumps beneath the skin)\n• Cystic lesions (painful, pus-filled lumps beneath the skin)",
            "treatment": "• Topical treatments (benzoyl peroxide, salicylic acid, retinoids)\n• Oral medications (antibiotics, combined oral contraceptives, anti-androgen agents, isotretinoin)\n• Proper cleansing routine\n• Diet modification\n• Stress management",
            "when_to_see_doctor": "Consider seeing a doctor when acne doesn't respond to over-the-counter treatments, is severe, causes emotional distress, or leads to scarring. A dermatologist can offer more effective treatment options."
        },
        "Nail Psoriasis": {
            "description": "Nail psoriasis is a manifestation of psoriasis that affects the fingernails and toenails. It can occur in patients with or without skin psoriasis. The condition can cause significant physical and functional impairments to affected individuals.",
            "symptoms": "• Pitting (small depressions) in the nails\n• Onycholysis (separation of the nail from the nail bed)\n• Changes in nail color (yellow-brown discoloration)\n• Abnormal nail growth\n• Thickening of the nails\n• Crumbling of the nail\n• Blood beneath the nail",
            "treatment": "• Topical treatments (corticosteroids, calcipotriol, tazarotene)\n• Intralesional injections (steroids into nail area)\n• Systemic medications (methotrexate, cyclosporine)\n• Biologics for severe cases\n• Phototherapy\n• Nail care and protection strategies",
            "when_to_see_doctor": "See a doctor if you notice persistent changes in your nails, especially if you have psoriasis elsewhere on your body. Also seek medical advice if nail changes cause pain, affect daily activities, or show signs of infection."
        },
        "Vitiligo": {
            "description": "Vitiligo is a long-term condition where pale white patches develop on the skin due to the lack of melanin, the pigment that gives skin its color. It can affect any area of skin, but commonly occurs on the face, neck, hands, and skin creases.",
            "symptoms": "• White patches on the skin that are smooth in texture\n• Premature whitening of hair on scalp, eyebrows, eyelashes, or beard\n• Loss of color in the tissues that line the inside of mouth and nose\n• Loss of or change in color of the inner layer of the eye (retina)",
            "treatment": "• Topical corticosteroids\n• Calcineurin inhibitors\n• Phototherapy (narrowband UVB, PUVA)\n• Excimer laser\n• Surgical options for stable vitiligo\n• Depigmentation (for widespread vitiligo)\n• Sunscreen and camouflage products",
            "when_to_see_doctor": "See a doctor if you notice the beginning of a loss of skin color or if you experience rapid changes. While vitiligo isn't medically dangerous, it can cause psychological and emotional distress that warrants treatment."
        },
        "SJS-TEN": {
            "description": "Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) represent different severities of the same life-threatening skin condition, usually caused by a reaction to medications. These conditions cause the skin to blister and peel off, affecting mucous membranes as well.",
            "symptoms": "• Flu-like symptoms initially (fever, sore throat, fatigue)\n• Widespread skin pain\n• Blistering of skin and mucous membranes\n• Skin peeling in sheets\n• Red or purple skin rash that spreads\n• Swelling of face and tongue\n• Hives\n• Burning eyes and blurred vision",
            "treatment": "• Immediate discontinuation of causing medication\n• Hospital admission, often to burn unit or ICU\n• Supportive care (fluid replacement, nutrition)\n• Wound care\n• Pain management\n• Eye care\n• Immunomodulating treatments (IVIG, cyclosporine)\n• Prevention of sepsis",
            "when_to_see_doctor": "SEEK EMERGENCY MEDICAL CARE IMMEDIATELY if you experience symptoms of SJS-TEN, particularly if you've recently started a new medication and develop widespread rash, blisters, or peeling skin. This is a medical emergency with high mortality if untreated."
        }
    }
    
    return disease_info.get(disease_name, {
        "description": "Information not available.",
        "symptoms": "Information not available.",
        "treatment": "Information not available.",
        "when_to_see_doctor": "Always consult with a healthcare professional for proper diagnosis and treatment."
    })
