#!/usr/bin/env python
# coding: utf-8

# In[13]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import warnings

warnings.filterwarnings("ignore")

# (All the setup code for NLTK, spaCy, etc. remains the same)
# ...
@st.cache_resource
def download_nltk_resources():
    """Downloads all necessary NLTK models safely."""
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet.zip": "wordnet",
        "corpora/omw-1.4.zip": "omw-1.4",
        "sentiment/vader_lexicon.zip": "vader_lexicon"
    }
    for path, model_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(model_id)

download_nltk_resources()

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model safely."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        st.stop()

nlp = load_spacy_model()
sia = SentimentIntensityAnalyzer()

KNOWN_LOCATIONS = [
    "T Nagar", "Velachery", "Guindy", "Adyar", "Anna Nagar", "Tambaram", "Egmore",
    "Chennai", "Coimbatore", "Madurai", "Bengaluru", "Hyderabad", "Visakhapatnam",
    "Puducherry", "Thiruvananthapuram", "Kochi"
]

LOCATION_HIERARCHY = {
    "Chennai": ["T Nagar", "Velachery", "Guindy", "Adyar", "Anna Nagar", "Tambaram", "Egmore"],
    "Coimbatore": ["Peelamedu", "Gandhipuram", "Saibaba Colony", "Town Hall"],
    "Madurai": ["Thirumalai Nayak Palace", "Alagar Koil", "Anna Nagar"],
    "Bengaluru": ["Whitefield", "Koramangala", "Indiranagar", "MG Road"],
    "Hyderabad": ["Banjara Hills", "Hitech City", "Gachibowli"],
    "Visakhapatnam": ["Dwaraka Nagar", "MVP Colony", "Gajuwaka"]
}

# ------------------------------
# Analysis Functions (Unchanged)
# ------------------------------
def extract_entities(text):
    text_for_ner = re.sub(r"#", "", text.strip())
    doc = nlp(text_for_ner)
    locations = {ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}
    for loc in KNOWN_LOCATIONS:
        if loc.lower() in text_for_ner.lower():
            locations.add(loc)
    hazards = set()
    needs = set()
    ocean_hazard_keywords = {
        "Tsunami": ["tsunami"],
        "Cyclone/Storm": ["cyclone", "storm", "hurricane", "typhoon", "gale"],
        "Flooding": ["flood", "waterlogged", "submerged", "inundated", "coastal flooding"],
        "High Waves": ["high waves", "swell surge", "storm surge", "rough seas"],
        "Heavy Rainfall": ["heavy rain", "downpour", "rainfall"],
    }
    needs_keywords = {
        "Rescue": ["rescue", "help", "evacuate", "stuck", "stranded", "trapped"],
        "Medical": ["medical", "medic", "doctor", "hospital", "ambulance", "injury"],
        "Food": ["food", "ration", "meal", "hungry"],
        "Water": ["water", "drinking water", "thirsty"]
    }
    lower_text = text.lower()
    for hazard, keywords in ocean_hazard_keywords.items():
        if any(kw in lower_text for kw in keywords):
            hazards.add(hazard)
    for need, keywords in needs_keywords.items():
        if any(kw in lower_text for kw in keywords):
            needs.add(need)
    all_hazard_words = {kw for keyword_list in ocean_hazard_keywords.values() for kw in keyword_list}
    final_locations = [loc for loc in locations if loc.lower() not in all_hazard_words]
    return final_locations, list(hazards), list(needs)

def filter_locations_by_hierarchy(locations):
    filtered = set(locations)
    neighborhood_to_city = {
        neigh: city for city, neighborhoods in LOCATION_HIERARCHY.items() for neigh in neighborhoods
    }
    for loc in locations:
        if loc in neighborhood_to_city:
            parent_city = neighborhood_to_city[loc]
            filtered.discard(parent_city)
    return list(filtered)

def analyze_sentiment(text, hazards, needs):
    scores = sia.polarity_scores(text)
    urgent_words = ["urgent", "rescue", "help", "stranded", "emergency", "trapped", "immediate", "critical"]
    if any(word in text.lower() for word in urgent_words) or (hazards and needs):
        return "Negative / Urgent", scores
    if scores['compound'] >= 0.05:
        return "Positive / Safe", scores
    elif scores['compound'] <= -0.05:
        return "Negative / Urgent", scores
    else:
        return "Neutral", scores

def process_text(text):
    clean_text = text.strip()
    if not clean_text:
        return None
    locations, hazards, needs = extract_entities(clean_text)
    if "Tsunami" in hazards:
        locations.insert(0, "Tsunami Warning Area")
    final_locations = filter_locations_by_hierarchy(locations)
    sentiment_label, sentiment_scores = analyze_sentiment(clean_text, hazards, needs)
    return {
        "locations": final_locations,
        "hazards": hazards,
        "needs": needs,
        "sentiment": sentiment_label,
        "sentiment_scores": sentiment_scores
    }

# ------------------------------
# Streamlit UI (Updated Section)
# ------------------------------
st.set_page_config(layout="wide", page_title="Ocean Disaster Analyzer")
st.title("🌊 Ocean Disaster Message Analyzer")
st.markdown("Enter a message about a coastal or ocean disaster to extract key information.")
st.info("Select a language from the dropdown for more accurate translation!")

# --- NEW: LANGUAGE SELECTION DROPDOWN ---
LANGUAGES = {
    "Auto-Detect": "auto",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "English": "en"
}
lang_name = st.selectbox(
    "Select Input Language (leave as 'Auto-Detect' if unsure)",
    options=list(LANGUAGES.keys())
)
# Get the language code (e.g., 'ta' for Tamil) from the selection
selected_lang_code = LANGUAGES[lang_name]

user_input = st.text_area("Enter disaster message here:", height=150)

if st.button("Analyze Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            # --- UPDATED: Use the selected language code for translation ---
            translated_text = GoogleTranslator(source=selected_lang_code, target='en').translate(user_input)
            st.write("---")
            st.markdown(f"**Translated Text (for analysis):** *{translated_text}*")
        except Exception as e:
            st.error(f"Translation failed. Please check your internet connection. Error: {e}")
            translated_text = user_input

        result = process_text(translated_text)
        st.write("---")
        st.subheader("Analysis Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("####  Locations Detected")
            if result["locations"]:
                for loc in result["locations"]:
                    st.success(f"- {loc}")
            else:
                st.error("No locations found.")
        with col2:
            st.markdown("####  Hazards Identified")
            if result["hazards"]:
                for hazard in result["hazards"]:
                    st.warning(f"- {hazard}")
            else:
                st.error("No hazards found.")
        with col3:
            st.markdown("#### Needs Requested")
            if result["needs"]:
                for need in result["needs"]:
                    st.info(f"- {need}")
            else:
                st.error("No specific needs found.")
        with col4:
            st.markdown("####  Sentiment / Urgency")
            st.write(f"**{result['sentiment']}**")
            st.json({"VADER_compound_score": f"{result['sentiment_scores']['compound']:.2f}"})


# In[ ]:





# In[ ]:




