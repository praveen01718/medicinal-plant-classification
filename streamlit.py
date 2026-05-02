import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Botanica — Medicinal Plant Identifier",
    layout="centered",
    page_icon="🌿"
)

# -------------------------------
# Global CSS Injection
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=DM+Mono:wght@300;400&display=swap');

/* ── Root Variables ── */
:root {
    --forest:   #1a3329;
    --moss:     #2d5240;
    --sage:     #5a7d6a;
    --fern:     #8aad97;
    --parchment:#f5f0e4;
    --cream:    #faf7f0;
    --copper:   #b8732a;
    --gold:     #d4a847;
    --ink:      #1e1a14;
    --muted:    #6b6355;
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'Crimson Text', Georgia, serif;
    background-color: var(--cream);
    color: var(--ink);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 780px !important;
}

/* ── Paper texture background ── */
.stApp {
    background-color: var(--cream);
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
}

/* ── Hero Banner ── */
.hero-banner {
    background: var(--forest);
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(90,125,106,0.25) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(180,115,42,0.15) 0%, transparent 50%);
    border-radius: 4px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(26,51,41,0.3), inset 0 1px 0 rgba(212,168,71,0.3);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60' viewBox='0 0 60 60'%3E%3Ccircle cx='30' cy='30' r='1' fill='rgba(212,168,71,0.08)'/%3E%3C/svg%3E");
    pointer-events: none;
}

.hero-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.75rem;
    opacity: 0.9;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--parchment);
    line-height: 1.15;
    margin: 0 0 0.5rem;
}

.hero-title em {
    font-style: italic;
    color: var(--gold);
}

.hero-sub {
    font-family: 'Crimson Text', serif;
    font-size: 1.1rem;
    color: rgba(245,240,228,0.65);
    font-style: italic;
    margin-top: 0.5rem;
}

.hero-divider {
    width: 60px;
    height: 1.5px;
    background: linear-gradient(90deg, var(--gold), transparent);
    margin: 1.25rem 0;
}

/* ── Decorative rule ── */
.ornament {
    text-align: center;
    color: var(--sage);
    font-size: 1.1rem;
    letter-spacing: 0.5em;
    margin: 2rem 0;
    opacity: 0.5;
}

/* ── Upload Section ── */
.upload-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--copper);
    margin-bottom: 0.4rem;
}

.upload-hint {
    font-family: 'Crimson Text', serif;
    font-size: 1rem;
    color: var(--muted);
    font-style: italic;
    margin-bottom: 1rem;
}

/* Streamlit file uploader overrides */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(90,125,106,0.4) !important;
    border-radius: 4px !important;
    background: rgba(245,240,228,0.5) !important;
    transition: border-color 0.3s, background 0.3s;
    padding: 0.5rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--copper) !important;
    background: rgba(245,240,228,0.8) !important;
}

[data-testid="stFileUploader"] label {
    color: var(--moss) !important;
    font-family: 'Crimson Text', serif !important;
    font-size: 1.05rem !important;
}

[data-testid="stFileUploader"] small {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ── Image display ── */
[data-testid="stImage"] {
    border-radius: 3px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(26,51,41,0.15), 0 0 0 1px rgba(90,125,106,0.2);
}

/* ── Result Card ── */
.result-card {
    background: var(--forest);
    background-image: radial-gradient(ellipse at top left, rgba(90,125,106,0.2), transparent 70%);
    border-radius: 4px;
    padding: 2rem 2.25rem;
    margin: 1.75rem 0 0;
    box-shadow: 0 6px 32px rgba(26,51,41,0.25), inset 0 1px 0 rgba(212,168,71,0.2);
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.5s ease both;
}

.result-card::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(212,168,71,0.08), transparent 70%);
    pointer-events: none;
}

.result-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    opacity: 0.8;
    margin-bottom: 0.4rem;
}

.result-plant-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--parchment);
    font-style: italic;
    line-height: 1.1;
    margin-bottom: 1rem;
}

.confidence-bar-wrap {
    margin-bottom: 1.5rem;
}

.confidence-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: rgba(245,240,228,0.5);
    text-transform: uppercase;
    margin-bottom: 0.35rem;
    display: flex;
    justify-content: space-between;
}

.confidence-value {
    color: var(--gold);
    font-weight: 400;
}

.confidence-bar-bg {
    height: 3px;
    background: rgba(245,240,228,0.1);
    border-radius: 2px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--sage), var(--gold));
    border-radius: 2px;
    transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── Uses Section ── */
.uses-section {
    background: var(--parchment);
    border-left: 3px solid var(--copper);
    border-radius: 0 4px 4px 0;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    animation: fadeUp 0.6s 0.15s ease both;
}

.uses-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--copper);
    letter-spacing: 0.05em;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.uses-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.uses-list li {
    font-family: 'Crimson Text', serif;
    font-size: 1.05rem;
    color: var(--ink);
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(90,125,106,0.12);
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    line-height: 1.4;
}

.uses-list li:last-child {
    border-bottom: none;
}

.use-dot {
    color: var(--sage);
    margin-top: 0.1rem;
    flex-shrink: 0;
}

/* ── Streamlit widget overrides ── */
.stButton > button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: var(--forest) !important;
    color: var(--parchment) !important;
    border: 1px solid var(--sage) !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: var(--moss) !important;
    border-color: var(--gold) !important;
    color: var(--gold) !important;
}

/* ── Caption override ── */
[data-testid="caption"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.08em !important;
    text-align: center !important;
    margin-top: 0.4rem !important;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0);    }
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(90,125,106,0.2);
}

.app-footer p {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Hero Header
# -------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-label">Botanical Intelligence · v1.0</div>
    <div class="hero-title">Identify <em>Medicinal</em><br>Plants</div>
    <div class="hero-divider"></div>
    <div class="hero-sub">Upload a photograph — receive instant botanical classification<br>and traditional medicinal insight.</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model (only once)
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("my_model.keras")

model = load_my_model()

# -------------------------------
# Class Names & Plant Data
# -------------------------------
class_names = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha',
    'Avacado', 'Bamboo', 'Basale', 'Betel', 'Betel_Nut', 'Brahmi',
    'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Gauva',
    'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon',
    'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa',
    'Nooni', 'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose',
    'Sapota', 'Tulasi', 'Wood_sorel'
]

plant_usages = {
    "Aloevera": [
        "Treats burns, wounds, and skin irritation",
        "Soothes digestive discomfort and improves gut health",
        "Boosts immune response naturally",
        "Widely used in skincare, haircare, and cosmetics",
    ],
    "Amla": [
        "One of the richest natural sources of Vitamin C",
        "Strengthens immunity and fights oxidative stress",
        "Promotes hair growth and reduces hair fall",
        "Supports liver detoxification and healthy digestion",
    ],
    "Amruta_Balli": [
        "Powerful immunomodulator used extensively in Ayurveda",
        "Treats fever, dengue, and chronic infections",
        "Reduces inflammation and arthritic pain",
        "Purifies blood and improves digestive health",
    ],
    "Arali": [
        "Used externally for skin conditions like eczema and psoriasis",
        "Traditionally used in controlled doses for cardiac support",
        "Applied topically to relieve joint pain",
        "Caution: highly toxic if ingested; external use only",
    ],
    "Ashoka": [
        "Treats uterine disorders and menstrual irregularities",
        "Anti-inflammatory properties help with pain relief",
        "Bark used for leucorrhea and gynaecological conditions",
        "Supports reproductive health in women",
    ],
    "Ashwagandha": [
        "Powerful adaptogen that reduces stress and anxiety",
        "Boosts energy, stamina, and physical endurance",
        "Improves cognitive function and memory",
        "Balances hormones and supports thyroid health",
    ],
    "Avacado": [
        "Rich in healthy monounsaturated fats for heart health",
        "High in potassium and folate for blood pressure regulation",
        "Anti-inflammatory properties benefit joints and skin",
        "Supports weight management and digestive health",
    ],
    "Bamboo": [
        "Young shoots improve digestion and gut health",
        "Rich in silica for bone and joint strength",
        "Anti-inflammatory and wound-healing properties",
        "Used in traditional medicine for respiratory ailments",
    ],
    "Basale": [
        "Rich in iron and folate for healthy blood",
        "Treats constipation and improves bowel regularity",
        "Anti-inflammatory properties soothe skin conditions",
        "Acts as a natural cooling and laxative agent",
    ],
    "Betel": [
        "Antiseptic properties used for oral hygiene",
        "Treats digestive issues and relieves flatulence",
        "Applied to wounds to promote faster healing",
        "Used as a natural breath freshener and stimulant",
    ],
    "Betel_Nut": [
        "Traditionally used to expel intestinal worms",
        "Stimulates digestion and salivary production",
        "Used in Ayurveda for certain neurological conditions",
        "Caution: excessive use linked to oral health risks",
    ],
    "Brahmi": [
        "Enhances memory, concentration, and cognitive function",
        "Reduces anxiety and stress naturally",
        "Neuroprotective properties support brain health",
        "Used in treating epilepsy and anxiety disorders",
    ],
    "Castor": [
        "Castor oil acts as a powerful natural laxative",
        "Anti-inflammatory properties ease joint and muscle pain",
        "Promotes wound healing and skin regeneration",
        "Strengthens hair roots and stimulates hair growth",
    ],
    "Curry_Leaf": [
        "Rich in antioxidants that neutralise free radicals",
        "Helps control blood sugar levels in diabetics",
        "Promotes hair growth and prevents premature greying",
        "Aids digestion and relieves nausea",
    ],
    "Doddapatre": [
        "Treats cough, cold, and respiratory infections",
        "Soothes sore throat and relieves nasal congestion",
        "Anti-inflammatory and antimicrobial properties",
        "Relieves digestive issues and flatulence",
    ],
    "Ekka": [
        "Used in treating skin disorders including leprosy",
        "Anti-inflammatory for arthritis and joint inflammation",
        "Traditionally applied for toothache relief",
        "Treats certain digestive and respiratory conditions",
    ],
    "Ganike": [
        "Anti-inflammatory and antioxidant properties",
        "Treats liver disorders and supports liver function",
        "Used for skin conditions including eczema",
        "Reduces fever and acts as a mild sedative",
    ],
    "Gauva": [
        "Exceptionally rich in Vitamin C and antioxidants",
        "Treats diarrhea and digestive disorders effectively",
        "Leaves used to manage blood sugar in diabetics",
        "Anti-inflammatory properties aid wound healing",
    ],
    "Henna": [
        "Natural hair dye that also conditions and strengthens hair",
        "Antifungal properties treat scalp infections and dandruff",
        "Cooling topical application relieves headaches",
        "Treats skin conditions like eczema and psoriasis",
    ],
    "Hibiscus": [
        "Reduces blood pressure and LDL cholesterol levels",
        "Rich in antioxidants and Vitamin C",
        "Promotes hair growth and prevents premature hair loss",
        "Aids digestion and supports liver health",
    ],
    "Honge": [
        "Seed oil treats various chronic skin diseases",
        "Anti-inflammatory for rheumatic conditions",
        "Antimicrobial properties accelerate wound healing",
        "Used in treating digestive and urinary disorders",
    ],
    "Insulin": [
        "Traditionally used to control blood sugar levels",
        "Rich in corosolic acid that improves insulin sensitivity",
        "Anti-diabetic properties help manage Type 2 diabetes",
        "Antioxidant properties reduce oxidative cellular stress",
    ],
    "Jasmine": [
        "Aromatherapy use reduces anxiety and depression",
        "Antiseptic properties promote wound healing",
        "Relieves menstrual cramps and associated pain",
        "Used in skincare for moisturising and anti-ageing benefits",
    ],
    "Lemon": [
        "Rich in Vitamin C significantly boosting immunity",
        "Aids digestion, reduces bloating, and detoxifies",
        "Antibacterial properties support oral health",
        "Natural diuretic that promotes kidney health",
    ],
    "Lemon_grass": [
        "Treats digestive issues, bloating, and stomach cramps",
        "Anti-inflammatory and analgesic for pain relief",
        "Reduces fever and acts as a natural diuretic",
        "Antimicrobial properties help combat infections",
    ],
    "Mango": [
        "Rich in Vitamins A, C, and E for strong immunity",
        "Improves digestion with natural amylase enzymes",
        "Leaves used to manage blood sugar levels",
        "Antioxidants protect cells against oxidative damage",
    ],
    "Mint": [
        "Relieves digestive issues, nausea, and IBS symptoms",
        "Natural decongestant for sinus and respiratory relief",
        "Antimicrobial properties promote oral hygiene",
        "Cooling menthol effect relieves tension headaches",
    ],
    "Nagadali": [
        "Treats digestive disorders and liver conditions",
        "Used in Ayurveda for chronic skin diseases",
        "Anti-inflammatory for arthritis and joint pain",
        "Stimulates appetite and improves metabolism",
    ],
    "Neem": [
        "Powerful antibacterial and antifungal properties",
        "Treats acne, eczema, and various skin infections",
        "Supports oral health and prevents gum disease",
        "Purifies blood, boosts immunity, and reduces inflammation",
    ],
    "Nithyapushpa": [
        "Contains alkaloids used in cancer treatment (vinblastine, vincristine)",
        "Traditionally used to manage blood sugar in diabetics",
        "Lowers high blood pressure naturally",
        "Antiseptic properties aid wound healing",
    ],
    "Nooni": [
        "Rich in antioxidants promoting overall cellular health",
        "Treats joint pain and reduces arthritic inflammation",
        "Antimicrobial and antiviral immune-boosting properties",
        "Increases energy levels and reduces fatigue",
    ],
    "Pappaya": [
        "Rich in papain enzyme that breaks down proteins and aids digestion",
        "Treats dengue fever by boosting platelet count",
        "Anti-inflammatory properties accelerate wound healing",
        "Rich in antioxidants preventing oxidative cell damage",
    ],
    "Pepper": [
        "Enhances bioavailability of nutrients and other medicines",
        "Anti-inflammatory and antioxidant properties",
        "Aids digestion and relieves constipation",
        "Antimicrobial properties treat respiratory infections",
    ],
    "Pomegranate": [
        "Exceptionally rich in antioxidants protecting cardiovascular health",
        "Anti-inflammatory reducing risk of chronic disease",
        "Treats diarrhea and various digestive disorders",
        "Strengthens immune function and combats infections",
    ],
    "Raktachandini": [
        "Anti-inflammatory properties help treat fever and infections",
        "Treats digestive disorders including dysentery",
        "Antimicrobial properties benefit various skin conditions",
        "Natural antioxidant used as a traditional food colorant",
    ],
    "Rose": [
        "Anti-inflammatory reduces skin redness and irritation",
        "Antibacterial properties promote wound healing",
        "Rich in Vitamin C that boosts overall immunity",
        "Supports digestive health and reduces bloating",
    ],
    "Sapota": [
        "Rich in tannins with powerful anti-inflammatory properties",
        "Treats diarrhea and various digestive disorders",
        "High in antioxidants that protect against oxidative stress",
        "Good source of sustained energy and essential minerals",
    ],
    "Tulasi": [
        "Powerful adaptogen that reduces stress and anxiety",
        "Treats cold, cough, and respiratory infections",
        "Anti-inflammatory and antimicrobial properties",
        "Supports blood sugar regulation and cardiac health",
    ],
    "Wood_sorel": [
        "Rich in Vitamin C historically used to treat scurvy",
        "Anti-inflammatory for skin conditions and rashes",
        "Treats digestive issues and stimulates appetite",
        "Antimicrobial properties promote wound healing",
    ],
}

plant_icons = {
    "Aloevera":      "🌵",
    "Amla":          "🍃",
    "Amruta_Balli":  "🌿",
    "Arali":         "🌸",
    "Ashoka":        "🌺",
    "Ashwagandha":   "🌾",
    "Avacado":       "🥑",
    "Bamboo":        "🎋",
    "Basale":        "🥬",
    "Betel":         "🍃",
    "Betel_Nut":     "🌰",
    "Brahmi":        "🧠",
    "Castor":        "🌱",
    "Curry_Leaf":    "🍃",
    "Doddapatre":    "🌿",
    "Ekka":          "🌼",
    "Ganike":        "🍇",
    "Gauva":         "🍈",
    "Henna":         "🌿",
    "Hibiscus":      "🌺",
    "Honge":         "🌳",
    "Insulin":       "🌿",
    "Jasmine":       "🌸",
    "Lemon":         "🍋",
    "Lemon_grass":   "🌾",
    "Mango":         "🥭",
    "Mint":          "🌿",
    "Nagadali":      "🌿",
    "Neem":          "🍃",
    "Nithyapushpa":  "🌸",
    "Nooni":         "🍈",
    "Pappaya":       "🍑",
    "Pepper":        "🌶️",
    "Pomegranate":   "🍎",
    "Raktachandini": "🌹",
    "Rose":          "🌹",
    "Sapota":        "🍂",
    "Tulasi":        "🌿",
    "Wood_sorel":    "🍀",
}

# -------------------------------
# Upload Section
# -------------------------------
st.markdown('<div class="upload-label">Specimen Upload</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-hint">Accepted formats: JPG, JPEG, PNG</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# -------------------------------
# Inference & Results
# -------------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)
    st.image(img, caption="Uploaded specimen", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Prediction
    with st.spinner("Analysing specimen…"):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100
    icon = plant_icons.get(predicted_class, "🌿")

    # ── Result Card ──
    bar_width = f"{confidence:.1f}%"
    st.markdown(f"""
    <div class="result-card">
        <div class="result-tag">Classification Result</div>
        <div class="result-plant-name">{icon} {predicted_class}</div>
        <div class="confidence-bar-wrap">
            <div class="confidence-label">
                <span>Model Confidence</span>
                <span class="confidence-value">{confidence:.2f}%</span>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{bar_width}"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Uses Section ──
    uses = plant_usages[predicted_class]
    uses_html = "".join(
        f'<li><span class="use-dot">◆</span>{use}</li>'
        for use in uses
    )
    st.markdown(f"""
    <div class="uses-section">
        <div class="uses-title">
            <span>📖</span> Medicinal Properties &amp; Uses
        </div>
        <ul class="uses-list">
            {uses_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class="app-footer">
    <p>Botanica · Medicinal Plant Classifier · For educational use only</p>
</div>
""", unsafe_allow_html=True)
