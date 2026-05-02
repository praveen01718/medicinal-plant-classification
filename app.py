import io
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_model.keras")
model = tf.keras.models.load_model(model_path)

class_names = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha',
    'Avacado', 'Bamboo', 'Basale', 'Betel', 'Betel_Nut', 'Brahmi',
    'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Gauva',
    'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon',
    'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa',
    'Nooni', 'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose',
    'Sapota', 'Wood_sorel'
]

plant_icons = {
    "Aloevera": "🌵", "Amla": "🍃", "Amruta_Balli": "🌿", "Arali": "🌸",
    "Ashoka": "🌺", "Ashwagandha": "🌾", "Avacado": "🥑", "Bamboo": "🎋",
    "Basale": "🥬", "Betel": "🍃", "Betel_Nut": "🌰", "Brahmi": "🧠",
    "Castor": "🌱", "Curry_Leaf": "🍃", "Doddapatre": "🌿", "Ekka": "🌼",
    "Ganike": "🍇", "Gauva": "🍈", "Henna": "🌿", "Hibiscus": "🌺",
    "Honge": "🌳", "Insulin": "🌿", "Jasmine": "🌸", "Lemon": "🍋",
    "Lemon_grass": "🌾", "Mango": "🥭", "Mint": "🌿", "Nagadali": "🌿",
    "Neem": "🍃", "Nithyapushpa": "🌸", "Nooni": "🍈", "Pappaya": "🍑",
    "Pepper": "🌶️", "Pomegranate": "🍎", "Raktachandini": "🌹", "Rose": "🌹",
    "Sapota": "🍂", "Wood_sorel": "🍀",
}

# ── Comprehensive plant data ────────────────────────────────────────────────
plant_details = {
    "Aloevera": {
        "subtitle": "The Miracle Succulent",
        "scientific_name": "Aloe barbadensis miller",
        "family": "Asphodelaceae",
        "origin": "Arabian Peninsula, naturalized globally",
        "description": "A stemless perennial succulent with thick fleshy leaves arranged in a rosette. Its clear inner gel and yellow latex have been used therapeutically for over 6,000 years across Egyptian, Greek, and Ayurvedic traditions.",
        "active_compounds": ["Aloin", "Aloe-emodin", "Acemannan", "Barbaloin", "Vitamins A, C, E"],
        "leaf_uses": [
            "Gel directly soothes burns, sunburns, cuts, and minor wounds on contact",
            "Anti-inflammatory gel relieves eczema, psoriasis, and dermatitis flare-ups",
            "Leaf juice consumed internally eases acid reflux, IBS, and gut inflammation",
            "Antifungal properties of gel treat nail infections and athlete's foot",
            "Applied to scalp to reduce dandruff and promote hair follicle health",
            "Gel used as a natural after-shave balm to prevent razor burns and redness",
        ],
        "stem_uses": [
            "Central leaf-stem tissue yields concentrated gel for pharmaceutical extraction",
            "Stem base latex contains aloin, used as a potent natural laxative",
            "Dried stem extracts used in traditional wound-closure poultices",
        ],
        "root_uses": [
            "Roots occasionally used in traditional African formulations for fever",
            "Root decoctions used in folk medicine for urinary tract support",
        ],
        "general_medicinal_uses": [
            "Treats burns, wounds, and skin irritation topically",
            "Soothes digestive discomfort and improves gut health when ingested",
            "Boosts immune response and reduces oxidative stress",
            "Widely prescribed in integrative medicine for inflammatory conditions",
        ],
        "pharmaceutical_uses": [
            "Acemannan extracted from gel is studied as an immunostimulant in oncology",
            "Aloin is the active ingredient in many OTC laxative tablets and syrups",
            "Pharmaceutical-grade aloe gel used in wound-dressing hydrogels",
            "Aloe extract used in ophthalmologic preparations for dry-eye conditions",
        ],
        "herbal_uses": [
            "Fresh leaf gel is central to Ayurvedic formulation Kumari Asava (digestive tonic)",
            "Gel mixed with turmeric forms a traditional anti-inflammatory paste",
            "Leaf juice blended with amla and honey used as a classical Rasayana tonic",
            "Widely used in Unani medicine as a hepatoprotective bitter tonic",
        ],
        "cosmetic_uses": [
            "Primary moisturising agent in face creams, serums, and body lotions",
            "Used in shampoos and conditioners to soothe scalp and add shine",
            "Core ingredient in after-sun gels and cooling sprays",
            "Incorporated into anti-ageing formulations for its collagen-supporting activity",
        ],
    },
    "Amla": {
        "subtitle": "Nature's Vitamin C Powerhouse",
        "scientific_name": "Phyllanthus emblica",
        "family": "Phyllanthaceae",
        "origin": "Indian subcontinent and Southeast Asia",
        "description": "A deciduous tree bearing small light-green fruits among the richest natural sources of Vitamin C. Central to Ayurvedic formulations including Chyawanprash and Triphala, it is considered a Rasayana rejuvenating tonic.",
        "active_compounds": ["Ascorbic acid", "Ellagic acid", "Gallic acid", "Emblicanin A & B", "Quercetin"],
        "leaf_uses": [
            "Leaf decoction used as a gargle for mouth ulcers and sore throat",
            "Leaf paste applied topically to treat skin inflammations and rashes",
            "Leaves boiled with water used as a herbal rinse to cool the body in fever",
            "Leaf extract has antimicrobial properties effective against oral bacteria",
        ],
        "stem_uses": [
            "Stem bark decoction used to treat diarrhoea and dysentery",
            "Bark extract used in traditional formulations for diabetes management",
            "Stem bark paste applied externally to relieve joint pain and swelling",
        ],
        "root_uses": [
            "Root bark used in Ayurveda for jaundice and liver disorders",
            "Root decoction used traditionally to manage ulcer-related conditions",
        ],
        "general_medicinal_uses": [
            "Strengthens immunity and fights oxidative stress systemically",
            "Promotes hair growth, reduces hair fall, and prevents premature greying",
            "Supports liver detoxification and healthy digestive function",
            "Potent anti-ageing Rasayana herb that rejuvenates body tissues",
        ],
        "pharmaceutical_uses": [
            "Standardised amla extract used in antioxidant supplement formulations",
            "Emblicanin A & B investigated as chemopreventive agents",
            "Used in Ayurvedic proprietary medicines for hepatoprotection",
            "Gallic acid extracted for use in antidiabetic pharmaceutical research",
        ],
        "herbal_uses": [
            "Primary ingredient in Chyawanprash, the classical Ayurvedic rejuvenating jam",
            "One of three fruits in Triphala, the most widely used Ayurvedic formula",
            "Dried amla powder used in Brahmi Rasayana for cognitive enhancement",
            "Amla murabba (preserve) used as a daily tonic in traditional Unani practice",
        ],
        "cosmetic_uses": [
            "Amla oil is a staple hair-care product preventing greying and hair fall",
            "Used in face packs and brightening serums for Vitamin C-driven skin radiance",
            "Ingredient in herbal hair dyes as a natural conditioning agent",
            "Added to lip balms and skin creams for its antioxidant protection",
        ],
    },
    "Amruta_Balli": {
        "subtitle": "The Divine Nectar Vine",
        "scientific_name": "Tinospora cordifolia",
        "family": "Menispermaceae",
        "origin": "Tropical India, Myanmar, and Sri Lanka",
        "description": "A large, glabrous, deciduous climbing shrub known as Guduchi in Sanskrit. Classified as a Rasayana herb in Ayurveda, it is revered for its broad-spectrum immunomodulatory and adaptogenic activity.",
        "active_compounds": ["Tinosporin", "Berberine", "Palmatine", "Tinosporide", "Cordifolide"],
        "leaf_uses": [
            "Leaf juice used for managing diabetes by improving insulin sensitivity",
            "Fresh leaves chewed or juiced to reduce fever and chronic inflammation",
            "Leaf decoction used for urinary tract infections and burning sensation",
            "Leaves used in formulations to purify blood and reduce skin diseases",
            "Leaf extract enhances white blood cell activity during dengue recovery",
        ],
        "stem_uses": [
            "Stem is the primary medicinal part — processed into Guduchi Satva (starch extract)",
            "Stem decoction used as a classical immunomodulator in Ayurvedic practice",
            "Stem bark used in anti-arthritic formulations to reduce joint inflammation",
            "Dried stem powder used in proprietary hepatoprotective Ayurvedic tablets",
        ],
        "root_uses": [
            "Roots used in traditional formulations for gout and metabolic disorders",
            "Root extracts show antispasmodic activity used for abdominal cramps",
        ],
        "general_medicinal_uses": [
            "Powerful immunomodulator used extensively in Ayurveda for immune support",
            "Treats fever, dengue, and chronic viral infections effectively",
            "Reduces inflammation and arthritic pain when taken regularly",
            "Purifies blood and improves overall digestive health",
        ],
        "pharmaceutical_uses": [
            "Berberine isolated from stem used in antidiabetic drug research",
            "Tinospora extract is the active in several patented anti-fever preparations",
            "Immunostimulant extracts studied as adjuvant therapy in cancer treatment",
            "Tinosporide studied as a hepatoprotective agent in clinical trials",
        ],
        "herbal_uses": [
            "Guduchi Satva (stem starch) is a primary ingredient in Ayurvedic fever formulas",
            "Included in classical Ayurvedic preparation Ashtavarga for rejuvenation",
            "Stem juice combined with neem used for chronic skin diseases in traditional practice",
            "Widely used in Unani medicine as an antipyretic and immune tonic",
        ],
        "cosmetic_uses": [
            "Stem extract used in herbal skin care for brightening and anti-inflammatory effect",
            "Incorporated into scalp tonics to reduce dandruff and promote hair growth",
            "Used in natural sunscreen formulations for its photoprotective activity",
        ],
    },
    "Arali": {
        "subtitle": "The Ornamental Oleander",
        "scientific_name": "Nerium oleander",
        "family": "Apocynaceae",
        "origin": "Mediterranean region to Southwest Asia",
        "description": "An evergreen shrub with beautiful trumpet-shaped flowers. All parts are highly toxic due to cardiac glycosides; traditional use is strictly external and under expert supervision.",
        "active_compounds": ["Oleandrin", "Neriine", "Digitoxigenin", "Folinerin", "Odoroside"],
        "leaf_uses": [
            "Crushed leaf paste applied externally to treat skin infections and ringworm",
            "Leaf extract used in controlled traditional preparations for arthritic joints",
            "Dried leaf powder used historically in minimal doses for cardiac conditions",
            "⚠️ CAUTION: Leaves are highly toxic — internal use strictly prohibited",
        ],
        "stem_uses": [
            "Stem latex used in controlled Ayurvedic preparations for skin disorders",
            "Bark decoction applied externally only for chronic skin diseases",
            "⚠️ CAUTION: Stem and bark are equally toxic as leaves",
        ],
        "root_uses": [
            "Root used in very controlled traditional formulations for leprosy (external)",
            "⚠️ CAUTION: Root extracts are toxic; avoid all internal use",
        ],
        "general_medicinal_uses": [
            "External application treats skin conditions including eczema and psoriasis",
            "Traditional controlled-dose use for cardiac support (not recommended without supervision)",
            "Applied topically to relieve joint pain and muscle inflammation",
            "All uses must be under expert Ayurvedic or medical supervision",
        ],
        "pharmaceutical_uses": [
            "Oleandrin studied in oncology research as a potential antitumour agent",
            "Cardiac glycosides researched for controlled cardiac rhythm applications",
            "Neriine investigated for neuroprotective properties in research settings",
        ],
        "herbal_uses": [
            "Very limited and strictly supervised use in traditional Siddha medicine",
            "Used in Ayurvedic formulations for skin diseases under toxic-herb protocols",
        ],
        "cosmetic_uses": [
            "Flower extracts occasionally used in perfumery for their scent profile",
            "Not recommended for cosmetic use due to toxicity concerns",
        ],
    },
    "Ashoka": {
        "subtitle": "The Sorrow-Less Sacred Tree",
        "scientific_name": "Saraca asoca",
        "family": "Fabaceae",
        "origin": "Indian subcontinent and Sri Lanka",
        "description": "A medium-sized evergreen tree sacred to Hindus and Buddhists. The bark and flowers are extensively used in Ayurvedic gynaecology for uterine and menstrual disorders.",
        "active_compounds": ["Catechins", "Haematoxylins", "Leucocyanidins", "Saracin", "Procyanidins"],
        "leaf_uses": [
            "Leaf decoction used to reduce excessive menstrual bleeding",
            "Fresh leaf juice applied to skin eruptions and boils as an antiseptic",
            "Leaf paste used externally for oedema and inflammatory swellings",
            "Leaf extract shows mild antidepressant activity in folk medicine traditions",
        ],
        "stem_uses": [
            "Stem bark is the primary medicinal part — used in Ashokarishta formulation",
            "Bark decoction is the classical Ayurvedic treatment for dysmenorrhoea",
            "Bark extract used for leucorrhoea and other gynaecological conditions",
            "Powdered bark used internally for internal haemorrhage and rectal disorders",
        ],
        "root_uses": [
            "Root bark used in traditional preparations for uterine fibroids",
            "Root decoction used for diabetes management in folk medicine",
        ],
        "general_medicinal_uses": [
            "Treats uterine disorders and menstrual irregularities effectively",
            "Anti-inflammatory properties help with dysmenorrhoea and pelvic pain",
            "Bark used for leucorrhoea and chronic gynaecological conditions",
            "Supports reproductive health and hormonal balance in women",
        ],
        "pharmaceutical_uses": [
            "Leucocyanidins from bark studied as uterine tonic in clinical research",
            "Saracin investigated for anti-oestrogenic properties in hormonal research",
            "Procyanidins extracted for haemostatic pharmaceutical applications",
        ],
        "herbal_uses": [
            "Primary ingredient in Ashokarishta, the most important Ayurvedic gynaecological tonic",
            "Bark used in Pradarantaka Lauha for heavy menstrual bleeding",
            "Part of classical Sapthasara Kashaya formulation for uterine health",
            "Used in Siddha medicine as Ashoka Choornam for reproductive disorders",
        ],
        "cosmetic_uses": [
            "Flower extracts used in natural perfumery and incense preparations",
            "Bark extract used in herbal face packs for skin brightening",
            "Included in aromatherapy preparations for stress and anxiety relief",
        ],
    },
    "Ashwagandha": {
        "subtitle": "Indian Winter Cherry",
        "scientific_name": "Withania somnifera",
        "family": "Solanaceae",
        "origin": "India, North Africa, and the Mediterranean",
        "description": "A short perennial shrub considered one of Ayurveda's most important Rasayana herbs. Adaptogenic withanolides in its roots are the subject of extensive modern clinical research for stress, cognition, and endurance.",
        "active_compounds": ["Withanolide A", "Withaferin A", "Withanone", "Sitoindosides VII–X", "Anaferine"],
        "leaf_uses": [
            "Fresh leaf poultice applied to wounds, boils, and carbuncles for healing",
            "Leaf juice used as a topical anti-inflammatory for joint pain",
            "Leaf tea used traditionally to induce calm sleep and reduce anxiety",
            "Dried leaf powder used in classical formulations for fever and malaise",
        ],
        "stem_uses": [
            "Stem bark decoction used in Ayurveda for rheumatic joint conditions",
            "Stem used in traditional African medicine for mood and anxiety disorders",
        ],
        "root_uses": [
            "Root is the primary medicinal part — rich in withanolides and adaptogenic compounds",
            "Root powder in warm milk is the classical daily Rasayana for strength and vitality",
            "Root extract clinically proven to reduce cortisol levels in chronic stress",
            "Root used for hypothyroidism, male infertility, and athletic performance",
        ],
        "general_medicinal_uses": [
            "Powerful adaptogen that significantly reduces stress and anxiety",
            "Boosts energy, stamina, and physical endurance in clinical studies",
            "Improves cognitive function, memory, and reaction time",
            "Balances thyroid hormones and supports reproductive health",
        ],
        "pharmaceutical_uses": [
            "Standardised KSM-66 root extract used in clinically studied anxiety supplements",
            "Withaferin A investigated as a potent anti-cancer agent in pre-clinical studies",
            "Root extract used in patented anti-inflammatory pharmaceutical preparations",
            "Withanolides studied for neuroprotection in Alzheimer's and Parkinson's research",
        ],
        "herbal_uses": [
            "Root powder with ghee and honey is a classical Ayurvedic Vajikaran (aphrodisiac) formula",
            "Included in Chyawanprash as an adaptogenic and immunomodulatory ingredient",
            "Ashwagandha Lehya is a traditional tonic for debility and convalescence",
            "Used in Unani medicine as Asgand for fatigue, anxiety, and sexual debility",
        ],
        "cosmetic_uses": [
            "Root extract used in anti-ageing creams for its collagen-boosting activity",
            "Incorporated into hair oils to reduce stress-related hair fall",
            "Used in natural deodorants for its antimicrobial properties",
            "Ashwagandha extract added to under-eye creams for dark circle reduction",
        ],
    },
    "Avacado": {
        "subtitle": "The Butter Fruit of the Americas",
        "scientific_name": "Persea americana",
        "family": "Lauraceae",
        "origin": "South-central Mexico and Central America",
        "description": "A large evergreen tree cultivated for its nutrient-dense, creamy fruit. Uniquely high in heart-healthy oleic acid and fat-soluble vitamins; avocado leaves and seeds also carry significant ethnomedicinal applications.",
        "active_compounds": ["Oleic acid", "Persin", "Glutathione", "Beta-sitosterol", "Vitamins K, E, B6"],
        "leaf_uses": [
            "Leaf tea used in Mexican traditional medicine for hypertension and headaches",
            "Boiled leaf decoction used as a diuretic for kidney and bladder disorders",
            "Leaf poultice applied externally for skin rashes and inflammatory conditions",
            "Leaf steam inhalation used for respiratory congestion and sinus relief",
        ],
        "stem_uses": [
            "Stem bark decoction used in traditional medicine for diarrhoea and dysentery",
            "Bark extract applied externally to wounds for antiseptic protection",
        ],
        "root_uses": [
            "Roots used in Central American folk medicine for menstrual irregularities",
        ],
        "general_medicinal_uses": [
            "Rich in healthy monounsaturated fats that protect cardiovascular health",
            "High potassium and folate content regulates blood pressure",
            "Anti-inflammatory oleic acid benefits joints, skin, and gut lining",
            "Supports weight management and sustained digestive health",
        ],
        "pharmaceutical_uses": [
            "Avocado-soybean unsaponifiables (ASU) used in approved osteoarthritis drugs in Europe",
            "Beta-sitosterol extracted for use in cholesterol-lowering pharmaceutical supplements",
            "Persin under investigation as an anticancer compound in breast cancer research",
            "Avocado oil used as an excipient in pharmaceutical topical formulations",
        ],
        "herbal_uses": [
            "Leaf tea is a popular herbal remedy for hypertension in Caribbean folk medicine",
            "Seed powder used in traditional Mexican herbal medicine for dandruff and skin conditions",
            "Leaf decoction used in West African herbal practice for malaria fever",
        ],
        "cosmetic_uses": [
            "Avocado oil is a premium emollient used in luxury moisturisers and hair masks",
            "Used in natural lip balms and body butters for deep hydration",
            "Cold-pressed avocado oil used in facial oils for dry and ageing skin",
            "Avocado butter used in natural hair conditioners for intense moisture",
        ],
    },
    "Bamboo": {
        "subtitle": "The Green Gold of the East",
        "scientific_name": "Bambusa vulgaris",
        "family": "Poaceae",
        "origin": "Tropical and subtropical Asia",
        "description": "One of the fastest-growing plants on Earth, integral to Asian culture and medicine for millennia. Silica-rich internodes are used in Ayurvedic formulations called Vanshalochana.",
        "active_compounds": ["Bamboo silica", "Chlorophyll", "Lignin", "Flavonoids", "Cyanogenic glycosides"],
        "leaf_uses": [
            "Leaf tea used as a mild diuretic and for urinary tract health",
            "Young leaf decoction used in traditional Chinese medicine for fever and anxiety",
            "Leaf juice applied to skin as a cooling and soothing agent",
            "Leaf extract used in traditional formulations for respiratory inflammation",
        ],
        "stem_uses": [
            "Stem silica (Vanshalochana/Tabasheer) is the primary Ayurvedic medicinal part",
            "Stem decoction used for respiratory ailments including bronchitis and asthma",
            "Young bamboo shoots used as a nutritious medicinal food for gut health",
            "Stem internodal silica used as a calcium-silica supplement for bones and nails",
        ],
        "root_uses": [
            "Root decoction used in traditional Chinese medicine as a diuretic",
            "Roots used topically for skin conditions in Southeast Asian folk medicine",
        ],
        "general_medicinal_uses": [
            "Young shoots improve digestion and gut microbiome health",
            "Rich in silica for strengthening bones, joints, and connective tissue",
            "Anti-inflammatory and wound-healing properties throughout the plant",
            "Used in traditional medicine for respiratory ailments and fever",
        ],
        "pharmaceutical_uses": [
            "Bamboo silica used in pharmaceutical bone and joint health supplements",
            "Bamboo charcoal used in detoxification and adsorption medical products",
            "Flavonoids from bamboo leaves investigated for cardiovascular protective effects",
        ],
        "herbal_uses": [
            "Vanshalochana (bamboo silica) is a classical Ayurvedic ingredient in respiratory formulas",
            "Young shoots used in Chinese traditional medicine for cooling digestive heat",
            "Leaf decoction used in Southeast Asian folk medicine for fever and infection",
        ],
        "cosmetic_uses": [
            "Bamboo extract used in anti-ageing creams for its silica content",
            "Bamboo powder used as a natural exfoliant in scrubs and cleansers",
            "Bamboo stem extract used in nail-strengthening formulations",
            "Bamboo charcoal used in pore-cleansing face masks",
        ],
    },
    "Basale": {
        "subtitle": "Malabar Spinach",
        "scientific_name": "Basella alba",
        "family": "Basellaceae",
        "origin": "Tropical Asia and Africa",
        "description": "A fast-growing heat-tolerant leafy vine used as a vegetable across tropical regions. The mucilaginous leaves are a valuable source of iron, calcium, and vitamins A and C.",
        "active_compounds": ["Iron", "Vitamin C", "Calcium", "Saponins", "Beta-carotene"],
        "leaf_uses": [
            "Leaf mucilage used as a gentle laxative for constipation relief",
            "Fresh leaves eaten regularly to treat iron-deficiency anaemia",
            "Leaf paste applied to skin burns and boils as a cooling emollient",
            "Leaves used as a diuretic food for urinary health",
            "Leaf decoction used traditionally for diarrhoea and dysentery",
        ],
        "stem_uses": [
            "Stem juice used topically for skin diseases and minor infections",
            "Stem decoction used as a mild anti-inflammatory for joints",
        ],
        "root_uses": [
            "Root used in traditional African medicine for gonorrhoea treatment",
            "Root paste applied externally for swellings and inflammatory conditions",
        ],
        "general_medicinal_uses": [
            "Rich in iron and folate for treating and preventing anaemia",
            "Treats constipation and improves bowel regularity naturally",
            "Anti-inflammatory properties soothe skin conditions and rashes",
            "Acts as a natural cooling, demulcent, and mild laxative agent",
        ],
        "pharmaceutical_uses": [
            "Iron and folate content used in nutritional supplement formulations",
            "Saponins studied for their cholesterol-lowering potential",
            "Beta-carotene extracted for use in vitamin A supplement production",
        ],
        "herbal_uses": [
            "Leaves used in traditional Indian medicine (Pasalai Keerai) for anaemia",
            "Included in Siddha formulations for skin diseases and heat conditions",
            "Leaf decoction used in African traditional medicine for diarrhoea management",
        ],
        "cosmetic_uses": [
            "Leaf mucilage used in natural hair conditioning masks",
            "Beta-carotene from leaves used in skin-brightening natural cosmetics",
            "Cooling leaf gel used in soothing sunburn relief preparations",
        ],
    },
    "Betel": {
        "subtitle": "The Sacred Paan Leaf",
        "scientific_name": "Piper betle",
        "family": "Piperaceae",
        "origin": "Southeast Asia, cultivated throughout South Asia",
        "description": "A perennial dioecious vine with glossy heart-shaped leaves central to cultural and religious rituals. Volatile oils in the leaves provide documented antimicrobial and anti-inflammatory properties.",
        "active_compounds": ["Chavicol", "Eugenol", "Beta-carotene", "Hydroxychavicol", "Caryophyllene"],
        "leaf_uses": [
            "Fresh leaf chewing promotes oral hygiene and destroys pathogenic oral bacteria",
            "Leaf juice applied to wounds accelerates healing and prevents infection",
            "Warm leaf applied to chest relieves congestion and respiratory discomfort",
            "Leaf decoction used as a gargle for sore throat and tonsillitis",
            "Leaf applied to skin eruptions as a natural antimicrobial compress",
            "Leaf oil used for treating headaches via topical application to temples",
        ],
        "stem_uses": [
            "Stem decoction used in traditional medicine for digestive discomfort",
            "Stem bark used in folk medicine for urinary disorders",
        ],
        "root_uses": [
            "Roots used in Ayurvedic preparations for cough and hoarseness",
            "Root decoction used traditionally for fever and body aches",
        ],
        "general_medicinal_uses": [
            "Antiseptic properties widely used for oral hygiene and dental health",
            "Treats digestive issues and relieves flatulence and bloating",
            "Applied to wounds to promote faster healing and prevent infection",
            "Used as a natural breath freshener and mild digestive stimulant",
        ],
        "pharmaceutical_uses": [
            "Eugenol extracted from betel is used as a dental analgesic and antiseptic",
            "Hydroxychavicol studied for its potent anticancer and antimicrobial activity",
            "Betel leaf extract investigated for anti-ulcer activity in gastric research",
        ],
        "herbal_uses": [
            "Central to Tamboolam (ritual paan) practice in Ayurveda for digestive support",
            "Betel leaf juice used in traditional Siddha medicine for cough and cold",
            "Used in Thai traditional medicine as Cha Phlu for digestive and skin conditions",
        ],
        "cosmetic_uses": [
            "Betel leaf oil used in dental care products for gum health",
            "Leaf extract used in natural mouthwash formulations for antibacterial action",
            "Included in herbal soaps and body washes for antimicrobial properties",
        ],
    },
    "Betel_Nut": {
        "subtitle": "The Areca Palm Seed",
        "scientific_name": "Areca catechu",
        "family": "Arecaceae",
        "origin": "Southeast Asia and the Pacific Islands",
        "description": "The seed of the areca palm, widely chewed across Asia. Contains psychoactive alkaloids that stimulate the central nervous system; classified as a Group 1 carcinogen by IARC at high chronic doses.",
        "active_compounds": ["Arecoline", "Arecaidine", "Guvacoline", "Tannins", "Flavonoids"],
        "leaf_uses": [
            "Young palm leaves used in traditional medicine for headache and fever",
            "Leaf extracts used externally for skin conditions in Southeast Asian folk medicine",
        ],
        "stem_uses": [
            "Areca palm heartwood used as a dietary supplement for digestive health in folk use",
            "Stem tissue used in traditional Cambodian medicine for urinary conditions",
        ],
        "root_uses": [
            "Root decoction used traditionally for toothache and oral conditions",
        ],
        "general_medicinal_uses": [
            "Traditionally used to expel intestinal worms as an anthelmintic",
            "Stimulates digestion and increases salivary production",
            "Used in Ayurveda in controlled doses for neurological conditions",
            "⚠️ Excessive use is strongly linked to oral cancer and cardiovascular disease",
        ],
        "pharmaceutical_uses": [
            "Arecoline studied as a cholinergic drug candidate for Alzheimer's disease",
            "Tannins extracted for use in pharmaceutical astringent formulations",
            "Arecaidine investigated for anthelmintic pharmaceutical applications",
        ],
        "herbal_uses": [
            "Used in Ayurvedic preparations Khadiradi Vati for oral health",
            "Part of traditional Chinese medicine formulations for digestive parasites",
            "Seed powder used in traditional anthelmintic preparations across Southeast Asia",
        ],
        "cosmetic_uses": [
            "Areca nut extract used in natural teeth-whitening preparations",
            "Tannins from seeds used in astringent facial toners in limited herbal formulations",
        ],
    },
    "Brahmi": {
        "subtitle": "The Memory Herb of Ayurveda",
        "scientific_name": "Bacopa monnieri",
        "family": "Plantaginaceae",
        "origin": "Wetlands of India, Nepal, Sri Lanka, and Australia",
        "description": "A creeping succulent herb thriving in shallow water. Bacosides have been shown in clinical trials to improve memory consolidation and cognitive processing speed in both healthy adults and the elderly.",
        "active_compounds": ["Bacoside A", "Bacoside B", "Bacopasaponins", "Brahmine", "Herpestine"],
        "leaf_uses": [
            "Fresh leaves eaten daily as a classical Ayurvedic memory-enhancement practice",
            "Leaf juice mixed with honey taken for anxiety, ADHD, and cognitive support",
            "Leaf paste applied to scalp to strengthen hair follicles and reduce hair fall",
            "Leaf decoction used for epilepsy and convulsive disorders in traditional practice",
            "Fresh leaf juice used in Ayurveda for thyroid function support",
        ],
        "stem_uses": [
            "Whole stem (along with leaves) processed into Brahmi Ghrita (medicated ghee)",
            "Stem decoction used for insomnia and nervous system calming",
        ],
        "root_uses": [
            "Whole plant including roots used in some traditional preparations for asthma",
        ],
        "general_medicinal_uses": [
            "Enhances memory, concentration, and cognitive function significantly",
            "Reduces anxiety, stress, and symptoms of ADHD naturally",
            "Neuroprotective properties support long-term brain health",
            "Used in treating epilepsy and anxiety disorders in traditional practice",
        ],
        "pharmaceutical_uses": [
            "Standardised Bacoside-A extract used in patented cognitive supplement formulations",
            "Brahmi extract in clinical trials for Alzheimer's disease and vascular dementia",
            "Neuroprotective compounds studied for potential in ALS and Parkinson's disease",
            "Anxiolytic properties studied as an alternative to benzodiazepine drugs",
        ],
        "herbal_uses": [
            "Brahmi Ghrita (medicated ghee) is the classical Ayurvedic preparation for memory",
            "Saraswatarishta, a classical Ayurvedic tonic, has Brahmi as its primary ingredient",
            "Brahmi Vati used for epilepsy and convulsive disorders in classical Ayurveda",
            "Used in Siddha medicine as Neerbrahmi for cognitive and nervous conditions",
        ],
        "cosmetic_uses": [
            "Brahmi oil used as a premium hair tonic for brain-cooling and hair growth",
            "Leaf extract incorporated into scalp serums for anti-dandruff and soothing effects",
            "Used in natural face creams for antioxidant and anti-ageing benefits",
            "Brahmi extract added to aromatherapy products for anxiety and stress relief",
        ],
    },
    "Castor": {
        "subtitle": "The Palma Christi",
        "scientific_name": "Ricinus communis",
        "family": "Euphorbiaceae",
        "origin": "Eastern Africa and the Mediterranean basin",
        "description": "A fast-growing shrub with large, deeply lobed leaves. Cold-pressed castor oil is widely used in medicine and cosmetics for its unique ricinoleic acid content; seeds contain the highly toxic ricin.",
        "active_compounds": ["Ricinoleic acid", "Ricin (seeds only)", "Undecylenic acid", "Ricinine", "Flavonoids"],
        "leaf_uses": [
            "Warmed castor leaf applied as a poultice on inflamed joints and headaches",
            "Leaf paste applied externally to boils, abscesses, and swollen glands",
            "Leaf decoction used as a galactagogue to promote breast milk production",
            "Leaf used in traditional African medicine for wound healing and infection control",
        ],
        "stem_uses": [
            "Stem bark used in traditional formulations for liver and spleen disorders",
            "Stem extract used in folk medicine as an anti-inflammatory remedy",
        ],
        "root_uses": [
            "Root bark used in Ayurveda as Eranda Mool for rheumatic and neurological conditions",
            "Root decoction used for lumbago and sciatica in traditional practice",
            "Root preparations used in classical Ayurvedic Panchakarma treatments",
        ],
        "general_medicinal_uses": [
            "Castor oil is a powerful and reliable natural laxative for constipation",
            "Anti-inflammatory ricinoleic acid eases joint and muscle pain",
            "Promotes wound healing and skin regeneration when applied topically",
            "Strengthens hair follicles and stimulates significant hair growth",
        ],
        "pharmaceutical_uses": [
            "Castor oil is a USP-approved laxative and inducing agent in obstetrics",
            "Used as an excipient and carrier oil in many injectable pharmaceutical formulations",
            "Ricinoleic acid used in manufacturing biodegradable pharmaceutical polymers",
            "Undecylenic acid derived from castor used in antifungal pharmaceutical creams",
        ],
        "herbal_uses": [
            "Eranda (castor) oil is central to Ayurvedic Panchakarma detoxification therapy",
            "Used in classical preparations Eranda Paka and Erandabhrishta for Vata disorders",
            "Leaf poultices used in traditional African and Caribbean herbal healing practices",
        ],
        "cosmetic_uses": [
            "Castor oil is the base ingredient in many lipstick and lip gloss formulations",
            "Widely used in hair growth serums and brow/lash enhancement products",
            "Used in natural moisturisers and body butters for its thick emollient properties",
            "Incorporated into nail-strengthening treatments and cuticle oils",
        ],
    },
    "Curry_Leaf": {
        "subtitle": "The Kadi Patta Treasure",
        "scientific_name": "Murraya koenigii",
        "family": "Rutaceae",
        "origin": "Indian subcontinent and Sri Lanka",
        "description": "A small tropical tree whose aromatic leaves are indispensable in South Indian cuisine and medicine. Carbazole alkaloids, notably mahanimbine, show demonstrated hypoglycaemic and antidyslipidaemic activities.",
        "active_compounds": ["Mahanimbine", "Koenigine", "Carbazole alkaloids", "Beta-carotene", "Vitamin C"],
        "leaf_uses": [
            "Fresh leaves consumed daily shown to reduce blood sugar in Type 2 diabetics",
            "Leaf juice applied to hair scalp prevents premature greying and hair fall",
            "Leaf decoction used for diarrhoea, nausea, and digestive disorders",
            "Leaves eaten to protect liver and reduce cholesterol levels naturally",
            "Antioxidant-rich leaf extract neutralises free radicals and reduces inflammation",
        ],
        "stem_uses": [
            "Stem bark decoction used in traditional medicine for skin diseases",
            "Bark extract used in Ayurvedic preparations for bites and stings",
        ],
        "root_uses": [
            "Root bark used in traditional Siddha medicine for body pain and eruptions",
            "Root decoction used for kidney and urinary conditions in folk practice",
        ],
        "general_medicinal_uses": [
            "Rich in antioxidants that protect cells from oxidative damage",
            "Helps control blood sugar levels and improves insulin sensitivity",
            "Promotes hair growth and prevents premature greying",
            "Aids digestion and effectively relieves nausea",
        ],
        "pharmaceutical_uses": [
            "Mahanimbine studied as a potent antidiabetic and anti-obesity compound",
            "Carbazole alkaloids investigated for anticancer and antimicrobial activity",
            "Koenigine researched for antifungal pharmaceutical applications",
        ],
        "herbal_uses": [
            "Leaves used in classical South Indian Ayurvedic digestive preparations",
            "Incorporated in herbal hair oils for premature greying prevention",
            "Used in Siddha medicine as Karuveppilai for digestive and liver conditions",
        ],
        "cosmetic_uses": [
            "Curry leaf oil used in hair serums and oils for anti-greying effect",
            "Leaf extract incorporated in face creams for antioxidant skin protection",
            "Used in herbal shampoos for scalp health and hair strengthening",
            "Beta-carotene from leaves used in natural skin-brightening products",
        ],
    },
    "Doddapatre": {
        "subtitle": "Indian Borage",
        "scientific_name": "Plectranthus amboinicus",
        "family": "Lamiaceae",
        "origin": "Southern and Eastern Africa, naturalised across Asia",
        "description": "A succulent aromatic herb with soft velvety leaves and pungent oregano-like fragrance. Carvacrol and thymol give it strong antimicrobial activity, widely used in respiratory folk medicine.",
        "active_compounds": ["Carvacrol", "Thymol", "Rosmarinic acid", "Luteolin", "Ursolic acid"],
        "leaf_uses": [
            "Fresh leaf juice with honey is a traditional remedy for cough and sore throat",
            "Crushed leaves inhaled for immediate relief of nasal and sinus congestion",
            "Leaf decoction used for bronchitis, asthma, and chronic respiratory infections",
            "Warm leaf compress applied to chest for congestion and breathlessness",
            "Leaf juice used directly on insect bites and minor skin infections",
        ],
        "stem_uses": [
            "Stem decoction used for digestive complaints and flatulence",
            "Stem juice used in traditional preparations for renal conditions",
        ],
        "root_uses": [],
        "general_medicinal_uses": [
            "Treats cough, cold, and respiratory infections across multiple traditions",
            "Soothes sore throat and relieves nasal and bronchial congestion",
            "Anti-inflammatory and antimicrobial properties benefit many conditions",
            "Relieves digestive issues, flatulence, and stomach discomfort",
        ],
        "pharmaceutical_uses": [
            "Thymol extracted for use in antiseptic pharmaceutical and dental preparations",
            "Carvacrol studied for broad-spectrum antimicrobial pharmaceutical applications",
            "Rosmarinic acid investigated for anti-inflammatory drug development",
        ],
        "herbal_uses": [
            "Leaf juice is a primary folk remedy for cold and cough across South Asia and the Caribbean",
            "Used in Siddha medicine as Karpooravalli for respiratory conditions",
            "Part of traditional Indonesian and Filipino herbal cold/flu preparations",
        ],
        "cosmetic_uses": [
            "Leaf essential oil used in aromatherapy for respiratory and stress relief",
            "Carvacrol and thymol used in natural antimicrobial skincare products",
            "Incorporated into herbal mouthwash formulations for antibacterial action",
        ],
    },
    "Ekka": {
        "subtitle": "Crown Flower",
        "scientific_name": "Calotropis gigantea",
        "family": "Apocynaceae",
        "origin": "South and Southeast Asia",
        "description": "A large shrub with waxy crown-shaped flowers. The latex-rich plant contains cardiac glycosides and is used in traditional medicine with extreme caution under expert supervision.",
        "active_compounds": ["Calotropin", "Calactin", "Uscharin", "Voruscharin", "Calotropagenin"],
        "leaf_uses": [
            "Warmed leaf applied externally to relieve joint pain and arthritis",
            "Leaf paste used in controlled traditional treatments for skin diseases",
            "Leaf extract applied to wounds in very small doses for accelerated healing",
            "⚠️ CAUTION: All leaf uses must be external only — toxic if ingested",
        ],
        "stem_uses": [
            "Stem latex used in controlled Ayurvedic preparations for skin disorders",
            "Stem bark used in very controlled traditional formulations for leprosy",
            "⚠️ CAUTION: Latex and bark are highly toxic",
        ],
        "root_uses": [
            "Root bark used in Ayurveda (Arka) for digestive and respiratory conditions in micro-doses",
            "Root ash used externally for skin eruptions in controlled traditional practice",
            "⚠️ CAUTION: Root use requires strict Ayurvedic detoxification (Shodhana) protocols",
        ],
        "general_medicinal_uses": [
            "Used in treating chronic skin disorders under expert supervision",
            "Anti-inflammatory for arthritis and joint inflammation (external use)",
            "Traditionally applied for toothache relief with extreme caution",
            "All therapeutic use must be under qualified Ayurvedic or medical supervision",
        ],
        "pharmaceutical_uses": [
            "Calotropin studied as a potential anticancer compound in oncological research",
            "Cardiac glycosides researched for controlled cardiac rhythm applications",
            "Latex studied for anti-tumour and anti-inflammatory pharmaceutical research",
        ],
        "herbal_uses": [
            "Used as Arka in classical Ayurvedic formulations after mandatory detoxification",
            "Part of Siddha medicine formulations for skin diseases under toxic-herb protocols",
        ],
        "cosmetic_uses": [
            "Flowers occasionally used in floral arrangements and cultural rituals",
            "Not recommended for cosmetic use due to significant toxicity concerns",
        ],
    },
    "Ganike": {
        "subtitle": "Turkey Berry",
        "scientific_name": "Solanum torvum",
        "family": "Solanaceae",
        "origin": "Caribbean, naturalised in tropical Asia and Africa",
        "description": "A wild prickly shrub bearing clusters of small green berries. Steroidal saponins and alkaloids show hepatoprotective and anti-inflammatory properties, widely used across Southeast Asian and Ayurvedic medicine.",
        "active_compounds": ["Torvonin", "Chlorogenin", "Diosgenin", "Neochlorogenin", "Vitamins A & C"],
        "leaf_uses": [
            "Leaf paste applied to skin eruptions, eczema, and fungal infections",
            "Leaf decoction used as a tonic for fever and immune support",
            "Leaves used in traditional medicine to treat coughs and respiratory ailments",
            "Leaf juice used externally for joint pain and inflammatory swellings",
        ],
        "stem_uses": [
            "Stem bark decoction used for liver disorders and jaundice",
            "Stem extract used in traditional preparations for rheumatic pain",
        ],
        "root_uses": [
            "Root used in traditional African medicine for sexually transmitted infections",
            "Root decoction used for intestinal worms and parasitic infections",
        ],
        "general_medicinal_uses": [
            "Anti-inflammatory and hepatoprotective properties benefit liver health",
            "Treats liver disorders and supports overall liver function",
            "Used for skin conditions including eczema and fungal infections",
            "Reduces fever and acts as a mild sedative and analgesic",
        ],
        "pharmaceutical_uses": [
            "Diosgenin extracted as a precursor in steroid hormone pharmaceutical synthesis",
            "Steroidal saponins studied for anti-inflammatory drug development",
            "Torvonin investigated for hepatoprotective pharmaceutical applications",
        ],
        "herbal_uses": [
            "Used in Siddha medicine as Sundaikkai for digestive and liver conditions",
            "Small berries used in South Indian Ayurvedic cooking as a medicinal food",
            "Widely used in Thai traditional medicine for cough and respiratory conditions",
        ],
        "cosmetic_uses": [
            "Berry extract used in herbal skin preparations for anti-inflammatory benefits",
            "Diosgenin from berries used in phytoestrogen cosmetic formulations",
        ],
    },
    "Gauva": {
        "subtitle": "The Tropical Apple",
        "scientific_name": "Psidium guajava",
        "family": "Myrtaceae",
        "origin": "Mexico and Central America",
        "description": "A small evergreen tree bearing nutritionally rich fruits. The leaves, high in quercetin and flavonoids, are widely used in ethnomedicine for diarrhoea and blood-sugar management.",
        "active_compounds": ["Quercetin", "Guaijaverin", "Ellagic acid", "Vitamin C", "Carotenoids"],
        "leaf_uses": [
            "Leaf decoction is a highly effective traditional treatment for acute diarrhoea",
            "Fresh leaf tea consumed daily to reduce blood sugar in Type 2 diabetes",
            "Leaf extract applied topically for acne, skin infections, and wound healing",
            "Leaf decoction used as a gargle for tooth pain, gum disease, and mouth ulcers",
            "Leaf tea used for menstrual cramps and dysmenorrhoea relief",
            "Boiled leaf steam inhaled for nasal congestion and sinusitis",
        ],
        "stem_uses": [
            "Stem bark decoction used for diarrhoea and gastroenteritis",
            "Bark extract applied externally for skin infections and wounds",
        ],
        "root_uses": [
            "Root bark used in traditional medicine for epilepsy and convulsions",
            "Root decoction used for fever management in Central American folk medicine",
        ],
        "general_medicinal_uses": [
            "Exceptionally rich in Vitamin C and antioxidants for immune support",
            "Highly effective treatment for diarrhoea and digestive disorders",
            "Leaves manage blood sugar levels through alpha-glucosidase inhibition",
            "Anti-inflammatory properties aid wound healing and skin health",
        ],
        "pharmaceutical_uses": [
            "Quercetin from leaves used in anti-inflammatory supplement formulations",
            "Guaijaverin studied as a natural COX-2 inhibitor for anti-inflammatory drugs",
            "Leaf extract patented in antidiabetic pharmaceutical preparations in Asia",
            "Vitamin C from guava fruit used in pharmaceutical supplement manufacturing",
        ],
        "herbal_uses": [
            "Guava leaf tea is one of the most widely used herbal remedies for diarrhoea globally",
            "Used in Philippine traditional medicine (Bayabas) for wound healing and oral care",
            "Part of Ayurvedic preparations for diabetes and digestive conditions",
        ],
        "cosmetic_uses": [
            "Guava leaf extract used in skin-brightening creams for even complexion",
            "Antioxidant-rich fruit extract used in anti-ageing serums and masks",
            "Used in natural hair products for scalp health and anti-dandruff effect",
            "Vitamin C-rich extract used in illuminating face washes and toners",
        ],
    },
    "Geranium": {
        "subtitle": "The Rose-Scented Geranium",
        "scientific_name": "Pelargonium graveolens",
        "family": "Geraniaceae",
        "origin": "South Africa, naturalized in temperate Asia and Europe",
        "description": "A bushy aromatic shrub with deeply lobed, velvety leaves and small pink flowers. The leaves yield a rose-scented essential oil rich in geraniol and citronellol, widely used in perfumery, aromatherapy, and herbal medicine for its antimicrobial, anti-inflammatory, and anxiolytic properties.",
        "active_compounds": ["Geraniol", "Citronellol", "Linalool", "Isomenthone", "Citronellyl formate"],
        "leaf_uses": [
            "Leaf essential oil applied topically for skin infections, acne, and eczema",
            "Leaf tea used for anxiety, stress relief, and improving mood",
            "Crushed leaves applied to wounds for antiseptic and wound-healing effects",
            "Leaf oil diluted in carrier oil used for nerve pain and shingles relief",
            "Leaf decoction used as a gargle for sore throat and oral infections",
            "Leaf steam inhalation used for respiratory conditions and mental stress relief",
        ],
        "stem_uses": [
            "Stem decoction used in traditional preparations for digestive complaints",
            "Stem extract used in herbal preparations for mild anti-inflammatory effects",
        ],
        "root_uses": [
            "Root used in traditional South African medicine for diarrhoea and dysentery",
            "Root decoction used in folk medicine for fever and malarial conditions",
        ],
        "general_medicinal_uses": [
            "Powerful antimicrobial properties effective against various skin infections",
            "Anxiolytic and antidepressant benefits widely used in aromatherapy",
            "Anti-inflammatory properties benefit skin, joints, and nerve conditions",
            "Used in traditional medicine for diabetes management and neuropathic pain",
        ],
        "pharmaceutical_uses": [
            "Geraniol extracted for use in antimicrobial and antifungal pharmaceutical preparations",
            "Rose geranium oil studied as a natural agent for neuropathic pain management",
            "Citronellol investigated for antihypertensive and antispasmodic pharmaceutical applications",
            "Geranium extract studied for anti-inflammatory drug development",
        ],
        "herbal_uses": [
            "Rose geranium oil is a primary herbal remedy for skin and nerve conditions",
            "Used in traditional Cape Malay herbal medicine for stress and infections",
            "Incorporated in herbal tea blends for anxiety and digestive support",
            "Used in traditional South African medicine for respiratory and skin conditions",
        ],
        "cosmetic_uses": [
            "Rose geranium essential oil is a key ingredient in natural and luxury perfumery",
            "Used in facial toners and serums for balancing oily and acne-prone skin",
            "Incorporated in hair care products for scalp health and natural shine",
            "Used in natural deodorants and body sprays for its long-lasting antimicrobial fragrance",
        ],
    },
    "Henna": {
        "subtitle": "The Mehndi Plant",
        "scientific_name": "Lawsonia inermis",
        "family": "Lythraceae",
        "origin": "Northern Africa and South Asia",
        "description": "A flowering shrub whose powdered leaves yield a red-orange dye used for millennia in body art and hair colouring. Lawsone provides antifungal, antioxidant, and UV-protective properties.",
        "active_compounds": ["Lawsone", "Gallic acid", "Glucose", "Mannitol", "Mucilage"],
        "leaf_uses": [
            "Dried and powdered leaves are the source of natural henna dye for hair and skin",
            "Leaf paste applied to scalp treats dandruff, scalp infections, and hair fall",
            "Fresh leaf extract applied to burns, wounds, and skin rashes for cooling relief",
            "Leaf juice used traditionally to reduce fever and as a cooling agent",
            "Leaf paste on palms and soles is a traditional remedy for excessive sweating",
        ],
        "stem_uses": [
            "Stem bark decoction used in traditional medicine for jaundice and liver disorders",
            "Bark extract used for headache and fever management in folk medicine",
        ],
        "root_uses": [
            "Root decoction used in traditional medicine for jaundice and enlarged spleen",
            "Root extract used for skin eruptions and boils in North African folk medicine",
        ],
        "general_medicinal_uses": [
            "Natural hair dye that simultaneously conditions and strengthens hair",
            "Antifungal properties treat scalp infections effectively",
            "Cooling topical application relieves headaches and body heat",
            "Treats skin conditions like eczema and psoriasis when applied as paste",
        ],
        "pharmaceutical_uses": [
            "Lawsone studied for antifungal and antimicrobial pharmaceutical applications",
            "Gallic acid extracted for use in antioxidant and anti-inflammatory formulations",
            "Henna extract investigated for UV-photoprotective pharmaceutical applications",
        ],
        "herbal_uses": [
            "Mehndi paste used in Ayurveda as a body coolant and fever remedy",
            "Used in Unani medicine as Hinna for skin diseases and body cooling",
            "Part of traditional North African and Middle Eastern herbal hair care",
        ],
        "cosmetic_uses": [
            "Globally the most popular natural hair dye and conditioning treatment",
            "Primary ingredient in natural hair colour products ranging from brown to auburn",
            "Used in traditional body art (mehndi) for weddings and celebrations",
            "Incorporated into natural shampoos and hair packs for scalp health",
        ],
    },
    "Hibiscus": {
        "subtitle": "The Rose of Sharon",
        "scientific_name": "Hibiscus rosa-sinensis",
        "family": "Malvaceae",
        "origin": "East Asia (China or India)",
        "description": "A vigorous evergreen shrub with large showy flowers rich in anthocyanins. Clinical studies support use in reducing blood pressure and LDL cholesterol.",
        "active_compounds": ["Anthocyanins", "Hibiscin", "Gossypetin", "Hibiscic acid", "Vitamin C"],
        "leaf_uses": [
            "Leaf paste applied to hair promotes significant hair growth and prevents fall",
            "Leaf decoction used as a mild laxative and digestive aid",
            "Leaf extract applied externally to skin diseases and inflammatory conditions",
            "Leaves used in traditional medicine for treating wounds and boils",
        ],
        "stem_uses": [
            "Stem mucilage used as a natural hair gel and conditioner",
            "Stem bark fibre used in traditional medicine for digestive support",
        ],
        "root_uses": [
            "Root extract used in Ayurveda for managing hair loss and premature greying",
            "Root decoction used for cough and respiratory ailments in folk medicine",
        ],
        "general_medicinal_uses": [
            "Reduces blood pressure and LDL cholesterol levels significantly",
            "Rich source of antioxidants and Vitamin C for immune and cellular health",
            "Promotes hair growth and prevents premature hair loss",
            "Aids digestion and provides hepatoprotective support",
        ],
        "pharmaceutical_uses": [
            "Standardised hibiscus extract used in clinically studied antihypertensive supplements",
            "Anthocyanins from flowers used in antioxidant pharmaceutical formulations",
            "Hibiscic acid studied for anti-inflammatory pharmaceutical development",
        ],
        "herbal_uses": [
            "Hibiscus tea (Agua de Jamaica) is a popular herbal beverage for blood pressure",
            "Flower used in Ayurvedic hair oils and tonics as Japakusum",
            "Used in Egyptian Karkadeh herbal tea tradition for cardiovascular support",
        ],
        "cosmetic_uses": [
            "Hibiscus extract ('natural botox') used in anti-ageing and firming skincare",
            "Primary ingredient in herbal hair growth oils and shampoos",
            "Flower extract used as a natural pH indicator and colorant in cosmetics",
            "AHA-rich flower extract used in natural exfoliating face washes",
        ],
    },
    "Honge": {
        "subtitle": "Indian Beech Tree",
        "scientific_name": "Millettia pinnata",
        "family": "Fabaceae",
        "origin": "Indian subcontinent and Southeast Asia",
        "description": "A medium to large deciduous tree traditionally planted along roadsides. Karanja oil extracted from seeds is used for skin diseases, rheumatism, and biofuel.",
        "active_compounds": ["Karanjin", "Pongamol", "Kanugin", "Isopongaflavone", "Oleic acid"],
        "leaf_uses": [
            "Leaf paste applied to skin wounds, ulcers, and rheumatic joints",
            "Leaf decoction used for urinary disorders and related inflammation",
            "Fresh leaves used as green manure and in traditional wound healing",
            "Leaf extract used in traditional preparations for digestive disorders",
        ],
        "stem_uses": [
            "Stem bark decoction used for skin diseases and leprosy in traditional practice",
            "Bark extract used for ulcers and wounds as an antiseptic wash",
            "Bark preparation used for dyspepsia and digestive complaints",
        ],
        "root_uses": [
            "Root bark used in traditional medicine for leucoderma (vitiligo)",
            "Root decoction used for rheumatic conditions and muscular pain",
        ],
        "general_medicinal_uses": [
            "Karanja seed oil treats various chronic skin diseases effectively",
            "Anti-inflammatory properties help with rheumatic and arthritic conditions",
            "Antimicrobial oil accelerates wound and skin infection healing",
            "Used in treating digestive and urinary tract disorders",
        ],
        "pharmaceutical_uses": [
            "Karanjin studied as a natural UV-absorbing compound for sunscreen formulations",
            "Pongamol investigated for antifungal and antibacterial pharmaceutical activity",
            "Oil used as a biodiesel component and pharmaceutical carrier",
        ],
        "herbal_uses": [
            "Karanja oil used in Ayurvedic external preparations for skin diseases",
            "Used in traditional Panchakarma treatments for Vata disorders",
            "Bark used in Ayurvedic formulations for skin conditions like Kushtha",
        ],
        "cosmetic_uses": [
            "Pongamia oil used as a skin-conditioning ingredient in natural cosmetics",
            "Oil incorporated in natural insect-repellent personal care products",
            "Used in veterinary cosmetics and pet care products for skin conditions",
        ],
    },
    "Insulin": {
        "subtitle": "The Insulin Plant",
        "scientific_name": "Costus pictus",
        "family": "Costaceae",
        "origin": "Mexico and Central America, cultivated in tropical Asia",
        "description": "An ornamental ginger relative with spiral-arranged striped leaves. Corosolic acid activates insulin receptors and improves glucose uptake, widely used by diabetics in South India.",
        "active_compounds": ["Corosolic acid", "Diosgenin", "Quercetin", "Saponins", "Ascorbic acid"],
        "leaf_uses": [
            "1–2 fresh leaves chewed daily is the primary traditional diabetic treatment",
            "Leaf juice consumed regularly lowers fasting blood glucose significantly",
            "Leaf extract used to improve insulin sensitivity and receptor activity",
            "Dried leaf powder used in anti-diabetic herbal formulations",
            "Leaf decoction used for urinary tract infections and kidney health",
        ],
        "stem_uses": [
            "Stem used in traditional preparations for cholesterol management",
            "Stem extract used in Ayurvedic formulations for metabolic disorders",
        ],
        "root_uses": [
            "Root used in traditional South American medicine for diabetes and infections",
        ],
        "general_medicinal_uses": [
            "Primary traditional treatment for managing Type 2 diabetes naturally",
            "Rich in corosolic acid that significantly improves insulin sensitivity",
            "Antioxidant properties reduce oxidative cellular stress from hyperglycaemia",
            "Supports kidney health and reduces diabetic complications",
        ],
        "pharmaceutical_uses": [
            "Corosolic acid patented in several anti-diabetic supplement formulations",
            "Plant extract under clinical investigation for Type 2 diabetes management",
            "Diosgenin used as a precursor in steroid hormone pharmaceutical synthesis",
        ],
        "herbal_uses": [
            "Fresh leaf consumption is widely practised as a folk antidiabetic in South India",
            "Used in Siddha medicine as Insulin Keerai for diabetes and metabolic conditions",
            "Part of traditional diabetic management protocols across Southeast Asia",
        ],
        "cosmetic_uses": [
            "Antioxidant leaf extract used in skin-protective cosmetic formulations",
            "Quercetin from leaves used in anti-inflammatory skincare products",
        ],
    },
    "Jasmine": {
        "subtitle": "The Queen of Fragrant Flowers",
        "scientific_name": "Jasminum sambac",
        "family": "Oleaceae",
        "origin": "Tropical Asia — India, Myanmar, and Philippines",
        "description": "A twining shrub with intensely fragrant white flowers, national flower of the Philippines. Linalool and benzyl acetate in jasmine absolute are proven anxiolytic and sedative compounds.",
        "active_compounds": ["Linalool", "Benzyl acetate", "Indole", "Jasmone", "Benzyl benzoate"],
        "leaf_uses": [
            "Leaf juice applied to wounds and ulcers as an antiseptic agent",
            "Leaf decoction used for eye infections and conjunctivitis as a wash",
            "Leaf paste applied to skin diseases and dermatitis for anti-inflammatory relief",
            "Leaves used in traditional formulations for breast milk regulation",
        ],
        "stem_uses": [
            "Stem bark extract used in traditional medicine for fever management",
            "Stem decoction used for digestive complaints in Southeast Asian folk medicine",
        ],
        "root_uses": [
            "Root used in Chinese traditional medicine for pain relief and dislocations",
            "Root decoction used traditionally to cease lactation when needed",
        ],
        "general_medicinal_uses": [
            "Aromatherapy reduces anxiety, depression, and stress significantly",
            "Antiseptic properties promote wound healing and infection prevention",
            "Relieves menstrual cramps and associated hormonal pain",
            "Used in skincare for moisturising and powerful anti-ageing benefits",
        ],
        "pharmaceutical_uses": [
            "Linalool extracted for use in calming and anxiolytic aromatherapy formulations",
            "Benzyl acetate used in pharmaceutical fragrance compounds",
            "Jasmine extract investigated for antidepressant drug-alternative research",
        ],
        "herbal_uses": [
            "Jasmine tea is a widely consumed herbal beverage for stress and anxiety",
            "Used in Ayurvedic preparations for skin, mental health, and reproductive conditions",
            "Malligai (jasmine) used in Siddha medicine for eye diseases and skin disorders",
        ],
        "cosmetic_uses": [
            "Jasmine absolute is one of the most prized ingredients in luxury perfumery",
            "Used in high-end skin creams and serums for its anti-ageing and soothing effect",
            "Jasmine essential oil used in premium body oils and massage preparations",
            "Incorporated into hair serums and scalp treatments for fragrance and scalp health",
        ],
    },
    "Lemon": {
        "subtitle": "The Golden Citrus",
        "scientific_name": "Citrus limon",
        "family": "Rutaceae",
        "origin": "Northeast India and Northeast Pakistan",
        "description": "A small thorny evergreen tree believed to be a natural hybrid of bitter orange and citron. The juice is rich in Vitamin C; the peel contains limonene and hesperidin with anti-inflammatory activity.",
        "active_compounds": ["Vitamin C", "Citric acid", "Limonene", "Hesperidin", "Eriocitrin"],
        "leaf_uses": [
            "Leaf tea used for anxiety, insomnia, and nervous tension relief",
            "Crushed leaves inhaled for headache and nausea relief",
            "Leaf decoction used for fever reduction as a diaphoretic",
            "Leaf essential oil used in aromatherapy for mood elevation and stress relief",
        ],
        "stem_uses": [
            "Stem bark used in traditional medicine for malaria and fever in some regions",
            "Bark extract used as a bitter digestive tonic in folk practice",
        ],
        "root_uses": [
            "Root decoction used in traditional Southeast Asian medicine for diarrhoea",
        ],
        "general_medicinal_uses": [
            "Rich in Vitamin C that significantly boosts immunity and collagen synthesis",
            "Aids digestion, reduces bloating, and supports liver detoxification",
            "Antibacterial properties support oral health and prevent gum disease",
            "Natural diuretic that promotes kidney health and prevents kidney stones",
        ],
        "pharmaceutical_uses": [
            "Vitamin C (ascorbic acid) isolated from lemon used in pharmaceutical supplements",
            "Hesperidin and eriocitrin studied for cardiovascular protective pharmaceutical effects",
            "Limonene used as a solvent and active in pharmaceutical topical formulations",
            "Citric acid is a widely used pharmaceutical excipient and acidulant",
        ],
        "herbal_uses": [
            "Lemon juice with honey and ginger is a universal folk remedy for cold and flu",
            "Nimbu pani (lemon water) used as a classical Indian digestive and detox beverage",
            "Lemon essential oil used in herbal aromatherapy for stress and concentration",
        ],
        "cosmetic_uses": [
            "Lemon juice used as a natural skin brightener and dark spot treatment",
            "Vitamin C from lemon is a key active ingredient in skin-brightening serums",
            "Lemon essential oil used in perfumery for its fresh citrus top note",
            "Used in natural hair lightening treatments and scalp cleansing preparations",
        ],
    },
    "Lemon_grass": {
        "subtitle": "Fever Grass",
        "scientific_name": "Cymbopogon citratus",
        "family": "Poaceae",
        "origin": "Maritime Southeast Asia, cultivated pantropically",
        "description": "A tall perennial grass with strong citrus aroma from its high citral content. Its essential oil is rich in monoterpenes with documented antimicrobial, antifungal, and analgesic properties.",
        "active_compounds": ["Citral", "Citronellal", "Geraniol", "Myrcene", "Neral"],
        "leaf_uses": [
            "Leaf tea consumed for digestive issues, bloating, and stomach cramps",
            "Leaf decoction used as a diuretic for kidney and urinary health",
            "Leaf tea used to reduce fever and body temperature naturally",
            "Fresh leaves boiled for steam inhalation to relieve sinus congestion",
            "Leaf oil applied to joints for anti-inflammatory pain relief",
        ],
        "stem_uses": [
            "Stem base (lemongrass stalk) used in culinary medicine for digestive health",
            "Stem extract used in traditional preparations for cholesterol management",
            "Stem decoction used in Southeast Asian traditional medicine for fungal infections",
        ],
        "root_uses": [
            "Root decoction used for menstrual irregularities in folk medicine",
            "Root used in traditional Brazilian medicine for anxiety and nervous conditions",
        ],
        "general_medicinal_uses": [
            "Treats digestive issues, bloating, and stomach cramps effectively",
            "Anti-inflammatory and analgesic properties provide natural pain relief",
            "Reduces fever and acts as a reliable natural diuretic",
            "Antimicrobial properties help combat bacterial and fungal infections",
        ],
        "pharmaceutical_uses": [
            "Citral extracted for use in antifungal pharmaceutical preparations",
            "Geraniol studied as an anticancer and antimicrobial pharmaceutical compound",
            "Lemongrass essential oil used in pharmaceutical topical pain formulations",
        ],
        "herbal_uses": [
            "Lemongrass tea is a popular Caribbean and Asian herbal remedy for fever",
            "Used in Ayurveda as Bhustrina for digestive and respiratory conditions",
            "Part of traditional Thai and Vietnamese herbal medicine for pain and digestion",
        ],
        "cosmetic_uses": [
            "Essential oil widely used in perfumery for its fresh, citrus-green aroma",
            "Used in natural insect-repellent personal care products",
            "Incorporated into toning and astringent facial formulations",
            "Used in natural deodorants and body sprays for its antimicrobial fragrance",
        ],
    },
    "Mango": {
        "subtitle": "The King of Fruits",
        "scientific_name": "Mangifera indica",
        "family": "Anacardiaceae",
        "origin": "South Asia — India–Myanmar region",
        "description": "A large long-lived tropical tree and one of the most widely cultivated fruits. Mango leaves, bark, and kernel have established roles in Ayurvedic and folk medicine for managing diabetes and inflammation.",
        "active_compounds": ["Mangiferin", "Quercetin", "Norathyriol", "Gallic acid", "Beta-carotene"],
        "leaf_uses": [
            "Fresh mango leaf tea consumed to regulate blood sugar in diabetes",
            "Leaf decoction used for diabetic retinopathy and associated vascular damage",
            "Powdered dried leaves used in Ayurveda for bleeding disorders and gum disease",
            "Leaf smoke inhalation used in folk practice for hiccups and voice disorders",
            "Young tender leaves eaten as a food source of mangiferin and antioxidants",
        ],
        "stem_uses": [
            "Stem bark decoction used for diarrhoea, rheumatism, and dental problems",
            "Bark used as a haemostatic agent for bleeding gums and wounds",
            "Bark extract used in traditional preparations for skin diseases",
        ],
        "root_uses": [
            "Root bark used in traditional Indian medicine for fever and diarrhoea",
        ],
        "general_medicinal_uses": [
            "Rich in Vitamins A, C, and E for robust immune system support",
            "Improves digestion with natural amylase enzymes in the fruit",
            "Leaves and bark effectively manage blood sugar levels",
            "Antioxidant mangiferin protects cells from oxidative damage comprehensively",
        ],
        "pharmaceutical_uses": [
            "Mangiferin patented and studied extensively as an antidiabetic pharmaceutical compound",
            "Norathyriol from mango studied for anti-inflammatory drug development",
            "Mangiferin investigated for cardiovascular and neuroprotective pharmaceutical applications",
            "Beta-carotene extracted for use in Vitamin A pharmaceutical supplements",
        ],
        "herbal_uses": [
            "Mango leaf tea is a widely used traditional antidiabetic remedy globally",
            "Used in Ayurveda as Amra in formulations for diabetes and digestive health",
            "Bark decoction used in West African herbal medicine for diarrhoea and malaria",
        ],
        "cosmetic_uses": [
            "Mango butter is a premium moisturiser used in body butters and lip products",
            "Mango seed butter used in hair conditioners and scalp treatments",
            "Beta-carotene from fruit used in skin-brightening cosmetic formulations",
            "Mango extract used in anti-ageing serums for antioxidant skin protection",
        ],
    },
    "Mint": {
        "subtitle": "The Cool Refreshing Herb",
        "scientific_name": "Mentha spicata / Mentha piperita",
        "family": "Lamiaceae",
        "origin": "Europe and Asia, widely naturalized globally",
        "description": "A rhizomatous perennial herb famous for the crisp cooling sensation of its menthol. Modern studies confirm its efficacy in IBS, tension headaches, and respiratory congestion.",
        "active_compounds": ["Menthol", "Menthone", "Rosmarinic acid", "Eriocitrin", "Luteolin"],
        "leaf_uses": [
            "Fresh leaf tea relieves IBS symptoms, bloating, and nausea effectively",
            "Crushed leaves inhaled for immediate sinus decongestion and headache relief",
            "Leaf oil applied topically relieves muscle aches, tension headaches, and arthritis",
            "Fresh leaves chewed to promote oral hygiene and prevent bad breath",
            "Leaf decoction used for fever reduction and to promote diaphoresis",
            "Leaf extract used for calming skin irritations, hives, and rashes",
        ],
        "stem_uses": [
            "Stem decoction used in traditional medicine for digestive complaints",
            "Stem extract used in formulations for respiratory conditions",
        ],
        "root_uses": [],
        "general_medicinal_uses": [
            "Relieves digestive issues, nausea, and IBS symptoms consistently",
            "Natural decongestant for sinus and respiratory congestion",
            "Antimicrobial menthol promotes oral hygiene and dental health",
            "Cooling analgesic effect relieves tension headaches and migraines",
        ],
        "pharmaceutical_uses": [
            "Menthol is an approved active ingredient in cough drops and throat sprays",
            "Peppermint oil capsules are clinically approved treatments for IBS in many countries",
            "Menthol used as a penetration enhancer in topical pharmaceutical formulations",
            "Rosmarinic acid studied for anti-inflammatory pharmaceutical applications",
        ],
        "herbal_uses": [
            "Peppermint tea is one of the most popular herbal teas globally for digestion",
            "Used in Ayurveda as Pudina for digestive, fever, and respiratory conditions",
            "Part of traditional European herbal medicine for cold, flu, and indigestion",
        ],
        "cosmetic_uses": [
            "Peppermint essential oil used in toothpastes, mouthwashes, and breath fresheners",
            "Menthol used in cooling body sprays, after-shave products, and muscle gels",
            "Incorporated into shampoos and scalp tonics for cooling, invigorating effect",
            "Used in natural lip balms and lip plumping products for its tingling sensation",
        ],
    },
    "Nagadali": {
        "subtitle": "Blue Whorled Sage",
        "scientific_name": "Clerodendrum serratum",
        "family": "Lamiaceae",
        "origin": "India, Sri Lanka, and Southeast Asia",
        "description": "A shrub with attractive whorled blue-purple flowers, revered in Ayurveda as Bharangi. Used in classical Ayurvedic compounds for respiratory disorders and fever.",
        "active_compounds": ["Serratagenic acid", "3-episerratagenic acid", "Baicalein", "Luteolin", "Pectolinarigenin"],
        "leaf_uses": [
            "Leaf decoction used for chronic cough and respiratory congestion",
            "Leaf juice applied to skin diseases and inflammatory conditions",
            "Leaves used in traditional Ayurvedic preparations for fever and infection",
            "Leaf extract used in folk medicine for arthritic and joint conditions",
        ],
        "stem_uses": [
            "Root and stem are the primary medicinal parts used in Ayurveda",
            "Stem bark decoction used for fever, rheumatism, and liver conditions",
        ],
        "root_uses": [
            "Root is the primary part used in classical Ayurvedic Bharangi formulations",
            "Root decoction used as an expectorant for asthma and chronic bronchitis",
            "Root powder used in anti-fever classical formulations like Dashamoola",
            "Root extract used for digestive disorders and appetite stimulation",
        ],
        "general_medicinal_uses": [
            "Treats digestive disorders and provides hepatoprotective benefits",
            "Used in Ayurveda for chronic skin diseases and inflammatory conditions",
            "Anti-inflammatory for arthritis and joint pain management",
            "Stimulates appetite and improves sluggish metabolism",
        ],
        "pharmaceutical_uses": [
            "Baicalein from plant studied for anti-inflammatory drug development",
            "Luteolin investigated as a natural anti-cancer and antioxidant compound",
            "Plant extracts studied for anti-asthmatic pharmaceutical applications",
        ],
        "herbal_uses": [
            "Bharangi is a key ingredient in Bharangyadi Kwatha for respiratory conditions",
            "Part of Dashamoola (ten roots) classical Ayurvedic formulation",
            "Used in Siddha medicine as Cherutekku for fever and respiratory ailments",
        ],
        "cosmetic_uses": [
            "Root extract used in natural skin preparations for anti-inflammatory benefits",
            "Incorporated in herbal hair care products for scalp health",
        ],
    },
    "Neem": {
        "subtitle": "The Village Pharmacy",
        "scientific_name": "Azadirachta indica",
        "family": "Meliaceae",
        "origin": "Indian subcontinent and dry tropical regions",
        "description": "A fast-growing drought-tolerant tree. Azadirachtin and nimbin, limonoids unique to neem, give remarkable antibacterial, antifungal, and anti-inflammatory properties used in over 75 Ayurvedic preparations.",
        "active_compounds": ["Azadirachtin", "Nimbin", "Nimbidin", "Nimbidol", "Gedunin"],
        "leaf_uses": [
            "Fresh leaves chewed daily to purify blood and boost immunity",
            "Leaf paste applied to acne, pimples, and skin infections with strong antibacterial action",
            "Leaf decoction used as a bath for chicken pox, measles, and fever",
            "Leaf juice consumed for diabetes management and blood sugar control",
            "Leaves used in anti-malarial traditional preparations across sub-Saharan Africa",
            "Leaf extract used as a natural pesticide and insect repellent",
        ],
        "stem_uses": [
            "Stem twigs (datun) used as natural toothbrushes for superior oral hygiene",
            "Stem bark decoction used for skin diseases, fever, and inflammation",
            "Bark extract used in Ayurvedic preparations for diabetes and liver conditions",
        ],
        "root_uses": [
            "Root bark used in Ayurveda for skin diseases and intestinal worms",
            "Root decoction used in traditional medicine for urinary disorders",
        ],
        "general_medicinal_uses": [
            "Powerful antibacterial and antifungal activity across numerous conditions",
            "Treats acne, eczema, and various skin infections highly effectively",
            "Supports oral health and prevents gum disease and tooth decay",
            "Purifies blood, boosts immunity, and reduces systemic inflammation",
        ],
        "pharmaceutical_uses": [
            "Azadirachtin is an approved biopesticide used in organic agriculture",
            "Nimbidin studied as a potent anti-inflammatory pharmaceutical candidate",
            "Gedunin investigated for antimalarial pharmaceutical applications",
            "Neem extract used in approved topical pharmaceutical preparations for skin conditions",
        ],
        "herbal_uses": [
            "Central ingredient in numerous Ayurvedic skin disease formulations (Nimbadi Churna)",
            "Neem oil used in classical Ayurvedic Panchakarma treatments",
            "Used in Siddha medicine as Vembu for skin diseases and fever",
            "Incorporated in traditional Unani preparations as Neem for blood purification",
        ],
        "cosmetic_uses": [
            "Neem oil widely used in natural skin care for acne, eczema, and blemishes",
            "Incorporated in natural toothpastes and mouthwashes for oral health",
            "Used in anti-dandruff shampoos and scalp treatments for antifungal action",
            "Neem extract added to natural insect-repellent personal care products",
        ],
    },
    "Nithyapushpa": {
        "subtitle": "The Miracle Periwinkle",
        "scientific_name": "Catharanthus roseus",
        "family": "Apocynaceae",
        "origin": "Madagascar",
        "description": "An evergreen subshrub with pink or white flowers, source of vincristine and vinblastine — frontline anticancer alkaloids. Widely used in traditional antidiabetic practice.",
        "active_compounds": ["Vincristine", "Vinblastine", "Catharanthine", "Ajmalicine", "Leurosine"],
        "leaf_uses": [
            "Leaf tea used traditionally for managing blood sugar in Type 2 diabetes",
            "Leaf extract applied to wounds and skin infections for antiseptic action",
            "Leaf decoction used for hypertension management in Caribbean folk medicine",
            "Leaf paste applied externally to wasp stings and insect bites",
        ],
        "stem_uses": [
            "Stem extract used in traditional preparations for diabetes and blood pressure",
            "Stem alkaloids are extracted for pharmaceutical vincristine and vinblastine production",
        ],
        "root_uses": [
            "Root extract used in traditional preparations for toothache and gum disease",
            "Root used in Ayurvedic formulations for diabetes and skin conditions",
        ],
        "general_medicinal_uses": [
            "Source of life-saving anticancer alkaloids vincristine and vinblastine",
            "Traditionally manages blood sugar in diabetics",
            "Lowers high blood pressure naturally in traditional practice",
            "Antiseptic properties aid wound healing and skin conditions",
        ],
        "pharmaceutical_uses": [
            "Vincristine is a frontline chemotherapy drug for leukaemia and lymphoma",
            "Vinblastine used in treatment of Hodgkin's lymphoma and testicular cancer",
            "Ajmalicine used in pharmaceutical preparations for improving cerebral blood flow",
            "Plant is one of the world's most important sources of pharmaceutical alkaloids",
        ],
        "herbal_uses": [
            "Leaf tea used across Caribbean and African traditional medicine for diabetes",
            "Used in Siddha medicine as Nityakalyani for skin and diabetic conditions",
            "Part of traditional Indian folk medicine for diabetes management",
        ],
        "cosmetic_uses": [
            "Flower extract used in natural perfumery and floral cosmetic formulations",
            "Plant extract used in skin formulations for its antioxidant and antiseptic properties",
        ],
    },
    "Nooni": {
        "subtitle": "The Polynesian Wonder Fruit",
        "scientific_name": "Morinda citrifolia",
        "family": "Rubiaceae",
        "origin": "Southeast Asia and Australasia",
        "description": "A small rugged tropical tree bearing pungent yellowish-white fruits. Used for over 2,000 years in Polynesian folk medicine; iridoids and polysaccharides are its primary bioactive components.",
        "active_compounds": ["Proxeronine", "Damnacanthal", "Scopoletin", "Alizarin", "Anthraquinones"],
        "leaf_uses": [
            "Fresh leaves placed on wounds and applied for anti-inflammatory relief",
            "Leaf decoction used for fever, cough, and respiratory infections",
            "Leaf poultice applied to arthritic joints for pain and swelling reduction",
            "Leaf juice used in traditional Hawaiian medicine for general health maintenance",
        ],
        "stem_uses": [
            "Stem bark used as a yellow-brown dye in traditional Pacific Island culture",
            "Stem bark decoction used in traditional medicine for pain and fever",
        ],
        "root_uses": [
            "Root contains alizarin, used as a red dye and in traditional antifungal preparations",
            "Root decoction used in Polynesian traditional medicine for digestive and fever conditions",
        ],
        "general_medicinal_uses": [
            "Rich in antioxidants promoting overall cellular protection and health",
            "Treats joint pain and significantly reduces arthritic inflammation",
            "Antimicrobial and antiviral properties boost immune response",
            "Increases energy levels and reduces chronic fatigue",
        ],
        "pharmaceutical_uses": [
            "Damnacanthal investigated as a potent anticancer compound in oncology research",
            "Scopoletin studied for vasorelaxant and hypotensive pharmaceutical applications",
            "Proxeronine investigated as an enzyme activator in cellular health research",
        ],
        "herbal_uses": [
            "Noni juice is a globally marketed traditional health tonic beverage",
            "Leaves used in traditional Hawaiian and Pacific Island healing practices",
            "Used in Southeast Asian folk medicine for pain, infection, and immune support",
        ],
        "cosmetic_uses": [
            "Noni extract used in anti-ageing skin care for its antioxidant and collagen-supporting activity",
            "Incorporated in skin brightening and hyperpigmentation treatment formulations",
            "Used in hair care products for shine, strength, and scalp health",
        ],
    },
    "Pappaya": {
        "subtitle": "The Fruit of Angels",
        "scientific_name": "Carica papaya",
        "family": "Caricaceae",
        "origin": "Southern Mexico and Central America",
        "description": "A large fast-growing herbaceous plant. Papain and chymopapain, cysteine proteases in its latex, have broad medical applications. Leaf extracts are used in dengue fever management.",
        "active_compounds": ["Papain", "Chymopapain", "Carpaine", "Caricin", "Lycopene"],
        "leaf_uses": [
            "Leaf juice significantly increases platelet count in dengue fever patients",
            "Leaf extract consumed for malaria and fever management in traditional practice",
            "Leaf decoction used for digestive disorders, flatulence, and constipation",
            "Leaf paste applied externally to skin infections, ulcers, and wounds",
            "Leaf juice used for managing blood sugar and diabetes-related conditions",
        ],
        "stem_uses": [
            "Stem latex used as a source of crude papain for digestive enzyme therapy",
            "Stem decoction used in traditional medicine for diuretic and antifungal effects",
        ],
        "root_uses": [
            "Root used in traditional Central American medicine for urinary tract disorders",
            "Root decoction used for amoebic dysentery and intestinal parasites",
        ],
        "general_medicinal_uses": [
            "Papain enzyme powerfully aids protein digestion and improves gut health",
            "Leaf juice treats dengue fever by boosting platelet count significantly",
            "Anti-inflammatory properties accelerate wound healing throughout the body",
            "Rich antioxidants from lycopene prevent oxidative cellular damage",
        ],
        "pharmaceutical_uses": [
            "Papain is a commercially extracted enzyme used in digestive enzyme pharmaceuticals",
            "Chymopapain used in approved pharmaceutical injections for herniated discs",
            "Papain used in wound-debridement pharmaceutical preparations",
            "Carpaine studied for antihypertensive and antiamoebic pharmaceutical applications",
        ],
        "herbal_uses": [
            "Green papaya used in Ayurveda as Erandakarkati for digestive disorders",
            "Leaf tea is a primary traditional remedy for dengue across Southeast Asia",
            "Used in Caribbean and Latin American folk medicine for digestive and wound conditions",
        ],
        "cosmetic_uses": [
            "Papain enzyme widely used in exfoliating face masks and chemical peels",
            "Papaya extract used in skin brightening and spot reduction cosmetics",
            "Incorporated in hair products for protein strengthening and shine",
            "Papaya seed oil used in luxury skin care for antioxidant and emollient properties",
        ],
    },
    "Pepper": {
        "subtitle": "The King of Spices",
        "scientific_name": "Piper nigrum",
        "family": "Piperaceae",
        "origin": "Western Ghats of India",
        "description": "A woody climbing vine historically used as currency. Piperine uniquely enhances the bioavailability of many nutrients and pharmaceutical compounds by inhibiting intestinal metabolism.",
        "active_compounds": ["Piperine", "Chavicine", "Piperettine", "Piperyline", "Volatile oils"],
        "leaf_uses": [
            "Leaves used in traditional preparations for digestive support and antimicrobial action",
            "Leaf decoction used in folk medicine for cold, fever, and respiratory conditions",
            "Leaf paste applied externally for skin diseases in traditional Southeast Asian practice",
        ],
        "stem_uses": [
            "Unripe and ripe berries on the vine are the primary medicinal and culinary part",
            "Stem decoction used in traditional medicine for digestive and respiratory ailments",
        ],
        "root_uses": [
            "Root used in traditional medicine for fever and digestive complaints",
            "Root decoction used in Ayurveda for urinary tract conditions",
        ],
        "general_medicinal_uses": [
            "Enhances bioavailability of nutrients and pharmaceutical compounds up to 20x",
            "Anti-inflammatory and antioxidant properties benefit multiple conditions",
            "Aids digestion, relieves constipation, and improves gut motility",
            "Antimicrobial properties treat respiratory and gastrointestinal infections",
        ],
        "pharmaceutical_uses": [
            "Piperine is patented as BioPerine, an approved bioavailability enhancer in supplements",
            "Used to significantly increase the absorption of curcumin in pharmaceutical formulations",
            "Piperine studied for antidepressant, anti-epileptic, and neuroprotective drug development",
            "Volatile oils researched for antimicrobial pharmaceutical applications",
        ],
        "herbal_uses": [
            "Trikatu (black pepper, long pepper, ginger) is a foundational Ayurvedic bioavailability formula",
            "Pepper used in classical Ayurvedic preparations for Kapha and digestive disorders",
            "Used in Unani medicine as Filfil Siyah for respiratory and digestive conditions",
        ],
        "cosmetic_uses": [
            "Pepper extract used in hair growth stimulating products for scalp circulation",
            "Piperine incorporated in body-warming massage oils and pain-relief rubs",
            "Black pepper essential oil used in spicy masculine fragrance compositions",
            "Used in anti-cellulite body treatments for its thermogenic properties",
        ],
    },
    "Pomegranate": {
        "subtitle": "The Jewel of Autumn",
        "scientific_name": "Punica granatum",
        "family": "Lythraceae",
        "origin": "Iran and the Himalayas in Northern India",
        "description": "A fruit-bearing deciduous shrub cultivated since ancient times. Unique punicalagins have documented anti-inflammatory and cardioprotective effects; one of the highest antioxidant capacities of any food.",
        "active_compounds": ["Punicalagins", "Punicic acid", "Ellagic acid", "Luteolin", "Anthocyanins"],
        "leaf_uses": [
            "Leaf decoction used as a gargle for sore throat and tonsillitis",
            "Leaf tea used for digestive disorders and diarrhoea management",
            "Leaf extract applied to wounds for antiseptic and healing properties",
            "Leaf paste used externally for inflammatory skin conditions",
        ],
        "stem_uses": [
            "Stem bark decoction used for tapeworm and intestinal parasite expulsion",
            "Bark preparation used for diarrhoea and dysentery in traditional practice",
            "Bark decoction used for mouth ulcers and gum disease as a wash",
        ],
        "root_uses": [
            "Root bark is the most potent anthelmintic part, used for intestinal worms",
            "Root decoction used for fever management in traditional Ayurvedic practice",
        ],
        "general_medicinal_uses": [
            "Exceptional antioxidant capacity protects cardiovascular health comprehensively",
            "Anti-inflammatory punicalagins reduce risk of chronic inflammatory diseases",
            "Treats diarrhoea and various digestive disorders effectively",
            "Strengthens immune function and combats infections naturally",
        ],
        "pharmaceutical_uses": [
            "Punicalagins are patented active compounds in cardiovascular supplement formulations",
            "Ellagic acid investigated as an anticancer and chemopreventive pharmaceutical",
            "Pomegranate extract used in approved anti-inflammatory nutritional supplements",
            "Punicic acid (conjugated fatty acid) studied for anticancer pharmaceutical applications",
        ],
        "herbal_uses": [
            "Pomegranate peel used in classical Ayurvedic formulations for diarrhoea (Dadimadi Ghrita)",
            "Used in Unani medicine as Anar for digestive, cardiac, and anti-anaemic preparations",
            "Bark decoction used as an anthelmintic in traditional Chinese and Ayurvedic medicine",
        ],
        "cosmetic_uses": [
            "Pomegranate seed oil used in premium anti-ageing serums for skin regeneration",
            "Ellagic acid used in skin-brightening and anti-hyperpigmentation cosmetics",
            "Fruit extract used in antioxidant-rich moisturisers and face masks",
            "Incorporated into scalp serums for anti-inflammatory and hair growth benefits",
        ],
    },
    "Raktachandini": {
        "subtitle": "Red Sandalwood",
        "scientific_name": "Pterocarpus santalinus",
        "family": "Fabaceae",
        "origin": "Southern Eastern Ghats of India (endemic)",
        "description": "A medium-sized slow-growing endemic tree with distinctive dark red heartwood. Contains pterostilbene and santalin pigments with antioxidant, anti-inflammatory, and antidiabetic properties.",
        "active_compounds": ["Santalin A & B", "Pterostilbene", "Liquiritigenin", "Isoliquiritigenin", "Tannins"],
        "leaf_uses": [
            "Leaf decoction used for fever management and inflammation reduction",
            "Leaf paste applied to skin diseases and boils in traditional practice",
            "Leaves used in traditional formulations for digestive disorders",
        ],
        "stem_uses": [
            "Heartwood (red sandalwood) is the primary medicinal part, powdered and used extensively",
            "Heartwood paste applied to skin for its anti-inflammatory and cooling properties",
            "Stem wood decoction used for fever, liver disorders, and dysentery",
        ],
        "root_uses": [
            "Root decoction used for digestive disorders and bilious conditions",
            "Root used in traditional preparations for rheumatic conditions",
        ],
        "general_medicinal_uses": [
            "Anti-inflammatory properties treat fever, infections, and inflammation",
            "Treats digestive disorders including dysentery effectively",
            "Antimicrobial properties benefit various skin conditions",
            "Natural antioxidant used as a traditional food colorant and medicine",
        ],
        "pharmaceutical_uses": [
            "Pterostilbene studied as a bioavailable analogue of resveratrol for cardiovascular health",
            "Santalin extracted as a natural red colorant with pharmaceutical applications",
            "Tannins used in astringent pharmaceutical preparations for oral health",
        ],
        "herbal_uses": [
            "Heartwood powder used in classical Ayurvedic formulations for skin and liver disorders",
            "Used in Siddha medicine for Pittaja conditions and skin inflammation",
            "Red sandalwood paste used in traditional Indian body-cooling and fever remedies",
        ],
        "cosmetic_uses": [
            "Red sandalwood paste used in traditional Indian face packs for complexion improvement",
            "Santalin dye used as a natural red colorant in cosmetics and personal care",
            "Incorporated in anti-inflammatory skin creams and soothing face masks",
            "Heartwood extract used in luxury skin care for its antioxidant properties",
        ],
    },
    "Rose": {
        "subtitle": "The Queen of Flowers",
        "scientific_name": "Rosa damascena / Rosa centifolia",
        "family": "Rosaceae",
        "origin": "Asia, North Africa, and Europe",
        "description": "One of the most widely cultivated flowering plants with medicinal use dating to ancient Persia. Rose oil and rose water, rich in geraniol and citronellol, are used in aromatherapy for depression and anxiety.",
        "active_compounds": ["Geraniol", "Citronellol", "Kaempferol", "Quercetin", "Vitamin C (hips)"],
        "leaf_uses": [
            "Leaf paste used externally for skin inflammations and minor wounds",
            "Leaf decoction used as a mild astringent for digestive disorders",
            "Leaves used in traditional preparations for fever management",
        ],
        "stem_uses": [
            "Rose hip (pseudo-fruit) is among the richest natural sources of Vitamin C",
            "Stem bark used in traditional preparations for diarrhoea and digestive conditions",
        ],
        "root_uses": [
            "Root decoction used in traditional preparations for fever and urinary conditions",
            "Root bark used in folk medicine for gastrointestinal disorders",
        ],
        "general_medicinal_uses": [
            "Anti-inflammatory properties reduce skin redness, irritation, and eczema",
            "Antibacterial and wound-healing properties for skin conditions",
            "Rose hip Vitamin C significantly boosts overall immunity",
            "Aromatherapy and internal use support digestive and mental health",
        ],
        "pharmaceutical_uses": [
            "Rose hip extract standardised for Vitamin C content in pharmaceutical supplements",
            "Kaempferol studied as an anti-inflammatory and chemopreventive pharmaceutical",
            "Geraniol studied for antimicrobial and anticancer pharmaceutical applications",
            "Rose water used as a pharmaceutical excipient in topical and ophthalmic preparations",
        ],
        "herbal_uses": [
            "Rose water (Arq-e-Gulab) is a classical Unani preparation for heart and brain health",
            "Gulkand (rose petal preserve) used in Ayurveda for digestive and cooling effects",
            "Rose hip tea is a popular herbal Vitamin C source across European tradition",
        ],
        "cosmetic_uses": [
            "Rose absolute is the most prized ingredient in luxury high-end perfumery",
            "Rose water used as a natural toner for skin hydration and pH balance",
            "Rose hip oil used in premium anti-ageing serums for regenerative properties",
            "Rose extract incorporated in hydrating face creams, mists, and masks globally",
        ],
    },
    "Sapota": {
        "subtitle": "The Sweet Chikoo",
        "scientific_name": "Manilkara zapota",
        "family": "Sapotaceae",
        "origin": "Southern Mexico and Central America",
        "description": "A long-lived evergreen tree bearing malt-flavoured rough-skinned fruits. Rich in dietary fibre and tannins; bark extracts contain triterpenes with anti-diarrhoeal and anti-inflammatory properties.",
        "active_compounds": ["Tannins", "Saponins", "Triterpenes", "Vitamins A & C", "Quinine"],
        "leaf_uses": [
            "Leaf decoction used for treating fever, cold, and respiratory infections",
            "Leaf extract applied to wounds for antiseptic and healing properties",
            "Leaf tea used as a mild diuretic and for digestive support",
            "Leaves used in traditional preparations for controlling diarrhoea",
        ],
        "stem_uses": [
            "Stem bark decoction used for diarrhoea and dysentery management",
            "Bark preparation used for fever and malaria in traditional Central American medicine",
            "Latex from stem (chicle) used traditionally as an astringent and wound sealant",
        ],
        "root_uses": [
            "Root decoction used in traditional medicine for urinary tract conditions",
            "Root used in folk preparations for febrile conditions",
        ],
        "general_medicinal_uses": [
            "Rich in anti-diarrhoeal tannins for digestive disorder management",
            "High antioxidant content protects against oxidative stress comprehensively",
            "Dietary fibre content supports sustained digestive health",
            "Vitamins A and C from fruit support immune and eye health",
        ],
        "pharmaceutical_uses": [
            "Latex (chicle) from stem historically used in pharmaceutical dental preparations",
            "Triterpenes from bark studied for anti-inflammatory pharmaceutical applications",
            "Tannins extracted for use in astringent pharmaceutical formulations",
        ],
        "herbal_uses": [
            "Sapota fruit used in traditional Ayurvedic preparations for energy and digestion",
            "Bark decoction used in traditional Mexican folk medicine for fever and diarrhoea",
            "Used in Caribbean traditional medicine for respiratory and digestive conditions",
        ],
        "cosmetic_uses": [
            "Chicle latex historically used as a base for chewing gum and dental preparations",
            "Fruit extract used in moisturising and antioxidant skin care formulations",
            "Seed oil used as an emollient in natural skin and hair care products",
        ],
    },
    "Tulasi": {
        "subtitle": "The Holy Basil of India",
        "scientific_name": "Ocimum tenuiflorum",
        "family": "Lamiaceae",
        "origin": "Indian subcontinent, Tropical Asia",
        "description": "A short-lived aromatic perennial herb considered sacred in Hinduism, planted in virtually every traditional Indian household. Classified as an adaptogen, its eugenol-rich volatile oil provides broad antimicrobial and immunomodulatory actions.",
        "active_compounds": ["Eugenol", "Ursolic acid", "Rosmarinic acid", "Caryophyllene", "Apigenin"],
        "leaf_uses": [
            "Fresh leaves chewed daily to boost immunity and prevent respiratory infections",
            "Leaf tea with ginger and honey is the most widely used Indian cold and flu remedy",
            "Leaf juice reduces blood sugar and supports diabetic management",
            "Leaf extract applied to skin infections, acne, and insect bites as antiseptic",
            "Leaf decoction used for fever, including malarial fever, across traditional practice",
            "Leaves used in classical Ayurvedic formulations for stress and anxiety relief",
        ],
        "stem_uses": [
            "Woody stems used in Ayurvedic rosary beads (mala) for mindfulness practices",
            "Stem decoction used for digestive disorders and headache management",
        ],
        "root_uses": [
            "Root used in classical Ayurvedic formulations for urinary disorders",
            "Root decoction used for bites and stings in traditional folk medicine",
        ],
        "general_medicinal_uses": [
            "Powerful adaptogen that significantly reduces stress, anxiety, and fatigue",
            "Treats cold, cough, and respiratory infections across all age groups",
            "Anti-inflammatory and broad-spectrum antimicrobial properties",
            "Supports blood sugar regulation and cardiovascular health",
        ],
        "pharmaceutical_uses": [
            "Eugenol extracted for use in dental analgesics and pharmaceutical antiseptics",
            "Ursolic acid studied extensively as an anti-inflammatory and anticancer compound",
            "Standardised Tulasi extract used in patented adaptogenic supplement formulations",
            "Rosmarinic acid from Tulasi used in pharmaceutical anti-inflammatory research",
        ],
        "herbal_uses": [
            "Primary ingredient in classical Ayurvedic preparations for respiratory conditions",
            "Used in Chyawanprash and other Rasayana formulations as an adaptogen",
            "Part of classical preparation Tulasi Swaras (fresh juice) in Ayurvedic Panchakarma",
            "Used in Siddha medicine as Thulasi for fever, infections, and stress conditions",
        ],
        "cosmetic_uses": [
            "Tulasi extract used in natural face washes and toners for acne control",
            "Incorporated in anti-ageing skin creams for its antioxidant ursolic acid content",
            "Used in herbal shampoos and conditioners for scalp health and hair strength",
            "Tulasi essential oil used in natural aromatherapy and stress-relief preparations",
        ],
    },
    "Wood_sorel": {
        "subtitle": "The Shamrock Sorrel",
        "scientific_name": "Oxalis corniculata",
        "family": "Oxalidaceae",
        "origin": "Cosmopolitan, origin possibly India",
        "description": "A low-growing clover-like perennial herb whose sour taste comes from oxalic acid. Historically used to prevent scurvy; used across Asia and Africa for skin conditions and fevers.",
        "active_compounds": ["Oxalic acid", "Vitamin C", "Flavonoids", "Tannins", "Beta-carotene"],
        "leaf_uses": [
            "Fresh leaves are rich in Vitamin C and eaten to treat and prevent scurvy",
            "Leaf juice applied to skin rashes, boils, and fungal infections",
            "Leaf paste used as a soothing poultice on insect stings and bites",
            "Leaf decoction used as an antipyretic for fever management",
            "Leaves eaten to stimulate appetite and improve digestion",
        ],
        "stem_uses": [
            "Stem decoction used in folk medicine for urinary conditions and gravel",
            "Stem juice used as a mild astringent for skin conditions",
        ],
        "root_uses": [
            "Root used in traditional formulations for liver and digestive conditions",
            "Root paste applied to boils and abscesses in folk practice",
        ],
        "general_medicinal_uses": [
            "Rich in Vitamin C, historically essential for preventing and treating scurvy",
            "Anti-inflammatory for skin conditions and inflammatory rashes",
            "Treats digestive issues and stimulates a sluggish appetite",
            "Antimicrobial properties promote wound healing and infection prevention",
        ],
        "pharmaceutical_uses": [
            "Vitamin C from plant used as a natural source in nutritional supplement research",
            "Oxalic acid studied for its role in calcium oxalate metabolism research",
            "Flavonoid fractions investigated for antioxidant pharmaceutical applications",
        ],
        "herbal_uses": [
            "Used in traditional Indian medicine as Changeri for liver and digestive conditions",
            "Part of African traditional medicine preparations for fever and skin diseases",
            "Used in traditional Chinese medicine for detoxification and digestive support",
        ],
        "cosmetic_uses": [
            "Plant extract used in skin-brightening formulations for Vitamin C activity",
            "Leaf juice used in natural preparation for blemish reduction",
            "Beta-carotene from plant incorporated in skin-protective cosmetic preparations",
        ],
    },
}

# ── Page contexts ───────────────────────────────────────────────────────────
PAGE_CONTEXTS = {
    'plant': {
        'page_type': 'plant',
        'hero_title': 'Identify <em>Medicinal</em><br>Plants',
        'hero_sub': 'Snap a leaf, stem, or whole plant and unlock its identity, medicinal benefits, and hidden secrets in seconds. Powered by AI, BOTANICA turns nature into knowledge—fast, smart, and right at your fingertips. 🌱✨.',
        'upload_hint': 'Best results with the full plant clearly visible in the frame',
    },
    'leaf': {
        'page_type': 'leaf',
        'hero_title': 'Identify by <em>Leaf</em>',
        'hero_sub': 'Snap a leaf, stem, or whole plant and unlock its identity, medicinal benefits, and hidden secrets in seconds. Powered by AI, BOTANICA turns nature into knowledge—fast, smart, and right at your fingertips. 🌱✨.',
        'upload_hint': 'Best results with a single leaf, well-lit against a neutral background',
    },
    'stem': {
        'page_type': 'stem',
        'hero_title': 'Identify by <em>Stem</em>',
        'hero_sub': 'Snap a leaf, stem, or whole plant and unlock its identity, medicinal benefits, and hidden secrets in seconds. Powered by AI, BOTANICA turns nature into knowledge—fast, smart, and right at your fingertips. 🌱✨.',
        'upload_hint': 'Best results with stem texture and colour clearly in focus under natural lighting',
    },
}

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
@app.route("/plant")
def page_plant():
    return render_template("index.html", **PAGE_CONTEXTS['plant'])

@app.route("/leaf")
def page_leaf():
    return render_template("index.html", **PAGE_CONTEXTS['leaf'])

@app.route("/stem")
def page_stem():
    return render_template("index.html", **PAGE_CONTEXTS['stem'])

@app.route("/result")
def page_result():
    return render_template("result.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img_array = np.array(img.resize((224, 224))) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    prediction = model.predict(img_array)
    predicted_class = class_names[int(np.argmax(prediction))]
    confidence = float(np.max(prediction)) * 100

    d = plant_details.get(predicted_class, {})
    return jsonify({
        "plant": predicted_class,
        "icon": plant_icons.get(predicted_class, "🌿"),
        "confidence": round(confidence, 2),
        "details": d,
    })


if __name__ == "__main__":
    app.run(debug=True)
