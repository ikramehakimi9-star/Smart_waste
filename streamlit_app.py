import os
import hashlib
import cv2
import joblib
import numpy as np
import streamlit as st
from skimage.feature import hog

# =====================================================
# 1. CONFIGURATION DE LA PAGE
# =====================================================
st.set_page_config(page_title="Smart Waste", page_icon="♻️", layout="centered")

# =====================================================
# 2. DESIGN (CSS)
# =====================================================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 720px;}

/* --- Bandeau du haut --- */
.hero {
    background: linear-gradient(135deg, #16a34a 0%, #0d9488 100%);
    border-radius: 22px;
    padding: 30px 22px;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(22, 163, 74, 0.25);
}
.hero-titre {color: #ffffff; font-size: 2.1rem; font-weight: 800; margin: 0;}
.hero-texte {color: #ecfdf5; font-size: 1.05rem; margin: 8px 0 0 0;}

/* --- Les 3 étapes --- */
.etape {
    background: #ffffff;
    border-radius: 16px;
    padding: 16px 10px;
    text-align: center;
    color: #1f2937;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.07);
    height: 100%;
}
.etape-emoji {font-size: 2rem;}
.etape-titre {font-weight: 700; margin-top: 4px;}
.etape-texte {font-size: 0.85rem; color: #6b7280;}

/* --- Carte résultat --- */
.resultat {
    background: #ffffff;
    border-radius: 20px;
    padding: 24px 22px;
    margin-top: 18px;
    color: #1f2937;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.10);
}
.res-emoji {font-size: 3rem; line-height: 1;}
.res-phrase {font-size: 1rem; color: #6b7280; margin-top: 8px;}
.res-classe {font-size: 2.2rem; font-weight: 800; margin: 2px 0 10px 0;}
.res-badge {
    display: inline-block;
    color: #ffffff;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
}
.res-bloc {
    background: #f9fafb;
    border-radius: 14px;
    padding: 12px 14px;
    margin-top: 14px;
    font-size: 0.97rem;
}
.res-bloc b {color: #111827;}

/* --- Barres de probabilités --- */
.ligne {display: flex; align-items: center; margin: 8px 0;}
.ligne-nom {width: 130px; font-size: 0.92rem; color: #1f2937;}
.piste {flex: 1; background: #e5e7eb; border-radius: 999px; height: 12px; overflow: hidden;}
.remplissage {height: 12px; border-radius: 999px;}
.ligne-pct {width: 56px; text-align: right; font-size: 0.9rem; color: #4b5563;}
.titre-section {font-weight: 700; color: #1f2937; margin-top: 22px; margin-bottom: 4px;}

/* --- Photo arrondie --- */
[data-testid="stImage"] img {border-radius: 18px;}

/* --- Pied de page --- */
.pied {text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 36px;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. INFORMATIONS SUR CHAQUE CLASSE
# =====================================================
infos = {
    "cardboard": {
        "emoji": "📦", "nom": "Carton", "couleur": "#b45309",
        "conseil": "Aplatis-le, puis dépose-le avec les papiers et cartons à recycler.",
        "astuce": "Un carton très gras (boîte de pizza) se recycle souvent mal : dans ce cas, poubelle ordinaire.",
    },
    "glass": {
        "emoji": "🍾", "nom": "Verre", "couleur": "#16a34a",
        "conseil": "Vide-le, puis dépose-le dans un conteneur à verre.",
        "astuce": "Le verre se recycle encore et encore sans perdre en qualité.",
    },
    "metal": {
        "emoji": "🥫", "nom": "Métal", "couleur": "#64748b",
        "conseil": "Vide-le et rince-le rapidement, puis dépose-le avec les emballages recyclables.",
        "astuce": "Canettes et boîtes de conserve se recyclent très bien.",
    },
    "paper": {
        "emoji": "📄", "nom": "Papier", "couleur": "#2563eb",
        "conseil": "Dépose-le avec les papiers à recycler.",
        "astuce": "Un papier mouillé ou gras ne se recycle pas.",
    },
    "plastic": {
        "emoji": "🧴", "nom": "Plastique", "couleur": "#f59e0b",
        "conseil": "Vide la bouteille ou l'emballage, puis dépose-le avec les emballages recyclables.",
        "astuce": "Écrase la bouteille pour gagner de la place.",
    },
    "trash": {
        "emoji": "🗑️", "nom": "Déchet ordinaire", "couleur": "#374151",
        "conseil": "Ça va dans la poubelle ordinaire.",
        "astuce": "En cas de doute, cherche le symbole de recyclage sur l'emballage.",
    },
}

# =====================================================
# 4. CHARGEMENT DU MODÈLE (une seule fois par session)
# =====================================================
if "paquet" not in st.session_state:
    dossier = os.path.dirname(os.path.abspath(__file__))
    st.session_state["paquet"] = joblib.load(os.path.join(dossier, "modele_hog_couleur_svm.joblib"))

paquet = st.session_state["paquet"]
scaler = paquet["scaler"]
pca = paquet["pca"]
model = paquet["model"]
CLASSES = []
for nom_classe in paquet["classes"]:
    CLASSES.append(str(nom_classe))

IMG_SIZE = tuple(paquet["img_size"])
HOG_ORIENTATIONS = paquet["hog_orientations"]
HOG_PIXELS_PER_CELL = tuple(paquet["hog_pixels_per_cell"])
HOG_CELLS_PER_BLOCK = tuple(paquet["hog_cells_per_block"])
BINS_H = paquet["bins_h"]
BINS_S = paquet["bins_s"]
BINS_V = paquet["bins_v"]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Compteurs de la session (pour la barre latérale)
if "compteurs" not in st.session_state:
    st.session_state["compteurs"] = {}
    for c in CLASSES:
        st.session_state["compteurs"][c] = 0
if "derniere_photo" not in st.session_state:
    st.session_state["derniere_photo"] = ""

# =====================================================
# 5. BARRE LATÉRALE (aide)
# =====================================================
st.sidebar.markdown("## ♻️ Smart Waste")
st.sidebar.markdown("**Comment ça marche ?**")
st.sidebar.markdown("1. Donne une photo de ton déchet\n2. Je l'analyse en quelques secondes\n3. Je te dis où le jeter")

st.sidebar.markdown("**📸 Pour une bonne photo**")
st.sidebar.markdown("- Un seul objet à la fois\n- Fond simple et uni\n- Bien éclairé, objet net\n- Objet assez proche")

st.sidebar.markdown("**Je reconnais 6 catégories**")
liste_categories = ""
for c in CLASSES:
    liste_categories = liste_categories + "- " + infos[c]["emoji"] + " " + infos[c]["nom"] + "\n"
st.sidebar.markdown(liste_categories)

# =====================================================
# 6. EN-TÊTE
# =====================================================
st.markdown('<div class="hero"><p class="hero-titre">♻️ Smart Waste</p><p class="hero-texte">Prends ton déchet en photo, je t\'aide à bien le trier.</p></div>', unsafe_allow_html=True)

# =====================================================
# 7. L'UTILISATEUR DONNE SA PHOTO
# =====================================================
choix = st.radio("Comment veux-tu donner ta photo ?",
                 ["📁 Importer une photo", "📸 Prendre une photo"], horizontal=True)

if choix == "📁 Importer une photo":
    fichier = st.file_uploader("Choisis une image (JPG ou PNG)", type=["jpg", "jpeg", "png"])
else:
    fichier = st.camera_input("Place ton déchet devant la caméra")

# =====================================================
# 8. PAS ENCORE DE PHOTO : message d'accueil
# =====================================================
if fichier is None:
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="etape"><div class="etape-emoji">📸</div><div class="etape-titre">1. Photo</div><div class="etape-texte">Prends ou importe une image</div></div>', unsafe_allow_html=True)
    col2.markdown('<div class="etape"><div class="etape-emoji">🔍</div><div class="etape-titre">2. Analyse</div><div class="etape-texte">Je regarde la forme et la couleur</div></div>', unsafe_allow_html=True)
    col3.markdown('<div class="etape"><div class="etape-emoji">✅</div><div class="etape-titre">3. Résultat</div><div class="etape-texte">Tu sais où le jeter</div></div>', unsafe_allow_html=True)

# =====================================================
# 9. UNE PHOTO EST DONNÉE : PRÉDICTION
# =====================================================
else:
    octets = fichier.getvalue()
    tableau = np.frombuffer(octets, np.uint8)
    img = cv2.imdecode(tableau, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Oups, je n'arrive pas à lire cette image. Essaie avec une autre photo.")
    else:
        # --- Afficher la photo (au milieu) ---
        gauche, milieu, droite = st.columns([1, 3, 1])
        milieu.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # --- Analyse ---
        with st.spinner("Je regarde ton déchet... 🔍"):
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

            # Forme (HOG)
            gris = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            vecteur_hog = hog(gris, orientations=HOG_ORIENTATIONS,
                              pixels_per_cell=HOG_PIXELS_PER_CELL,
                              cells_per_block=HOG_CELLS_PER_BLOCK,
                              block_norm="L2-Hys")

            # Couleur (HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [BINS_H], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [BINS_S], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [BINS_V], [0, 256]).flatten()
            hist_h = hist_h / hist_h.sum()
            hist_s = hist_s / hist_s.sum()
            hist_v = hist_v / hist_v.sum()

            # Scaler + PCA + SVM
            vecteur = np.concatenate([vecteur_hog, hist_h, hist_s, hist_v]).reshape(1, -1)
            vecteur = scaler.transform(vecteur)
            vecteur = pca.transform(vecteur)
            proba = model.predict_proba(vecteur)[0]

        # --- Classement des classes, de la plus probable à la moins probable ---
        ordre = np.argsort(proba)[::-1]
        premier = CLASSES[int(ordre[0])]
        second = CLASSES[int(ordre[1])]
        confiance = round(float(proba[ordre[0]]) * 100, 1)
        confiance2 = round(float(proba[ordre[1]]) * 100, 1)

        info = infos[premier]
        couleur = info["couleur"]

        # --- Compter la photo une seule fois (même si la page se recharge) ---
        signature = hashlib.md5(octets).hexdigest()
        if signature != st.session_state["derniere_photo"]:
            st.session_state["derniere_photo"] = signature
            st.session_state["compteurs"][premier] = st.session_state["compteurs"][premier] + 1

        # =================================================
        # 10. RÉSULTAT (message adapté à la confiance)
        # =================================================
        if confiance >= 75:
            phrase = "J'en suis presque certain, c'est du :"
            texte_badge = "Très sûr · " + str(confiance) + " %"
            couleur_badge = "#16a34a"
        elif confiance >= 50:
            phrase = "Je pense que c'est du :"
            texte_badge = "Assez sûr · " + str(confiance) + " %"
            couleur_badge = "#f59e0b"
        else:
            phrase = "Hmm, je ne suis pas sûr. Ma meilleure piste :"
            texte_badge = "Peu sûr · " + str(confiance) + " %"
            couleur_badge = "#dc2626"

        carte = '<div class="resultat" style="border-left: 8px solid ' + couleur + ';">'
        carte = carte + '<div class="res-emoji">' + info["emoji"] + '</div>'
        carte = carte + '<div class="res-phrase">' + phrase + '</div>'
        carte = carte + '<div class="res-classe" style="color:' + couleur + ';">' + info["nom"] + '</div>'
        carte = carte + '<span class="res-badge" style="background:' + couleur_badge + ';">' + texte_badge + '</span>'

        if confiance < 50:
            info2 = infos[second]
            carte = carte + '<div class="res-bloc">🤔 <b>Autre possibilité :</b> ' + info2["emoji"] + ' ' + info2["nom"] + ' (' + str(confiance2) + ' %).<br>Essaie une photo plus nette, avec un seul objet sur un fond simple.</div>'

        carte = carte + '<div class="res-bloc">🗂️ <b>Où le jeter :</b> ' + info["conseil"] + '</div>'
        carte = carte + '<div class="res-bloc">💡 <b>Le savais-tu ?</b> ' + info["astuce"] + '</div>'
        carte = carte + '</div>'
        st.markdown(carte, unsafe_allow_html=True)

        # =================================================
        # 11. DÉTAIL DES PROBABILITÉS (barres)
        # =================================================
        st.markdown('<div class="titre-section">📊 Ce que j\'ai hésité à répondre</div>', unsafe_allow_html=True)

        barres = ""
        for k in ordre:
            nom_k = CLASSES[int(k)]
            pct = round(float(proba[k]) * 100, 1)
            largeur = max(pct, 2)
            barres = barres + '<div class="ligne">'
            barres = barres + '<div class="ligne-nom">' + infos[nom_k]["emoji"] + ' ' + infos[nom_k]["nom"] + '</div>'
            barres = barres + '<div class="piste"><div class="remplissage" style="width:' + str(largeur) + '%; background:' + infos[nom_k]["couleur"] + ';"></div></div>'
            barres = barres + '<div class="ligne-pct">' + str(pct) + ' %</div>'
            barres = barres + '</div>'
        st.markdown(barres, unsafe_allow_html=True)

# =====================================================
# 12. BARRE LATÉRALE : COMPTEUR DE LA SESSION
# =====================================================
total = 0
for c in CLASSES:
    total = total + st.session_state["compteurs"][c]

st.sidebar.markdown("---")
st.sidebar.markdown("**📈 Ma session**")
if total == 0:
    st.sidebar.markdown("Aucun déchet trié pour l'instant.")
else:
    resume = "Déchets triés : **" + str(total) + "**\n"
    for c in CLASSES:
        nombre = st.session_state["compteurs"][c]
        if nombre > 0:
            resume = resume + "- " + infos[c]["emoji"] + " " + infos[c]["nom"] + " : " + str(nombre) + "\n"
    st.sidebar.markdown(resume)

# =====================================================
# 13. PIED DE PAGE
# =====================================================
st.markdown('<div class="pied">Smart Waste · Modèle HOG + couleur + SVM · Je peux me tromper, en cas de doute vérifie l\'emballage.</div>', unsafe_allow_html=True)
