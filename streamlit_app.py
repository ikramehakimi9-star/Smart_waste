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
st.set_page_config(
    page_title="Smart Waste",
    page_icon="♻️",
    layout="wide",
)

# =====================================================
# 2. DESIGN (CSS)
# =====================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

/* ---------- Une seule police pour toute la page ---------- */
html, body, .stApp, .stMarkdown, .stMarkdown *, [data-testid="stMarkdownContainer"] *,
button, label, input, textarea, [data-baseweb="tab"] *, [data-testid="stMetric"] * {
    font-family: 'Poppins', 'Segoe UI', Roboto, Arial, sans-serif !important;
}

/* ---------- Plus de barre latérale ---------- */
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {display: none !important;}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px;}

/* ---------- Bandeau ---------- */
.hero {
    background: linear-gradient(135deg, #16a34a 0%, #0d9488 100%);
    border-radius: 22px;
    padding: 28px 24px;
    text-align: center;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(22, 163, 74, 0.25);
}
.hero-titre {color: #fff; font-size: 2rem; font-weight: 800; margin: 0;}
.hero-texte {color: #ecfdf5; font-size: 1.05rem; margin: 6px 0 0 0;}

/* ---------- Barre de navigation (onglets) ---------- */
div[data-baseweb="tab-list"] {
    gap: 16px;
    background: #ffffff;
    border-radius: 18px;
    padding: 8px 12px;
    box-shadow: 0 3px 14px rgba(0,0,0,0.07);
    justify-content: center;
    margin-bottom: 18px;
}
button[data-baseweb="tab"] {
    height: auto;
    padding: 10px 24px;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    color: #4b5563;
    transition: background 0.15s;
}
button[data-baseweb="tab"]:hover {background: #f0fdf4; color: #166534;}
button[data-baseweb="tab"][aria-selected="true"] {background: #dcfce7; color: #166534;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display: none;}

/* ---------- Titres de zone ---------- */
.zone-titre {
    font-size: 1.15rem; font-weight: 700; color: #14532d;
    margin: 6px 0 10px 0;
}
.zone-num {
    display: inline-block; background: #16a34a; color: #fff;
    width: 26px; height: 26px; line-height: 26px; text-align: center;
    border-radius: 50%; font-size: 0.9rem; margin-right: 8px;
}

/* ---------- Carte générique ---------- */
.carte {
    background: #fff; border-radius: 18px; padding: 18px 20px;
    box-shadow: 0 3px 14px rgba(0,0,0,0.07); color: #1f2937;
}

/* ---------- Zone des consignes (conseils photo) ---------- */
.consignes {
    background: #ecfdf5; border: 1px solid #bbf7d0; border-radius: 16px;
    padding: 14px 18px; margin-top: 14px; color: #14532d;
}
.consignes-titre {font-weight: 700; margin-bottom: 6px;}
.consignes ul {margin: 0; padding-left: 20px;}
.consignes li {margin: 3px 0; font-size: 0.95rem;}

/* ---------- Zone d'attente (avant la photo) ---------- */
.attente {text-align: center; padding: 26px 18px;}
.attente-emoji {font-size: 3.2rem; line-height: 1;}
.attente-titre {font-weight: 700; font-size: 1.15rem; margin: 8px 0 14px 0;}
.etape {display: flex; align-items: center; text-align: left; margin: 10px 0;}
.etape-emoji {
    font-size: 1.6rem; background: #f0fdf4; border-radius: 12px;
    width: 46px; height: 46px; display: flex; align-items: center;
    justify-content: center; margin-right: 12px; flex-shrink: 0;
}
.etape-titre {font-weight: 600; font-size: 0.95rem;}
.etape-texte {font-size: 0.87rem; color: #6b7280;}

/* ---------- Carte résultat ---------- */
.res-haut {display: flex; align-items: center;}
.res-emoji {font-size: 3.4rem; line-height: 1; margin-right: 16px;}
.res-phrase {font-size: 0.95rem; color: #6b7280;}
.res-classe {font-size: 1.8rem; font-weight: 800; margin: 0 0 6px 0; line-height: 1.1;}
.res-badge {
    display: inline-block; color: #fff; font-size: 0.82rem; font-weight: 600;
    padding: 4px 12px; border-radius: 999px;
}
.res-bloc {
    background: #f9fafb; border-radius: 14px; padding: 12px 14px;
    margin-top: 12px; font-size: 0.95rem;
}
.res-bloc b {color: #111827;}
.res-alerte {background: #fffbeb; border: 1px solid #fde68a;}

/* ---------- Barres de probabilités ---------- */
.ligne {display: flex; align-items: center; margin: 7px 0;}
.ligne-nom {width: 150px; font-size: 0.9rem; color: #1f2937;}
.piste {flex: 1; background: #e5e7eb; border-radius: 999px; height: 11px; overflow: hidden;}
.remplissage {height: 11px; border-radius: 999px;}
.ligne-pct {width: 56px; text-align: right; font-size: 0.88rem; color: #4b5563;}

/* ---------- Guide des catégories ---------- */
.cat {border-top: 6px solid #16a34a; height: 100%;}
.cat-emoji {font-size: 2.2rem;}
.cat-nom {font-weight: 700; font-size: 1.15rem; margin: 2px 0 8px 0;}
.cat-texte {font-size: 0.95rem; margin-bottom: 8px;}
.cat-astuce {font-size: 0.85rem; color: #6b7280;}

/* ---------- Photo arrondie ---------- */
[data-testid="stImage"] img {border-radius: 16px;}

/* ---------- Pied de page ---------- */
.pied {text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 30px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =====================================================
# 3. INFORMATIONS SUR CHAQUE CLASSE
# =====================================================
INFOS = {
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

CONSIGNES_PHOTO = [
    "Un seul objet à la fois",
    "Fond simple et uni",
    "Bien éclairé, objet net",
    "Objet assez proche de la caméra",
]

# =====================================================
# 4. CHARGEMENT DU MODÈLE (mis en cache)
# =====================================================
@st.cache_resource(show_spinner="Chargement du modèle...")
def charger_modele():
    dossier = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(dossier, "modele_hog_couleur_svm.joblib"))


paquet = charger_modele()
scaler = paquet["scaler"]
pca = paquet["pca"]
model = paquet["model"]
CLASSES = [str(c) for c in paquet["classes"]]

IMG_SIZE = tuple(paquet["img_size"])
HOG_ORIENTATIONS = paquet["hog_orientations"]
HOG_PIXELS_PER_CELL = tuple(paquet["hog_pixels_per_cell"])
HOG_CELLS_PER_BLOCK = tuple(paquet["hog_cells_per_block"])
BINS_H = paquet["bins_h"]
BINS_S = paquet["bins_s"]
BINS_V = paquet["bins_v"]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# État de la session
if "compteurs" not in st.session_state:
    st.session_state["compteurs"] = {c: 0 for c in CLASSES}
if "derniere_photo" not in st.session_state:
    st.session_state["derniere_photo"] = ""


# =====================================================
# 5. FONCTIONS (analyse)
# =====================================================
def predire(img_bgr):
    """Retourne le vecteur de probabilités pour une image BGR."""
    img = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)

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
    return model.predict_proba(vecteur)[0]


# =====================================================
# 6. FONCTIONS (affichage)
# =====================================================
def titre_zone(numero, texte):
    st.markdown(
        f'<div class="zone-titre"><span class="zone-num">{numero}</span>{texte}</div>',
        unsafe_allow_html=True,
    )


def afficher_consignes():
    items = "".join(f"<li>{c}</li>" for c in CONSIGNES_PHOTO)
    st.markdown(
        f'<div class="consignes"><div class="consignes-titre">📸 Pour une bonne photo</div>'
        f"<ul>{items}</ul></div>",
        unsafe_allow_html=True,
    )


def afficher_attente():
    st.markdown("""
    <div class="carte attente">
        <div class="attente-emoji">🕒</div>
        <div class="attente-titre">En attente de ta photo</div>
        <div class="etape"><div class="etape-emoji">📸</div><div>
            <div class="etape-titre">1. Photo</div>
            <div class="etape-texte">Prends ou importe une image de ton déchet</div></div></div>
        <div class="etape"><div class="etape-emoji">🔍</div><div>
            <div class="etape-titre">2. Analyse</div>
            <div class="etape-texte">Je regarde la forme et la couleur</div></div></div>
        <div class="etape"><div class="etape-emoji">✅</div><div>
            <div class="etape-titre">3. Résultat</div>
            <div class="etape-texte">Tu sais où le jeter</div></div></div>
    </div>
    """, unsafe_allow_html=True)


def niveau_confiance(confiance):
    if confiance >= 75:
        return "J'en suis presque certain, c'est du :", f"Très sûr · {confiance} %", "#16a34a"
    if confiance >= 50:
        return "Je pense que c'est du :", f"Assez sûr · {confiance} %", "#f59e0b"
    return "Hmm, je ne suis pas sûr. Ma meilleure piste :", f"Peu sûr · {confiance} %", "#dc2626"


def afficher_resultat(proba):
    ordre = np.argsort(proba)[::-1]
    premier = CLASSES[int(ordre[0])]
    second = CLASSES[int(ordre[1])]
    confiance = round(float(proba[ordre[0]]) * 100, 1)
    confiance2 = round(float(proba[ordre[1]]) * 100, 1)
    info = INFOS[premier]
    couleur = info["couleur"]
    phrase, texte_badge, couleur_badge = niveau_confiance(confiance)

    html = f'<div class="carte" style="border-left: 8px solid {couleur};">'
    html += (
        f'<div class="res-haut"><div class="res-emoji">{info["emoji"]}</div><div>'
        f'<div class="res-phrase">{phrase}</div>'
        f'<div class="res-classe" style="color:{couleur};">{info["nom"]}</div>'
        f'<span class="res-badge" style="background:{couleur_badge};">{texte_badge}</span>'
        f"</div></div>"
    )
    if confiance < 50:
        info2 = INFOS[second]
        html += (
            f'<div class="res-bloc res-alerte">🤔 <b>Autre possibilité :</b> '
            f'{info2["emoji"]} {info2["nom"]} ({confiance2} %).<br>'
            f"Essaie une photo plus nette, avec un seul objet sur un fond simple.</div>"
        )
    html += f'<div class="res-bloc">🗂️ <b>Où le jeter :</b> {info["conseil"]}</div>'
    html += f'<div class="res-bloc">💡 <b>Le savais-tu ?</b> {info["astuce"]}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # Détail des probabilités
    st.markdown('<div class="zone-titre" style="margin-top:18px;">📊 Ce que j\'ai hésité à répondre</div>',
                unsafe_allow_html=True)
    barres = '<div class="carte">'
    for k in ordre:
        nom_k = CLASSES[int(k)]
        pct = round(float(proba[k]) * 100, 1)
        largeur = max(pct, 2)
        barres += (
            f'<div class="ligne"><div class="ligne-nom">{INFOS[nom_k]["emoji"]} {INFOS[nom_k]["nom"]}</div>'
            f'<div class="piste"><div class="remplissage" style="width:{largeur}%; '
            f'background:{INFOS[nom_k]["couleur"]};"></div></div>'
            f'<div class="ligne-pct">{pct} %</div></div>'
        )
    barres += "</div>"
    st.markdown(barres, unsafe_allow_html=True)
    return premier


# =====================================================
# 8. EN-TÊTE
# =====================================================
st.markdown(
    '<div class="hero"><p class="hero-titre">♻️ Smart Waste</p>'
    '<p class="hero-texte">Prends ton déchet en photo, je t\'aide à bien le trier.</p></div>',
    unsafe_allow_html=True,
)

onglet_trier, onglet_guide, onglet_session, onglet_apropos = st.tabs(
    ["♻️  Trier un déchet", "📚  Guide du tri", "📈  Ma session", "ℹ️  À propos"]
)

# =====================================================
# 9. ONGLET 1 : TRIER (2 colonnes : photo | résultat)
# =====================================================
with onglet_trier:
    col_gauche, col_droite = st.columns(2, gap="large")

    # ---- Colonne gauche : la photo + les consignes ----
    with col_gauche:
        titre_zone(1, "Ta photo")
        choix = st.radio(
            "Comment veux-tu donner ta photo ?",
            ["📁 Importer une photo", "📸 Prendre une photo"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if choix == "📁 Importer une photo":
            fichier = st.file_uploader("Choisis une image (JPG ou PNG)", type=["jpg", "jpeg", "png"])
        else:
            fichier = st.camera_input("Place ton déchet devant la caméra")

        img = None
        octets = None
        if fichier is not None:
            octets = fichier.getvalue()
            img = cv2.imdecode(np.frombuffer(octets, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                st.error("Oups, je n'arrive pas à lire cette image. Essaie avec une autre photo.")
            elif choix == "📁 Importer une photo":
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        afficher_consignes()

    # ---- Colonne droite : le résultat ----
    with col_droite:
        titre_zone(2, "Résultat")
        if img is None:
            afficher_attente()
        else:
            with st.spinner("Je regarde ton déchet... 🔍"):
                proba = predire(img)
            premier = afficher_resultat(proba)

            # Compter la photo une seule fois (même si la page se recharge)
            signature = hashlib.md5(octets).hexdigest()
            if signature != st.session_state["derniere_photo"]:
                st.session_state["derniere_photo"] = signature
                st.session_state["compteurs"][premier] += 1

# =====================================================
# 10. ONGLET 2 : GUIDE DU TRI
# =====================================================
with onglet_guide:
    st.markdown('<div class="zone-titre">Les 6 catégories que je reconnais</div>', unsafe_allow_html=True)
    cles = list(INFOS.keys())
    for debut in range(0, len(cles), 3):
        colonnes = st.columns(3, gap="medium")
        for col, cle in zip(colonnes, cles[debut:debut + 3]):
            i = INFOS[cle]
            col.markdown(
                f'<div class="carte cat" style="border-top-color:{i["couleur"]};">'
                f'<div class="cat-emoji">{i["emoji"]}</div>'
                f'<div class="cat-nom" style="color:{i["couleur"]};">{i["nom"]}</div>'
                f'<div class="cat-texte">🗂️ {i["conseil"]}</div>'
                f'<div class="cat-astuce">💡 {i["astuce"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.write("")

# =====================================================
# 10 bis. ONGLET : MA SESSION
# =====================================================
with onglet_session:
    st.markdown('<div class="zone-titre">Ma session</div>', unsafe_allow_html=True)
    total = sum(st.session_state["compteurs"].values())
    if total == 0:
        st.markdown('<div class="carte">Aucun déchet trié pour l\'instant. '
                    'Va dans « Trier un déchet » pour commencer.</div>', unsafe_allow_html=True)
    else:
        st.metric("Déchets triés", total)
        barres = '<div class="carte">'
        for c in sorted(CLASSES, key=lambda x: -st.session_state["compteurs"][x]):
            n = st.session_state["compteurs"][c]
            largeur = max(100 * n / total, 2) if n > 0 else 0
            barres += (
                f'<div class="ligne"><div class="ligne-nom">{INFOS[c]["emoji"]} {INFOS[c]["nom"]}</div>'
                f'<div class="piste"><div class="remplissage" style="width:{largeur}%; '
                f'background:{INFOS[c]["couleur"]};"></div></div>'
                f'<div class="ligne-pct">{n}</div></div>'
            )
        barres += "</div>"
        st.markdown(barres, unsafe_allow_html=True)
        if st.button("🔄 Réinitialiser la session"):
            st.session_state["compteurs"] = {c: 0 for c in CLASSES}
            st.session_state["derniere_photo"] = ""
            st.rerun()

# =====================================================
# 11. ONGLET 3 : À PROPOS
# =====================================================
with onglet_apropos:
    st.markdown('<div class="zone-titre">Comment ça marche ?</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="carte">
<b>1. Prétraitement</b> : l'image est redimensionnée, puis le contraste est amélioré (CLAHE).<br>
<b>2. Caractéristiques</b> : la <b>forme</b> est décrite par HOG, la <b>couleur</b> par des histogrammes HSV.<br>
<b>3. Classification</b> : normalisation (scaler), réduction de dimension (PCA), puis classifieur <b>SVM</b>
qui donne une probabilité pour chaque catégorie.
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="zone-titre" style="margin-top:18px;">Limites</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="carte">
Le modèle peut se tromper, surtout avec plusieurs objets, un fond chargé, une photo floue ou sombre.
En cas de doute, vérifie le symbole de recyclage sur l'emballage.
</div>
""", unsafe_allow_html=True)

# =====================================================
# 12. PIED DE PAGE
# =====================================================
st.markdown(
    '<div class="pied">Smart Waste · Modèle HOG + couleur + SVM · '
    "Je peux me tromper, en cas de doute vérifie l'emballage.</div>",
    unsafe_allow_html=True,
)

