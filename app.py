import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Tri automatique de tickets", page_icon="📩")

MODEL_NAME = "Ahma2re/ticket-classifier-distilbert"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def translate_to_en(text):
    try:
        return GoogleTranslator(source='fr', target='en').translate(text)
    except:
        return text

response_templates = {
    "Hardware": "Merci pour votre signalement ! Notre équipe technique a bien reçu votre demande et revient vers vous rapidement pour planifier un diagnostic ou une réparation. Délai estimé : 24-48h. 🔧",
    "HR Support": "Merci d'avoir contacté les Ressources Humaines ! Votre demande a bien été reçue et sera traitée par notre équipe RH. Vous aurez un retour sous 2 jours ouvrés. 😊",
    "Access": "Votre demande d'accès a bien été enregistrée ! Elle sera validée par l'administrateur système concerné, avec l'accord de votre manager si nécessaire. Comptez jusqu'à 24h de traitement. ✅",
    "Miscellaneous": "Merci pour votre message ! Votre demande a été enregistrée et sera transmise à l'équipe la plus adaptée. On revient vers vous très vite. 📩",
    "Storage": "Merci pour votre demande liée au stockage ! Notre équipe infrastructure a été notifiée et va traiter ça (quota, boîte mail pleine, etc.) sous 24h. 💾",
    "Purchase": "Votre demande d'achat a bien été reçue et transmise à l'équipe achats pour approbation. Vous serez informé dès qu'un devis ou une décision sera disponible. 🛒",
    "Internal Project": "Merci pour votre message concernant ce projet interne ! Le responsable du projet a été notifié et reviendra vers vous rapidement avec les prochaines étapes. 📋",
    "Administrative rights": "Votre demande concernant des droits administratifs/système a bien été reçue. Pour des raisons de sécurité, elle nécessite une validation managériale avant traitement. Retour sous 48h. 🔐",
}

st.title("📩 Tri automatique de tickets support")
st.write("Colle un message/ticket ci-dessous (en français ou en anglais) et l'IA le classe automatiquement.")

text_input = st.text_area("Contenu du ticket :", height=150, placeholder="Ex: Ma boîte mail est pleine, je ne reçois plus de messages...")

if st.button("Analyser le ticket") and text_input.strip():
    text_for_model = translate_to_en(text_input)

    inputs = tokenizer(text_for_model, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    pred_id = torch.argmax(probs).item()
    pred_label = model.config.id2label[pred_id]
    confidence = probs[pred_id].item()

    st.subheader("Résultat")
    col1, col2 = st.columns(2)
    col1.metric("Catégorie prédite", pred_label)
    col2.metric("Confiance", f"{confidence*100:.1f}%")

    with st.expander("Voir le texte traduit envoyé au modèle"):
        st.write(text_for_model)

    st.subheader("Réponse automatique suggérée")
    st.info(response_templates.get(pred_label, "Merci pour votre message ! Votre demande a bien été reçue et sera traitée sous peu. 😊"))

    with st.expander("Voir les scores détaillés par catégorie"):
        for label, prob in sorted(zip(model.config.id2label.values(), probs.tolist()), key=lambda x: -x[1]):
            st.write(f"{label} : {prob*100:.1f}%")
