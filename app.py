import streamlit as st
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm

st.set_page_config(
    page_title="Tarification Assurance GLM - Maram Chebbi",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_models():
    with open('glm_model.pkl', 'rb') as f:
        glm_model = pickle.load(f)
    with open('scaler_glm.pkl', 'rb') as f:
        scaler_glm = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler_cluster.pkl', 'rb') as f:
        scaler_cluster = pickle.load(f)
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open('premium_multipliers.pkl', 'rb') as f:
        premium_multipliers = pickle.load(f)
    with open('metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    return glm_model, scaler_glm, kmeans, scaler_cluster, metadata, premium_multipliers, metrics

try:
    glm_model, scaler_glm, kmeans, scaler_cluster, metadata, premium_multipliers, metrics = load_models()
    models_loaded = True
except:
    models_loaded = False

st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(135deg, #1A367E 0%, #4A8FE7 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        width: 100%;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1A367E;
    }
</style>
""", unsafe_allow_html=True)

st.title("💰 Moteur de Tarification d'Assurance")
st.markdown("### Calcul de Prime avec Modélisation GLM & Segmentation Client")
st.markdown("**Développé par** : Maram Chebbi | ESPRIT & IRA Le Mans")
st.markdown("---")

if not models_loaded:
    st.error("⚠️ Modèles non chargés. Veuillez uploader les fichiers requis.")
    st.stop()

st.sidebar.header("📊 Performance du Modèle")
st.sidebar.metric("R² Score", f"{metrics['test_r2']:.3f}")
st.sidebar.metric("RMSE", f"${metrics['test_rmse']:.2f}")
st.sidebar.metric("MAE", f"${metrics['test_mae']:.2f}")
st.sidebar.metric("Nombre de Segments", metrics['n_segments'])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Modèle GLM")
st.sidebar.write("**Type**: Gamma (Log Link)")
st.sidebar.write(f"**Features**: {metrics['n_features']}")
st.sidebar.write(f"**Dataset**: {metrics['dataset_size']} clients")

st.subheader("📝 Informations du Client")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Âge", min_value=18, max_value=100, value=35, step=1)
    bmi = st.number_input("BMI (Indice de Masse Corporelle)", min_value=10.0, max_value=60.0, value=27.5, step=0.1)

with col2:
    sex = st.selectbox("Sexe", options=['male', 'female'])
    smoker = st.selectbox("Fumeur", options=['no', 'yes'])

with col3:
    children = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=2, step=1)
    region = st.selectbox("Région", options=['northeast', 'northwest', 'southeast', 'southwest'])

if st.button("💰 Calculer la Prime", use_container_width=True):
    with st.spinner("Calcul en cours..."):
        label_encoders = metadata['label_encoders']
        
        sex_encoded = label_encoders['sex'].transform([sex])[0]
        smoker_encoded = label_encoders['smoker'].transform([smoker])[0]
        region_encoded = label_encoders['region'].transform([region])[0]
        
        features_cluster = np.array([[age, bmi, children, smoker_encoded, sex_encoded]])
        features_cluster_scaled = scaler_cluster.transform(features_cluster)
        segment = int(kmeans.predict(features_cluster_scaled)[0])
        features_glm = np.array([[age, bmi, children, sex_encoded, 
                          smoker_encoded, region_encoded, segment]])
        features_glm_scaled = scaler_glm.transform(features_glm)

        # Vérifier si le modèle est statsmodels ou sklearn
        try:
            # Si statsmodels GLM
            features_glm_sm = sm.add_constant(features_glm_scaled, has_constant='add')
            predicted_premium = float(glm_model.predict(features_glm_sm)[0])
        except:
            # Si sklearn GammaRegressor
            predicted_premium = float(glm_model.predict(features_glm_scaled)[0])
        segment_multiplier = float(premium_multipliers[segment])
        base_premium = float(metrics['base_premium'])
        
        st.markdown("---")
        st.subheader("📈 Résultat de la Tarification")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Prime Annuelle", f"${predicted_premium:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Prime Mensuelle", f"${predicted_premium/12:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Segment Client", f"Segment {segment}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            diff_percent = ((predicted_premium - base_premium) / base_premium) * 100
            st.metric("vs Moyenne", f"{diff_percent:+.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Analyse de Risque")
            
            risk_factors = []
            if smoker == 'yes':
                risk_factors.append("🚬 Fumeur (Risque élevé)")
            if bmi > 30:
                risk_factors.append("⚠️ BMI élevé (Obésité)")
            if age > 60:
                risk_factors.append("👴 Âge avancé")
            if children > 3:
                risk_factors.append("👨‍👩‍👧‍👦 Famille nombreuse")
            
            if risk_factors:
                st.warning("**Facteurs de risque identifiés:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("✅ Profil à faible risque")
            
            st.info(f"**Multiplicateur du segment**: {segment_multiplier:.2f}x la prime moyenne")
        
        with col2:
            st.markdown("### 💡 Recommandations")
            
            if smoker == 'yes':
                potential_savings = predicted_premium * 0.3
                st.write(f"🚭 **Arrêt du tabac**: Économie potentielle de ~${potential_savings:.2f}/an")
            
            if bmi > 30:
                st.write("🏃 **Programme de santé**: Réduction possible de la prime")
            
            if segment_multiplier > 1.2:
                st.write("📉 **Prime élevée**: Considérez des options de franchise plus élevée")
            else:
                st.write("✅ **Prime compétitive**: Vous bénéficiez d'un bon tarif")
            
            st.write(f"📊 **Prime de base moyenne**: ${base_premium:.2f}")

st.markdown("---")

with st.expander("📚 À propos du modèle"):
    st.markdown("""
    ### Méthodologie
    
    **Modèle GLM (Generalized Linear Model)**
    - Distribution: Gamma
    - Link Function: Log
    - Optimisé pour les coûts d'assurance
    
    **Segmentation Client**
    - K-Means Clustering (5 segments)
    - Basé sur: Âge, BMI, Statut fumeur, Nombre d'enfants
    
    **Performance**
    - R² Score: {:.3f}
    - Erreur moyenne: ${:.2f}
    - Dataset: {} clients
    """.format(metrics['test_r2'], metrics['test_mae'], metrics['dataset_size']))

st.markdown("---")
st.markdown("### 📞 Contact")

contact_col1, contact_col2, contact_col3, contact_col4 = st.columns(4)

with contact_col1:
    st.markdown("**📧 Email**")
    st.caption("chebbimaram0@gmail.com")

with contact_col2:
    st.markdown("**💼 LinkedIn**")
    st.markdown("[Profil](https://linkedin.com/in/maramchebbi)")

with contact_col3:
    st.markdown("**💻 GitHub**")
    st.markdown("[Repos](https://github.com/maramchebbi)")

with contact_col4:
    st.markdown("**📱 Téléphone**")
    st.caption("+216 53 907 108")
