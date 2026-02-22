import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA, NMF
import polars as pl

# Configuration de la page
st.set_page_config(page_title="Omics Data Simulator", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_parquet("data/colza_genotype_cleaned.parquet")
    df = df.set_index('SNP')
    df_treated = df.drop(columns="Chr").T
    return df, df_treated

df_genotype_raw, df_genotype = load_data()

def simulate_omics_from_real_snp(snp_series, ate_2, noise_lv, noise_metabo, n_total_metabolites, n_signal_metabolites):

    # On ne garde que 0 et 2 - discutable de le faire maintenant, 
    # on aurait pu les supprimer dès le début dans data_cleaning par exemple
    mask = snp_series.isin([0, 2])
    snp = snp_series[mask].values
    n_individuals = len(snp)
    
    # Simulation de la VL
    epsilon_lv = np.random.normal(0, noise_lv, size=n_individuals)
    VL = np.zeros(n_individuals)
    VL[snp == 0] = 0 + epsilon_lv[snp == 0]
    VL[snp == 2] = ate_2 + epsilon_lv[snp == 2]

    # Génération des données métaboliques (Haute Dimension) - colonne par colonne
    metabolites_data = np.zeros((n_individuals, n_total_metabolites))
    for i in range(n_signal_metabolites):
        metabolites_data[:, i] = VL + np.random.normal(0, noise_metabo, size=n_individuals)
    for i in range(n_signal_metabolites, n_total_metabolites):
        metabolites_data[:, i] = np.random.normal(0, 1, size=n_individuals)
        
    columns = [f"Metabolite_{i+1}" for i in range(n_total_metabolites)]
    df = pd.DataFrame(metabolites_data, columns=columns)
    df.insert(0, "SNP", snp)
    df.insert(1, "VL_True", VL)
    df["SNP_cat"] = df["SNP"].astype(str)
    
    return df

st.sidebar.title("Configuration")
alpha = st.sidebar.number_input("Occurrences min (0 et 2)", value=40)

# Filtrage des SNPs valides - on parcourt les colonnes et vérifie si elles respectent le critère
valid_snps = []
for col in df_genotype.columns:
    counts = df_genotype[col].value_counts()
    if counts.get(0, 0) >= alpha and counts.get(2, 0) >= alpha:
        valid_snps.append(col)

if not valid_snps:
    st.error(f"Aucun SNP ne possède au moins {alpha} occurrences de 0 et de 2.")
    st.stop()

page = st.sidebar.radio("Navigation", ["Exploration", "Validation all genome", "Analyse de puissance", "Analyse de Sensibilité"])

if page == "Exploration":
    st.title("Exploration des SNPs (0 vs 2)")
    selected_snp_name = st.selectbox("Sélectionner un SNP", options=valid_snps)
    n_total = st.number_input("Total métabolites", value=50)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        ate_2 = st.slider("ATE (2 vs 0)", 0.0, 3.0, 0.8)
        noise_lv = st.slider("Bruit VL", 0.01, 1.0, 0.2)
    with col_p2:
        n_signal = st.slider("Métabolites avec signal", 1, 50, 15)
        noise_metabo = st.slider("Bruit technique omique", 0.01, 1.0, 0.2)
    
    if st.button("Générer"):
        df_sim = simulate_omics_from_real_snp(df_genotype[np.random.choice(valid_snps)], ate_2, noise_lv, noise_metabo, n_total, n_signal)
        st.session_state['data_exp'] = df_sim

    if 'data_exp' in st.session_state:
        data = st.session_state['data_exp']

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(data, x="SNP_cat", y="VL_True", color="SNP_cat", title = "Distribution de la VL selon les valeurs du SNP", labels={
                    "SNP_cat": "SNP",
                    "VL_True": "VL"})
            fig.update_layout(
            # Taille du titre principal
            title_font_size=18,
            # Taille de la police globale (pour la légende, etc.)
            font=dict(size=18),
            legend_font_size=16, 
            # Taille des titres des axes (X et Y)
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            # Taille des graduations (les chiffres sur les axes)
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16
        )
            st.plotly_chart(fig)

        with col2:
            # deuxième colonne pas si utilise que ça mais le plot est plus jolie si il y a une colonne à droite
            st.markdown("""
            > En utilisant l'opérateur $do$-calcul de Judea Pearl, l'ATE mesure l'impact d'une intervention directe sur le génome. 
            > Ici, l'effet causal est estimé par la différence des espérances entre les deux groupes de génotypes homozygotes.
            """)

            st.latex(r'''
                ATE = \mathbb{E}(VL \mid do(SNP = 2)) - \mathbb{E}(VL \mid do(SNP = 0))
                ''')

            st.latex(r'''
                \widehat{ATE} = \mathbb{E}(VL \mid SNP = 2) - \mathbb{E}(VL \mid SNP = 0)
                ''')
            
        st.subheader("Aperçu du Dataset")
        st.dataframe(data.head(20))
        
elif page == "Validation all genome":

    st.title("Exploration des SNPs (0 vs 2)")
    selected_snp_name = st.selectbox("Sélectionner un SNP", options=valid_snps)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        ate_2 = st.slider("ATE (2 vs 0)", 0.0, 3.0, 0.8)
    with col_p2:
        noise_lv = st.slider("Bruit VL", 0.01, 1.0, 0.2)

    if st.button("Manhattan Plot like graph"):
        # Pour générer le manhattan plot : :
        # 1. On simule une VL et les metabos associés
        # 2. On récupère dans le jeu global (SNP * Indiv) les indiv tel que le SNP d'intérêt respecte la contrainte sur les 0 et 2
        # Ca va permettre de récupérer le vecteur des individus associés au SNP d'intérêt  

        # pour le jeu VL simulée :
        # On ajoute juste la colonne des individus

        # pour le jeu genotypes raw :
        # On filtre le jeu global + on ajoute le nom des chromosomes -> On a doncn_snp lignes et n_individus + 1 colonnes 
        # On convertit en format long cad : 4 colonnes = SNP, Chr, value (genotype) et variable (individus)

        # On peut maintenant ajouter la VL simulée à ce df_long selon les individus 
        # On se retrouve avec les 3 colonnes d'intérêt : SNP, Chr, Genotype - on peut calculer l'ATE avec un piepline polars, dit "lazy"
        # cad tout en une demande 
        
        # Simulation de la VL (df_genotype a les SNPs en colonnes, donc snp_name est valide ici)
        df_sim = simulate_omics_from_real_snp(df_genotype[selected_snp_name], ate_2, noise_lv, 0.01, 2, 1)
        
        # Filtrage des individus (on ne garde que 0 et 2 pour le SNP de référence) - on revient aux questionnements 
        # du début.
        # On récupère la liste des individus (index de df_genotype) qui valident le masque
        mask_indiv = df_genotype[selected_snp_name].isin([0, 2])
        selected_individuals = df_genotype.index[mask_indiv].tolist()
        
        # On filtre df_genotype_raw (SNP x Indiv) pour garder ces colonnes + la colonne Chr
        # On ajoute "Chr" car besoin plus tard dans le group_by
        df_all = df_genotype_raw[selected_individuals + ["Chr"]]
        df_all = df_all.reset_index() # Pour remettre le nom du SNP dans une colonne "SNP"
        
        # C'est un très bon moment pour mettre en oeuvre la nouvelle librairie de gestion de données 
        # qui est codée en Rust ! Polars
        df1 = pl.from_pandas(df_all) # Format: SNP | Chr | Indiv1 | Indiv2 ...
        df2 = pl.from_pandas(df_sim) # Format: SNP | VL_True | Metabolites...
        
        # On récupère les noms des individus présents après filtrage
        individus_noms = [col for col in df1.columns if col not in ["SNP", "Chr"]]
        
        # On ajoute la colonne "Individu" à df2 pour permettre la jointure 
        df2 = df2.with_columns(pl.Series(name="Individu", values=individus_noms))
        
        # Transformation format Long 
        df_long = df1.unpivot(
            index=["SNP", "Chr"],
            on=individus_noms,
            variable_name="Individu",
            value_name="Genotype_Value"
        )
        
        # Jointure pour récupérer la VL simulée associée à chaque individu
        df_combined = df_long.join(df2.select(["Individu", "VL_True"]), on="Individu", how="left")
        
        ALPHA = alpha
        resultat = (
            df_combined
            .with_columns(pl.col("Genotype_Value").cast(pl.Int32).cast(pl.String))
            .group_by(["SNP", "Chr", "Genotype_Value"])
            .agg([
                pl.col("VL_True").mean().alias("mean_VL"),
                pl.col("VL_True").count().alias("count_n")
            ])
            .pivot(
                values=["mean_VL", "count_n"],
                index=["SNP", "Chr"],
                on="Genotype_Value"
            )
            .filter(
                (pl.col("count_n_0") >= ALPHA) & 
                (pl.col("count_n_2") >= ALPHA)
            )
            .with_columns(
                (pl.col("mean_VL_2") - pl.col("mean_VL_0")).alias("ATE")
            )
            .select(["SNP", "Chr", "ATE"])
        )
        
        # Tri par Chromosome pour l'affichage
        df_ate = resultat.sort("Chr")
        
        # Affichage
        fig = px.scatter(
            df_ate.to_pandas(), # Conversion pandas pour Plotly 
            x="SNP",
            y="ATE",
            color="Chr",
            title=f"ATE pour chaque SNP (SNP de simulation : {selected_snp_name})",
            height=500
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(font=dict(size=18))
        
        st.info(f"SNP utilisé pour la simulation : **{selected_snp_name}**")
        st.plotly_chart(fig)

elif page == "Analyse de puissance":
    st.title("Réduction de Dimension : PCA vs NMF")
    st.write("On tente de reconstruire la VL à partir des métabolites pour tester l'association avec le SNP.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        n_simu = st.number_input("Nombre de répétitions", value=50)
        ate_2 = st.slider("ATE (2 vs 0)", 0.0, 3.0, 0.3)
        noise_lv = st.slider("Bruit VL", 0.01, 1.0, 0.3)
    with col_p2:
        n_total = st.number_input("Total métabolites", value=50)
        n_signal = st.slider("Métabolites avec signal", 1, 50, 15)
        noise_metabo = st.slider("Bruit technique omique", 0.01, 1.0, 0.2)

    if st.button("Comparer les méthodes"):
        results = {"Référence": [], "PCA": [], "NMF": []}
        progress_bar = st.progress(0)

        for i in range(n_simu):
            # Simulation
            df_sim = simulate_omics_from_real_snp(df_genotype[np.random.choice(valid_snps)], ate_2, noise_lv, noise_metabo, n_total, n_signal)
            
            # Préparation des données (on isole les métabolites)
            X = df_sim.drop(columns=["SNP", "VL_True", "SNP_cat"])
            
            # PCA 
            pca = PCA(n_components=1)
            vl_pca = pca.fit_transform(X).flatten()
            
            # NMF 
            X_pos = X - X.min() + 0.01 
            nmf = NMF(n_components=1, init='nndsvd', max_iter=500)
            vl_nmf = nmf.fit_transform(X_pos).flatten()
            # Tests de Student (p < 0.05)
            def check_sig(val_series, snp_series):
                # Avec la librairie scipy, on aurait pu utiliser pingouins ? 
                _, p = ttest_ind(val_series[snp_series == 0], val_series[snp_series == 2])
                return p < 0.05

            results["Référence"].append(check_sig(df_sim["VL_True"], df_sim["SNP"]))
            results["PCA"].append(check_sig(vl_pca, df_sim["SNP"]))
            results["NMF"].append(check_sig(vl_nmf, df_sim["SNP"]))
            
            progress_bar.progress((i + 1) / n_simu)

        # Calcul des scores
        final_scores = {k: np.mean(v) * 100 for k, v in results.items()}
        df_res = pd.DataFrame(list(final_scores.items()), columns=['Méthode', 'Puissance (%)'])

        fig = px.bar(df_res, x='Méthode', y='Puissance (%)', color='Méthode',
                        title="Comparaison des méthodes de réduction de dimension pour la détection de l'ATE",
                        text_auto='.1f', range_y=[0, 105])
        
        # Ensuite le pipeline plotly pour la taille des légendes 
        fig.update_layout(
            # Taille du titre principal
        title_font_size=18,
        
            # Taille de la police globale 
        font=dict(size=18),
        legend_font_size=16, 
        
            # Taille des titres des axes (X et Y)
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        
            # Taille des graduations (les chiffres sur les axes)
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16
    )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Analyse de Sensibilité":
    st.title("Analyse de Sensibilité (Courbe de Puissance)")
    st.write("Observez comment la puissance de détection évolue en faisant varier un paramètre spécifique.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        param_to_vary = st.selectbox("Paramètre à faire varier", 
                                    options=["Bruit VL", "ATE (2 vs 0)", "Bruit technique omique", "Métabolites avec signal"])
        n_simu = st.number_input("Nombre de répétitions par point", value=30)
        n_points = st.slider("Nombre de points sur la courbe", 5, 15, 8)

    with col_s2:
        # Le retour de tout les paramètres
        fix_ate = st.slider("Fixer ATE", 0.0, 3.0, 0.3) if param_to_vary != "ATE (2 vs 0)" else None
        fix_noise_lv = st.slider("Fixer Bruit VL", 0.01, 1.0, 0.2) if param_to_vary != "Bruit VL" else None
        fix_noise_metabo = st.slider("Fixer Bruit technique", 0.01, 1.0, 0.2) if param_to_vary != "Bruit technique omique" else None
        fix_n_signal = st.slider("Fixer Métabolites signal", 1, 50, 15) if param_to_vary != "Métabolites avec signal" else None
        n_total = st.number_input("Total métabolites (fixe)", value=50)

    if st.button("Lancer l'analyse"):
        # Définition de la plage de variation
        if param_to_vary == "Bruit VL":
            range_values = np.linspace(0.01, 1.0, n_points)
        elif param_to_vary == "ATE (2 vs 0)":
            range_values = np.linspace(0.1, 3.0, n_points)
        elif param_to_vary == "Bruit technique omique":
            range_values = np.linspace(0.01, 1.0, n_points)
        else: # Métabolites avec signal
            range_values = np.linspace(1, n_total, n_points).astype(int)

        all_results = []
        progress_bar = st.progress(0)

        for idx, val in enumerate(range_values):
            # Mise à jour des paramètres pour cette itération
            p_ate = val if param_to_vary == "ATE (2 vs 0)" else fix_ate
            p_noise_lv = val if param_to_vary == "Bruit VL" else fix_noise_lv
            p_noise_met = val if param_to_vary == "Bruit technique omique" else fix_noise_metabo
            p_n_sig = int(val) if param_to_vary == "Métabolites avec signal" else fix_n_signal

            current_scores = {"Référence": 0, "PCA": 0, "NMF": 0}
            
            for _ in range(n_simu):
                df_sim = simulate_omics_from_real_snp(df_genotype[np.random.choice(valid_snps)], p_ate, p_noise_lv, p_noise_met, n_total, p_n_sig)
                X = df_sim.drop(columns=["SNP", "VL_True", "SNP_cat"])
                
                # Réductions
                pca_v = PCA(n_components=1).fit_transform(X).flatten()
                nmf_v = NMF(n_components=1, init='nndsvd', max_iter=500).fit_transform(X - X.min() + 0.01).flatten()

                def is_sig(val_s, snp_s):
                    _, p = ttest_ind(val_s[snp_s == 0], val_s[snp_s == 2])
                    return p < 0.05

                if is_sig(df_sim["VL_True"], df_sim["SNP"]): current_scores["Référence"] += 1
                if is_sig(pca_v, df_sim["SNP"]): current_scores["PCA"] += 1
                if is_sig(nmf_v, df_sim["SNP"]): current_scores["NMF"] += 1

            # Stockage des moyennes pour ce point
            for methode, count in current_scores.items():
                all_results.append({
                    "Paramètre": val,
                    "Puissance (%)": (count / n_simu) * 100,
                    "Méthode": methode
                })
            
            progress_bar.progress((idx + 1) / n_points)

        df_plot = pd.DataFrame(all_results)
        
        # Graph
        fig = px.line(df_plot, x="Paramètre", y="Puissance (%)", color="Méthode", markers=True,
                     title=f"Impact de la variation de : {param_to_vary}",
                     labels={"Paramètre": param_to_vary})
        
        # pipeline pour les légende 
        fig.update_layout(yaxis_range=[-5, 105])
        fig.update_layout(
        # Taille du titre principal
        title_font_size=18,
        
        # Taille de la police globale
        font=dict(size=18),

        # Taille des légendes d'indicateurss
        legend_font_size=16, 

        # Taille des titres des axes (X et Y)
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        
        # Taille des graduations (les chiffres sur les axes)
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16
    )

        st.plotly_chart(fig, use_container_width=True)
        
        st.success("Analyse terminée.")
    

# Critiques sur le code : des redondances, comme la selection des paramètres qui reviennent à chaque fois ou le filtrage des 0 et 2 à faire au début 
# Dans la partie analyse de puissance, la NMF n'a pas assez d'iterations pour réellment converger 
# Dans cette même partie ce warning : 2026-02-13 09:54:54.985 Please replace `use_container_width` with `width`.
