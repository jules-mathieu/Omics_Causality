import pandas as pd

# on importe le jeu de données initial, l'index est la colonne des SNPs
df = pd.read_parquet("data/colza_genotype.parquet")
df = df.rename(columns={'Unnamed: 0' : 'SNP'})
df = df.set_index('SNP')


df=df.dropna() # retire les données manquantes


## Objectif ici : récupérer les SNP sur les bons chromosomes pour créer le jeu de données qui servira toute l'étude 
# On récupère les indicateurs issues des noms de SNPs - l'objetcif est de récupérer le chromosome
A = [c.split(".") for c in df.index]
df_chr = pd.DataFrame(A, columns = ("First", "Chr", "Pos"))
df_chr["Chr"].unique()

# On ajoute la colonne chr juste en fonction de l'indexe 
df = df.reset_index().join(df_chr["Chr"])

# On récupère uniquement les chromosomes d'intérêt
df_filtered = df[df["Chr"].isin(['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09',
       'A10','C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])]

# On remet le dataset en forme avec les SNP en index et sns la colonne Chr - df_filtered constitue le jeu de données initiale pour les simu qui suivront
df_export = df_filtered.set_index("SNP")
df.to_parquet('data/colza_genotype_cleaned.parquet', compression='snappy')



