import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from lightgbm import LGBMClassifier
import shap
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans
plt.style.use('fivethirtyeight')
#sns.set_style('darkgrid')


def main() :

    @st.cache
    def load_data():
        z = ZipFile("../data_trim.zip")
        data = pd.read_csv(z.open('data_trim.csv'), index_col='SK_ID_CURR')

        sample = pd.read_csv('../data_trim_sample.csv', index_col='SK_ID_CURR')

        data_info_client = pd.read_csv('../info_client.csv', index_col='SK_ID_CURR')
        
        description = pd.read_csv("../features_description.csv", 
                                  usecols=['Row', 'Description'], index_col=0, sep=';')

        target = data.iloc[:, -1:]

        return data, sample, data_info_client, target, description

    @st.cache
    def load_model():
        '''loading the trained model'''
        pickle_in = open('../model.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf


    @st.cache(allow_output_mutation=True)
    def load_knn(sample):
        knn = knn_training(sample)
        return knn


    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets


    def identite_client(data_info_client, id):
        data_client = data_info_client[data_info_client.index == int(id)]
        return data_client

    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/365)*-1, 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    @st.cache
    def load_prediction(sample, id, clf):
        X=sample.iloc[:, :-1]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score
    
    @st.cache
    def load_kmeans(sample, id, mdl):
        index = sample[sample.index == int(id)].index.values
        index = index[0]
        data_client = pd.DataFrame(sample.loc[sample.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data], axis=1)
        return df_neighbors.iloc[:,1:].sample(10)

    @st.cache
    def knn_training(sample):
        knn = KMeans(n_clusters=2).fit(sample)
        return knn 

    #Loading data……
    data, sample, data_info_client, target, description = load_data()
    id_client = sample.index.values
    clf = load_model()


    #######################################
    # SIDEBAR
    #######################################

    #Titre
    html_temp = """
    <div style="background-color: DarkGreen	; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard du Crédit score</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Support de décision pour l'accord d'un crédit</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #Sélection du Customer ID
    st.sidebar.header("**Information General**")

    #Charge la boite de sélection
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    #Charge les info général
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)


    ### Charge les info dans le sidebar ###
    #Nombre de prêt dans l'échantillon
    st.sidebar.markdown("<u>Nombre de prêt dans l'échantillon :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    #Revenue moyen
    st.sidebar.markdown("<u>Revenue moyen (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    #Montant du prêt moyen
    st.sidebar.markdown("<u>Montant du prêt moyen (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
    
    #PieChart de la répartition des target 

    st.sidebar.markdown("<u>Proportion de la solvabilité des clients dans l'échantillon :</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['Solvable', 'Non solvable'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)
        

    #######################################
    # Page principal - CONTENUE PRINCIPAL
    #######################################
    #Affiche le Customer ID dans le Sidebar
    st.write("Sélection Customer ID :", chk_id)


    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**Information Clients**")

    if st.checkbox("Montrer information client ?"):

        infos_client = identite_client(data_info_client, chk_id)
        st.write("**Genre :**", infos_client["CODE_GENDER"].values[0])
        st.write("**Age :**" , infos_client["DAYS_BIRTH"].values[0])
        st.write("**Statut familial :**", infos_client["FAMILY_STATUS"].values[0])
        st.write("**Nombre d'enfant :**", infos_client["CNT_CHILDREN"].values[0])

        #Distribution de l'âge
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values), color="green", linestyle='--')
        ax.set(title='Age client', xlabel='Age', ylabel='')
        st.pyplot(fig)
    
        
        st.subheader("**Revenue (USD)**")
        st.write("**Revenue total :** {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Montant crédit :** {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Nombre d'annuités :** {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        st.write("**Montant des crédits à la propriété:** {:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
        
        #Distribution du Revenue
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Revenue client', xlabel='Revenue (USD)', ylabel='')
        st.pyplot(fig)
        
        #Realtion Age / Revenue Total figure interactive 
        data_sk = data_info_client.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH']).round(1)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

        fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relation Age / Revenue Total", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Revenue total", title_font=dict(size=18, family='Verdana'))

        st.plotly_chart(fig)
    
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    #Customer solvability display
    st.header("**Analyse du dossier client**")
    prediction = load_prediction(sample, chk_id, clf)
    st.write("**Probabilité de solvabilité : **{:.0f} %".format(round(float(prediction)*100, 2)))

    st.markdown("<u>Donnée client :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))

    
    #Feature importance / description
    if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):
        shap.initjs()
        X = sample.iloc[:, :-1]
        X = X[X.index == chk_id]
        number = st.slider("Choisissez le nombre de variables…", 0, 20, 5)

        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(load_model())
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)
        
        if st.checkbox("Description des varaibles?") :
            list_features = description.index.to_list()
            feature = st.selectbox('Feature checklist…', list_features)
            st.table(description.loc[description.index == feature][:1])
        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
            
    

    #Similar customer files display
    chk_voisins = st.checkbox("Montrer le dossier des clients similaire?")

    if chk_voisins:
        knn = load_knn(sample)
        st.markdown("<u>Liste des 10 clients les plus proche :</u>", unsafe_allow_html=True)
        st.dataframe(load_kmeans(sample, chk_id, knn))
        st.markdown("<i>Target 1 = Client non Solvable</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        



if __name__ == '__main__':
    main()
