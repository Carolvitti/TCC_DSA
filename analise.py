# %% Install bibliotecas

!pip install pandas
!pip install numpy
!pip install statsmodels
!pip install linearmodels

# %% Importar bibliotecas

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.linalg as la


from linearmodels import PooledOLS
from linearmodels.panel import PanelOLS
from linearmodels.panel import compare
from linearmodels import RandomEffects


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_goldfeldquandt

from scipy import stats


#%% Tratamento PIB

pib = pd.read_excel("PIB.xlsx")
print(pib.columns.tolist())


pib = pd.melt(
    pib,
    id_vars=['Sigla', 'Codigo', 'Estado'],  # são as colunas que permanecem fixas
    var_name='Ano',                         # nome da nova coluna que conterá os anos
    value_name='PIB'                        # nome da nova coluna com os valores
)

pib['chave'] = pib['Sigla'].astype(str) + "_" + pib['Ano'].astype(str)


#%% Tratamento do de-para

# base de de para, removendo coluna desnecessária
depara = pd.read_csv("depara.csv" , sep=";")
depara = depara.drop(columns=['Código'])


#print(depara.columns.tolist()) #usado para confirmar o nome das colunas

#%% Tratando Abastecimento

abastecimento = pd.read_excel("Abastecimento de Água.xlsx")
abastecimento_tratado = pd.merge(
    abastecimento, 
    depara, 
    left_on='Unidades da Federação',
    right_on='Unidade da Federação',
    how='left'
)

abastecimento_tratado = abastecimento_tratado.drop(columns=['Unidade da Federação'])

print(abastecimento_tratado.columns.tolist()) #usado para confirmar o nome das colunas

abastecimento_tratado = abastecimento_tratado.rename(columns={'Rede pública de abastecimento (%) ':'Rede pública de abastecimento (%)',
                                            'Outras fontes de abastecimento (%) ': 'Outras fontes de abastecimento (%)',
                                            })
colunas_percent_abastecimento = ['Rede pública de abastecimento (%)', 'Outras fontes de abastecimento (%)']

for col_abs in colunas_percent_abastecimento:
    abastecimento_tratado[col_abs] = abastecimento_tratado[col_abs] / 100

abastecimento_tratado['chave'] = abastecimento_tratado['UF'].astype(str) + "_" + abastecimento_tratado['Ano'].astype(str)


#%% Tratando Bens

bens = pd.read_excel("Bens.xlsx")
bens_tratado = pd.merge(
    bens,
    depara,
    left_on='Unidades da Federação',
    right_on='Unidade da Federação',
    how='left')

bens_tratado = bens_tratado.drop(columns=
                                 ['Unnamed: 0',
                                  'Unidade da Federação',
                                  '% população com  Geladeira',
                                  '% população com Motocicleta',
                                  '% população com Microcomputador', 
                                  '% população com Telefone (fixo ou ao menos um celular)',
                                  ])

print(bens_tratado.columns.tolist()) #usado para confirmar o nome das colunas
colunas_percent_bens = ['% população com casa própria', 
                       '% população com Automóvel', 
                       '% população com Acesso à Internet',
                       '% população com Máquina de lavar roupa']

for col_ben in colunas_percent_bens:
    bens_tratado[col_ben] = bens_tratado[col_ben] / 100

bens_tratado['chave'] = bens_tratado['UF'].astype(str) + "_" + bens_tratado['Ano'].astype(str)


#%% Tratamento Educação

educacao = pd.read_excel("Educação.xlsx")
educ_tratado = pd.merge(
    educacao,
    depara,
    left_on='Grandes Regiões e Unidades da Federação',
    right_on='Unidade da Federação',
    how='left'
)

educ_tratado = educ_tratado.drop(columns=['Unidade da Federação',
                                                  'Distribuição percentual das pessoas de 25 anos ou mais de idade por nível de instrução (%) Sem instrução',
                                                  'Distribuição percentual das pessoas de 25 anos ou mais de idade por nível de instrução (%) Ensino fundamental incompleto', 
                                                  'Distribuição percentual das pessoas de 25 anos ou mais de idade por nível de instrução (%) Ensino fundamental completo', 
                                                  'Distribuição percentual das pessoas de 25 anos ou mais de idade por nível de instrução (%)Ensino médio incompleto'])

educ_tratado = educ_tratado.rename(columns={'Grandes Regiões e Unidades da Federação':"Unidades da Federação",
                                                    'Distribuição percentual das pessoas de 25 anos ou mais de idade por nível de instrução (%)Ensino médio completo' : '(%)Ensino médio completo',
                                                    'Distribuição percentual das pessoas de 25 anos ou mais de idade por nível de instrução (%)Ensino superior incompleto' : '(%)Ensino superior incompleto',
                                                    'Distribuição percentual das pessoas de 25 anos ou mais de idade por nível de instrução (%)Ensino superior completo ' : '(%)Ensino superior completo',
                                                    'Taxa de analfabetismo da população de 15 anos ou mais de idade (%)' : 'Taxa de analfabetismo da pop >= 15 anos (%)'})

print(educacao.columns.tolist()) #usado para confirmar o nome das colunas
colunas_percent_educ = ['(%)Ensino médio completo', 
                       '(%)Ensino superior incompleto', 
                       '(%)Ensino superior completo', 
                       'Taxa de analfabetismo da pop >= 15 anos (%)']

for col_educ in colunas_percent_educ:
    educ_tratado[col_educ] = educ_tratado[col_educ] / 100

educ_tratado['chave'] = educ_tratado['UF'].astype(str) + "_" + educ_tratado['Ano'].astype(str)

#%% Tratamento Emprego

emprego = pd.read_excel("Emprego.xlsx")
emprego_tratado = pd.merge(
    emprego,
    depara,
    left_on='Unidades da Federação',
    right_on='Unidade da Federação',
    how='left'
)

emprego_tratado = emprego_tratado.drop(columns=['Unnamed: 0','Unidade da Federação'])
                                                  
print(emprego_tratado.columns.tolist()) #usado para confirmar o nome das colunas
colunas_percent_empr = ['Nível de ocupação (%)', 
                        'Taxa de desocupação (%)', 
                        'Taxa de desocupação Homens (%)', 
                        'Taxa de desocupação Mulheres (%)']

for col_empr in colunas_percent_empr:
    emprego_tratado[col_empr] = emprego_tratado[col_empr] / 100
    
emprego_tratado['chave'] = emprego_tratado['UF'].astype(str) + "_" + emprego_tratado['Ano'].astype(str)

    
#%% Tratar renda

renda = pd.read_excel("Renda.xlsx")
renda_tratado = pd.merge(
    renda,
    depara,
    left_on='Unidades da Federação',
    right_on='Unidade da Federação',
    how='left'
)

renda_tratado = renda_tratado.drop(columns=['Unidade da Federação','Unnamed: 0'])
                                                  
print(renda.columns.tolist()) #usado para confirmar o nome das colunas

renda_tratado['chave'] = renda_tratado['UF'].astype(str) + "_" + renda_tratado['Ano'].astype(str)


#%% Unificar bases

homicidios = pd.read_csv("homicidios-mulheres.csv" , sep=";")
base_geral = homicidios.copy()
base_geral = base_geral[base_geral['período'] >= 2016]
base_geral = base_geral.drop(columns=['cod'])
base_geral = base_geral.rename(columns={'período':'Ano',
                                        'nome':'UF',
                                        'valor':'Qtdd homicidios (Mulheres)'})
base_geral['chave'] = base_geral['UF'].astype(str) + "_" + base_geral['Ano'].astype(str)


# Merge com as outras
print(abastecimento_tratado.columns.tolist()) #usado para confirmar o nome das colunas

base_geral = base_geral.merge(
    abastecimento_tratado
        [['chave',
        'População Total\n(1 000 pessoas)', 
        'Rede pública de abastecimento (%)'
        ]],
    on='chave',
    how='left')
             
print(bens_tratado.columns.tolist()) #usado para confirmar o nome das colunas
                                       
base_geral = base_geral.merge(
    bens_tratado
        [['chave',
        '% população com casa própria', 
        '% população com Automóvel', 
        '% população com Acesso à Internet'
        ]],
    on='chave',
    how='left')                                                    
                                                    
print(educ_tratado.columns.tolist()) #usado para confirmar o nome das colunas
                                       
base_geral = base_geral.merge(
    educ_tratado
        [['chave',
        '(%)Ensino superior completo', 
        'Taxa de analfabetismo da pop >= 15 anos (%)']],
    on='chave',
    how='left')                                                     
        

print(emprego_tratado.columns.tolist()) #usado para confirmar o nome das colunas

base_geral = base_geral.merge(
    emprego_tratado
        [['chave',
        'Nível de ocupação (%)', 
        'Taxa de desocupação Mulheres (%)']],
    on='chave',
    how='left')                                                
                                                    

print(renda_tratado.columns.tolist()) #usado para confirmar o nome das colunas

base_geral = base_geral.merge(
    renda_tratado
        [['chave',
          'Rendimento domiciliar per capita mediano Homem', 
          'Rendimento domiciliar per capita mediano Mulher']],
    on='chave',
    how='left') 

print(pib.columns.tolist()) #usado para confirmar o nome das colunas

base_geral = base_geral.merge(
    pib
        [['chave',
          'PIB']],
    on='chave',
    how='left')

#%% Ultimas modificações - 

print(base_geral.columns.tolist()) #usado para confirmar o nome das colunas
print(base_geral.dtypes)

base_geral['total_populacao'] = base_geral['População Total\n(1 000 pessoas)'] *1000
base_geral['homicidios per capita (mulheres)'] = base_geral['Qtdd homicidios (Mulheres)']/base_geral['População Total\n(1 000 pessoas)']
base_geral['PIB per capita'] = base_geral['PIB']/base_geral['População Total\n(1 000 pessoas)']

print(base_geral.dtypes)

colunas = ['% população com Automóvel',
           '% população com Acesso à Internet',
           '(%)Ensino superior completo',
           'Taxa de analfabetismo da pop >= 15 anos (%)',
           'Nível de ocupação (%)',
           'Taxa de desocupação Mulheres (%)',
           'PIB per capita']

base_geral[colunas] = base_geral[colunas].round(6)


base_geral = base_geral.drop(columns=['Qtdd homicidios (Mulheres)','total_populacao','PIB'])
base_geral_dropna = base_geral.dropna()

#%% Análise colinearidade

#criação de index com id + ano
base_geral_dropna['id'] = base_geral_dropna['UF'].astype(str)
base_geral_dropna['ano'] = base_geral_dropna['Ano'].astype(int)
base_geral_dropna = base_geral_dropna.set_index(['id', 'ano'])

# Determinar variável y e x
y_homicidio= base_geral_dropna['homicidios per capita (mulheres)']
X_basegeral = base_geral_dropna.drop(columns=['UF','chave','Ano','homicidios per capita (mulheres)'])

#criar uma constante
X_basegeral = sm.add_constant(X_basegeral)  # intercepto = 1

#correlação entre as variáveis
base_corr = (X_basegeral.corr())

plt.figure(figsize=(10, 8))
sns.heatmap(base_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlação')
plt.show()

#VIF
X_drop_const = X_basegeral.drop(columns='const')
vif = pd.DataFrame()
vif["variável"] = X_drop_const.columns
vif["VIF"] = [variance_inflation_factor(X_drop_const.values, i) for i in range(X_drop_const.shape[1])]
print(vif)

#%% Remover e testar colinearidade


X_drop_colinearidade = X_basegeral.drop(columns=['% população com Automóvel',
                                                 'Nível de ocupação (%)',
                                                 'Rede pública de abastecimento (%)',
                                                 'Nível de ocupação (%)',
                                                 'Rendimento domiciliar per capita mediano Homem'
                                                 ])

#correlação entre as variáveis
base_corr = (X_drop_colinearidade.corr())

plt.figure(figsize=(10, 8))
sns.heatmap(base_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlação')
plt.show()

#VIF
X_drop_const_2 = X_drop_colinearidade.drop(columns='const')
vif = pd.DataFrame()
vif["variável"] = X_drop_const_2.columns
vif["VIF"] = [variance_inflation_factor(X_drop_const_2.values, i) for i in range(X_drop_const_2.shape[1])]
print(vif)


#%% Modelo Simples PooledOLS

pooled_ols_result = PooledOLS(y_homicidio, X_drop_colinearidade)

#Clusterização por entidade - estimar variância dos coeficientes em modelos (erros robustos).
pooledOLS_res = pooled_ols_result.fit(cov_type='clustered', cluster_entity=True)
fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
residuals_pooled_OLS = pooledOLS_res.resids

#TESTE DE HOMOCEDASTICIDADE

fig, ax = plt.subplots()
ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color = 'slateblue')
ax.axhline(0, color = 'r', ls = '--')
ax.set_xlabel('Valores previstos', fontsize = 15)
ax.set_ylabel('Residuos', fontsize = 15)
ax.set_title('Teste de Homocedasticidade', fontsize = 30)
plt.show()


# WHITE-TEST

white_test_results = het_white(residuals_pooled_OLS, X_drop_colinearidade) #whitetest
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val']
resultado = dict(zip(labels, white_test_results))

print(resultado)

if resultado['F p-val'] < 0.05:
    print("➡️ Há evidência de heterocedasticidade (p-valor < 0.05)")
else:
    print("✅ Homocedasticidade: resíduos têm variância constante (p-valor ≥ 0.05)")

## Teste de Breusch-Pagan

breusch_pagan_test_results = het_breuschpagan(residuals_pooled_OLS, X_drop_colinearidade)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val'] 
print(dict(zip(labels, breusch_pagan_test_results)))

if resultado['F p-val'] < 0.05:
    print("➡️ Há evidência de heterocedasticidade (p-valor < 0.05)")
else:
    print("✅ Homocedasticidade: resíduos têm variância constante (p-valor ≥ 0.05)")


## Teste Goldfeld-Quandt

gq_test = sm.stats.diagnostic.het_goldfeldquandt(y_homicidio, X_drop_colinearidade, drop=0.2)

print('Goldfeld-Quandt test statistic:', gq_test[0])
print('p-value:', gq_test[1])

if gq_test[1] < 0.05:
    print("❌ Rejeita H0: Existe heterocedasticidade.")
else:
    print("✅ Não rejeita H0: Homocedasticidade.")


##Durbin-Watson-Test 

durbin_watson_test_results = durbin_watson(residuals_pooled_OLS) 
print(f'Estatística de Durbin-Watson: {durbin_watson_test_results:.3f}')

if durbin_watson_test_results < 1.5:
    print("❌ Autocorrelação positiva detectada nos resíduos.")
elif durbin_watson_test_results > 2.5:
    print("❌ Autocorrelação negativa detectada nos resíduos.")
else:
    print("✅ Sem autocorrelação significativa nos resíduos.")


print(pooledOLS_res.summary)

# Salvar como imagem

summary_str = str(pooledOLS_res.summary)

fig, ax = plt.subplots(figsize=(12, 6))  # Ajuste o tamanho conforme necessário
ax.axis('off')  # Sem eixos
ax.text(0, 1, summary_str, fontsize=10, fontfamily='monospace', verticalalignment='top')

plt.savefig("pooledOLS_res.png", bbox_inches='tight', dpi=300)
plt.show()

#%% Modelo FE-RE -> Teste

#FE (Fixed Effects) – Efeitos Fixos
#RE (Random Effects) – Efeitos Aleatórios


# Efeitos Aleatórios
model_re = RandomEffects(y_homicidio, X_drop_colinearidade) 

re_res = model_re.fit(cov_type='clustered', cluster_entity=True)
print(re_res.summary)

summary_re = str(re_res.summary)

fig, ax = plt.subplots(figsize=(12, 6))  # Ajuste o tamanho conforme necessário
ax.axis('off')  # Sem eixos
ax.text(0, 1, summary_re, fontsize=10, fontfamily='monospace', verticalalignment='top')

plt.savefig("summary_re.png", bbox_inches='tight', dpi=300)
plt.show()

# Efeitos Fixos
model_fe = PanelOLS(y_homicidio, X_drop_colinearidade, entity_effects = True) 
fe_res = model_fe.fit(cov_type='clustered', cluster_entity=True)
print(fe_res.summary)

summary_fe = str(fe_res.summary)

fig, ax = plt.subplots(figsize=(12, 6))  # Ajuste o tamanho conforme necessário
ax.axis('off')  # Sem eixos
ax.text(0, 1, summary_fe, fontsize=10, fontfamily='monospace', verticalalignment='top')

plt.savefig("summary_fe.png", bbox_inches='tight', dpi=300)
plt.show()



#%% Comparando modelos: PooledOLS, RE e FE

#Teste F (Efeitos fixos vs. pooled OLS)
comparison = compare({'PooledOLS': pooledOLS_res, 'FE': fe_res})
print(comparison)
f_stat = fe_res.f_statistic.stat
f_pval = fe_res.f_statistic.pval

if f_pval < 0.05:
    print("✅ Rejeita H₀ (sem efeitos fixos) → Use Efeitos Fixos (FE).")
else:
    print("ℹ️ Não rejeita H₀ → Pooled OLS pode ser suficiente.")
print(f_pval)

result_comparison_PooledxFE = str(comparison)

fig, ax = plt.subplots(figsize=(12, 6))  # Ajuste o tamanho conforme necessário
ax.axis('off')  # Sem eixos
ax.text(0, 1, result_comparison_PooledxFE, fontsize=10, fontfamily='monospace', verticalalignment='top')

plt.savefig("summary_fe.png", bbox_inches='tight', dpi=300)
plt.show()


#Teste LM de Breusch-Pagan (Efeitos aleatórios vs. pooled OLS)

lm_test = compare({'PooledOLS': pooledOLS_res, 'RE': re_res})
print(lm_test)

#print(dir(lm_test))
pval_lm = lm_test.f_statistic
pvalor = pval_lm.iloc[1, 1]
print(pvalor)

if pvalor < 0.05:
    print("✅ Rejeita H₀ (sem efeitos aleatórios) → Use Efeitos Aleatórios (RE).")
else:
    print("ℹ️ Não rejeita H₀ → Pooled OLS pode ser suficiente.")
    
    
result_comparison_RExPooled = str(lm_test)

fig, ax = plt.subplots(figsize=(12, 6))  # Ajuste o tamanho conforme necessário
ax.axis('off')  # Sem eixos
ax.text(0, 1, result_comparison_RExPooled, fontsize=10, fontfamily='monospace', verticalalignment='top')

plt.savefig("summary_fe_re.png", bbox_inches='tight', dpi=300)
plt.show()

# Teste de Hausman

result_hausman = compare({'FE': fe_res, 'RE': re_res})
print(result_hausman)

# Alinhar os parâmetros
def hausman(fe_res, re_res):
    b = fe_res.params
    B = re_res.params
    v_b = fe_res.cov
    v_B = re_res.cov

    df = b[np.abs(b) < 1e8].size
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))
    pval1 = stats.chi2.sf(chi2, df)

    return chi2, df, pval1

chi2_val, df, pval1 = hausman(fe_res, re_res)
print('qui-quadrado: ' + str(chi2_val))
print('p-Value: ' + str(pval1))

result_comparison_RExFE = str(result_hausman)

fig, ax = plt.subplots(figsize=(12, 6))  # Ajuste o tamanho conforme necessário
ax.axis('off')  # Sem eixos
ax.text(0, 1, result_comparison_RExFE, fontsize=10, fontfamily='monospace', verticalalignment='top')

plt.savefig("summary_fe_re.png", bbox_inches='tight', dpi=300)
plt.show()