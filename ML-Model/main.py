# %% Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split # pylint: disable=unused-import
from sklearn.metrics import confusion_matrix, accuracy_score

# %% Carregando o dataset
# fetch dataset
statlog_german_credit_data = fetch_ucirepo(id=144)

# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets
# %%
