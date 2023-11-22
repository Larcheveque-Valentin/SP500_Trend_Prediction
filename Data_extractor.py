

import yfinance as yf
import pandas as pd
sp500_companies = [
    'AAPL', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK',
    'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'APD',
    'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBY', 'BDX', 'BEN', 'BIIB',
    'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLL', 'BMY', 'BR', 'BRK.B', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB',
    'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CERN', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
    'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COG', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CRM', 'CSCO', 'CSX', 'CTAS',
    'CTLT', 'CTSH', 'CTVA', 'CTXS', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISCA',
    'DISCK', 'DISH', 'DLR', 'DLTR', 'DOV', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX',
    'EIX', 'EL', 'EMN', 'EMR', 'ENPH', 'EOG', 'EQIX', 'EQR', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR',
    'F', 'FANG', 'FAST', 'FB', 'FBHS', 'FCX', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLIR', 'FMC', 'FRT', 'FTNT', 'FTV', 'GD',
    'GE', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GPS', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN',
    'HBI', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM',
    'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INFO', 'INTC', 'INTU', 'IP', 'IPG', 'IPGP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW',
    'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO',
    'KR', 'KSU', 'L', 'LB', 'LDOS', 'LEG', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUMN',
    'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK',
    'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRO', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU',
    'MXIM', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NKE', 'NLOK', 'NLSN', 'NOC', 'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA',
    'NVR', 'NWL', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PAYC', 'PAYX', 'PBCT', 'PCAR',
    'PEAK', 'PEG', 'PENN', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PPG',
    'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PTC', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI',
    'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB',
    'SLG', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAK',
    'TAP', 'TDG', 'TDY', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TIF', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO',
    'TSN', 'TT', 'TTWO', 'TWTR', 'TXN', 'TXT', 'TYL', 'U', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNM', 'UNP', 'UPS',
    'URI', 'USB', 'V', 'VAR', 'VFC', 'VIAC', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA',
    'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WLTW', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WU', 'WY', 'WYNN', 'XEL', 'XLNX', 'XOM',
    'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS'
]

# Vérifier qu'il y a bien 500 entreprises dans la liste
# print(len(sp500_companies))
# symboles_matières_premières = [
#     'GC=F',  # Or
#     'BZ=F',    # Pétrole brut (Brent)
#     'CAD=X',  # Call sur le blé avec échéance 30 septembre 2021 et prix d'exercice de 500.00
#     'CL=F',    # Pétrole brut (WTI)
#     "HG=F",
#     'TFC',     # Charbon (contrat à terme)
#     'HG=F',    # Cuivre (contrat à terme)
#     'EUR=X',  # Dollar américain contre Euro
#     'JPY=X',   # Dollar américain contre Yen japonais
#     'GBP=X',  # Livre sterling contre Dollar américain
#     'ZW=F',    # Blé (contrat à terme)
#     'QM=F',    # Électricité (contrat à terme)
#     "^FCHI",  #CAC 40 
#     "SPY",       #SP500
# ]

# # Exemple d'accès à un symbole
# symbole_or = symboles_matières_premières[0]  # Symbole pour l'or
# print(f"Symbole pour l'or : {symbole_or}")

# # symbole_call_ble = symboles_matières_premières[-1]  # Symbole du call sur le blé
# # print(f"Symbole pour le call sur le blé : {symbole_call_ble}")

# # Créez une liste vide pour stocker les données de chaque entreprise
# donnees_matières = []

# # Bouclez à travers chaque entreprise et extrayez les données
# for matiere in symboles_matières_premières:
#     try:
#         # Utilisez yfinance pour obtenir les données historiques
#         print (matiere)
#         data = yf.download(matiere, period="3d", interval="1m")
#         # Ajoutez les données de l'entreprise à la liste
#         donnees_matières.append(data)
#     except Exception as e:
#         print(f"Erreur lors de l'extraction des données pour {matiere}: {str(e)}")

# # Créez un DataFrame pandas en utilisant la liste de données
# donnees_combinees_matiere = pd.concat(donnees_matières, keys=symboles_matières_premières)

# # Réinitialisez l'index pour avoir des indices numériques
# donnees_combinees_matiere.reset_index(level=0, inplace=True)

# # Renommez la colonne de l'index en 'Symbole'
# donnees_combinees_matiere.rename(columns={'level_0': 'Symbole'}, inplace=True)

# # Enregistrez les données dans un fichier CSV
# # donnees_combinees.to_csv("cours_de_bourse_1_an_en_heures.csv", index=True)
# donnees_combinees_matiere.to_csv("1_day_in_minutes_Mat.csv", index=True)

# # Affichez les données
# print(donnees_combinees_matiere.head())


# # Créez une liste vide pour stocker les données de chaque entrepris
# donnees_entreprises = []
def extract_API(Symboles,period,interval,filename='Stocks.csv',index=False):
# Bouclez à travers chaque entreprise et extrayez les données
    for entreprise in Symboles:
        try:
            # Utilisez yfinance pour obtenir les données historiques
            data = yf.download(entreprise, period=period, interval=interval)
            
            # Ajoutez les données de l'entreprise à la liste
            donnees_entreprises.append(data)
        except Exception as e:
            print(f"Erreur lors de l'extraction des données pour {entreprise}: {str(e)}")

    # Créez un DataFrame pandas en utilisant la liste de données
    donnees_combinees = pd.concat(donnees_entreprises, keys=sp500_companies)

    # Réinitialisez l'index pour avoir des indices numériques
    donnees_combinees.reset_index(level=0, inplace=True)

    # Renommez la colonne de l'index en 'Symbole'
    donnees_combinees.rename(columns={'level_0': 'Symbole'}, inplace=True)
    # Enregistrez les données dans un fichier CSV
    # donnees_combinees.to_csv("7_days_in_min_Stock_Market.csv", index=True)
    donnees_combinees.to_csv(filename, index=True)
    return donnees_combinees
    # Affichez les données
    # print(donnees_combinees.head())

