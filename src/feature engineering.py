import pandas as pd
import gc
from .encoding import one_hot_encoder

def bureau_and_balance(bureau, bb):
    bb, _ = one_hot_encoder(bb, nan_as_category=True)
    bb_agg = bb.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': ['min', 'max', 'size']
    })
    bb_agg.columns = pd.Index(['BB_' + col[0] + '_' + col[1].upper() for col in bb_agg.columns])
    bureau = bureau.join(bb_agg, on='SK_ID_BUREAU', how='left')
    del bb, bb_agg
    gc.collect()
    
    bureau, _ = one_hot_encoder(bureau, nan_as_category=True)
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'AMT_CREDIT_SUM': ['mean'],
        'AMT_CREDIT_SUM_DEBT': ['mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'BB_MONTHS_BALANCE_SIZE': ['mean'],
    })
    bureau_agg.columns = pd.Index(['BURO_' + col[0] + '_' + col[1].upper() for col in bureau_agg.columns])
    return bureau_agg


def pos_cash(pos):
    # Encodage des variables catégorielles de pos avec un encodage one-hot
    pos, _ = one_hot_encoder(pos, nan_as_category=True)
    
    # Agrégation des données de pos par SK_ID_CURR avec quelques calculs importants
    pos_agg = pos.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['max'],   # Dernière activité (mois de balance)
        'SK_DPD': ['mean'],          # Retards moyens de paiements
        'SK_DPD_DEF': ['mean']       # Retards sérieux moyens
    })
    
    # Modification des noms de colonnes après agrégation pour les rendre explicites
    pos_agg.columns = pd.Index(['POS_' + col[0] + '_' + col[1].upper() for col in pos_agg.columns])
    
    # Retour de l'agrégation finalisée
    return pos_agg
    

def credit_card_balance(cc):
    # Encodage des variables catégorielles de cc avec un encodage one-hot
    cc, _ = one_hot_encoder(cc, nan_as_category=True)
    
    # Suppression de la colonne SK_ID_PREV, qui n'est pas nécessaire dans ce contexte
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    
    # Agrégation des données de crédit par SK_ID_CURR avec quelques statistiques
    cc_agg = cc.groupby('SK_ID_CURR').agg({
        'AMT_BALANCE': ['mean', 'max', 'min'],            # Solde moyen, maximum et minimum
        'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'max', 'min'], # Paiement total moyen, maximum et minimum
        'SK_DPD': ['mean', 'max'],                        # Jours de retard moyens et max
    })
    
    # Modification des noms de colonnes après agrégation pour les rendre explicites
    cc_agg.columns = pd.Index(['CC_' + col[0] + '_' + col[1].upper() for col in cc_agg.columns])
    
    # Retour de l'agrégation finalisée
    return cc_agg

def installments_payments(ins):
    # Calcul du pourcentage de paiement effectué par rapport à l'échéance
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    
    # Calcul de la différence entre le montant d'échéance et le paiement effectué
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    # Calcul du retard en jours (clampé à 0 pour éviter les valeurs négatives)
    ins['DPD'] = (ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']).clip(lower=0)
    
    # Agrégation des données de remboursement par SK_ID_CURR avec plusieurs calculs
    ins_agg = ins.groupby('SK_ID_CURR').agg({
        'PAYMENT_PERC': ['mean'],  # Moyenne du pourcentage payé par rapport à l’échéance
        'PAYMENT_DIFF': ['mean', 'max'],  # Différence moyenne et max entre paiement effectué et dû
        'DPD': ['mean', 'max']            # Retard moyen et maximum de paiement
    })
    
    # Modification des noms de colonnes après agrégation pour les rendre explicites
    ins_agg.columns = pd.Index(['INSTAL_' + col[0] + '_' + col[1].upper() for col in ins_agg.columns])
    
    # Retour de l'agrégation finalisée
    return ins_agg

def previous_applications(prev):
    # Encodage des variables catégorielles de prev avec un encodage one-hot
    prev, _ = one_hot_encoder(prev, nan_as_category=True)
    
    # Calcul du pourcentage du crédit demandé par rapport à ce qui a été accordé
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    
    # Agrégation des données de prev par SK_ID_CURR (identifiant client) avec plusieurs calculs
    prev_agg = prev.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT': ['mean', 'max'],        # Montant moyen et maximum accordé
        'AMT_APPLICATION': ['mean', 'max'],   # Montant moyen et maximum demandé
        'APP_CREDIT_PERC': ['mean'],          # Moyenne du pourcentage de crédit demandé/accordé
        'DAYS_DECISION': ['min'],             # Date de la première demande (min)
        'CNT_PAYMENT': ['mean']               # Nombre moyen de paiements prévus
    })
    
    # Modification des noms de colonnes après agrégation pour les rendre explicites
    prev_agg.columns = pd.Index(['PREV_' + col[0] + '_' + col[1].upper() for col in prev_agg.columns])
    
    # Retour de l'agrégation finalisée
    return prev_agg


def add_custom_features(df):
    """
    Création de nouvelles features
    """
    df = df.copy()
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['AGE'] = (-df['DAYS_BIRTH']) // 365
    return df