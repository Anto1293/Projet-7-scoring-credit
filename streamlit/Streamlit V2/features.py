# features.py

MENU_TABS = ["Vue client", "Vue globale", "Comparaison", "Simulation"]
MENU_ICONS = ["person", "globe", "bar-chart", "sliders"]
DEFAULT_INDEX = 0

CLIENT_COLUMNS = [
    "AGE", "AMT_INCOME_TOTAL", "AMT_CREDIT", 
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CODE_GENDER"
]

GAUGE_STEPS = [
    {"range": [0, 10], "color": "lightgreen"},
    {"range": [10, 100], "color": "lightcoral"}
]

# features.py

FEATURES = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "AGE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "INCOME_PER_PERSON",
    "ANNUITY_INCOME_PERC",
    "PAYMENT_RATE",
    "CREDIT_TERM",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "YEARS_BUILD_AVG",
    "TOTALAREA_MODE",
    "BURO_DAYS_CREDIT_MEAN",
    "BURO_AMT_CREDIT_SUM_MEAN",
    "BURO_AMT_CREDIT_SUM_OVERDUE_MEAN",
    "BURO_CREDIT_DAY_OVERDUE_MEAN",
    "PREV_AMT_CREDIT_MAX",
    "PREV_AMT_APPLICATION_MEAN",
    "PREV_APP_CREDIT_PERC_MEAN",
    "PREV_DAYS_DECISION_MIN",
    "POS_SK_DPD_MEAN",
    "POS_SK_DPD_DEF_MEAN",
    "INSTAL_PAYMENT_DIFF_MEAN",
    "INSTAL_PAYMENT_DIFF_MAX",
    "INSTAL_DPD_MEAN",
    "CC_AMT_BALANCE_MEAN",
    "CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN",
    "CC_SK_DPD_MAX",
    "DAYS_EMPLOYED_PERC",
    "NAME_CONTRACT_TYPE_Revolving loans",
    "NAME_INCOME_TYPE_Commercial associate",
    "NAME_INCOME_TYPE_Pensioner",
    "NAME_INCOME_TYPE_State servant",
    "NAME_INCOME_TYPE_Unemployed",
    "NAME_INCOME_TYPE_Working",
    "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Lower secondary",
    "NAME_EDUCATION_TYPE_Secondary / secondary special",
    "NAME_FAMILY_STATUS_Married",
    "NAME_FAMILY_STATUS_Single / not married",
    "NAME_FAMILY_STATUS_Widow",
    "NAME_HOUSING_TYPE_House / apartment",
    "NAME_HOUSING_TYPE_Office apartment",
    "NAME_HOUSING_TYPE_Rented apartment",
    "NAME_HOUSING_TYPE_With parents",
    "SECTOR_Industry",
    "SECTOR_Trade",
    "SECTOR_Transport",
    "SECTOR_Business Entity",
    "SECTOR_Government",
    "SECTOR_Security",
    "SECTOR_Services",
    "SECTOR_Construction",
    "SECTOR_Medicine",
    "SECTOR_Police",
    "SECTOR_Other",
    "OCCUPATION_Labor_Work",
    "OCCUPATION_Sales_Services",
    "OCCUPATION_Medical_Staff",
    "OCCUPATION_Security",
    "OCCUPATION_Management_Core",
    "OCCUPATION_Other"
]
