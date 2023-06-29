from os.path import join

import numpy as np
import pandas as pd
from datetime import datetime
from mim.util.logs import get_logger
log = get_logger("In rsvd_plus")


def read_csv(name, **kwargs):
    base_path = "/mnt/air-crypt/air-crypt-esc-trop/thomas/data_copy"
    return pd.read_csv(
        join(base_path, name),
        encoding="ISO-8859-1",
        sep='|',
        **kwargs
    )


def make_befo_data():
    befo_short = read_csv(
        'RS_Befolkning_20201227.csv',
        usecols=[
            'Alias',
            'FödelseårOchMånad',
            'Kön'
        ]
    )
    befo_short = befo_short[befo_short["FödelseårOchMånad"] < 200501]
    befo_short = befo_short[befo_short["FödelseårOchMånad"] >= 195612]
    befo_short["Male"] = pd.get_dummies(befo_short["Kön"])["M"]
    befo_short["Female"] = pd.get_dummies(befo_short["Kön"])["K"]
    befo_short = befo_short.drop(["Kön"], axis=1)
    return befo_short


def make_lab_data(befo_alias_series):
    biokemi = read_csv(
        "Melior_Labanalyssvar_20190101_20220128.csv",
        usecols=[
            'Alias',
            'Analyssvar_ProvtagningDatum',
            'Labanalys_Beskrivning'
        ]
    )
    ecg = pd.read_csv(
        "/mnt/air-crypt/air-crypt-esc-trop/thomas/EKG_2022_11_25/GE_EKG_Persons_20201227.csv",
        sep=";",
        encoding="ISO-8859-1",
        usecols=[
            'Alias',
            'DBS_0_964'
        ]
    )

    immunologen = pd.read_csv(
        "/mnt/air-crypt/air-crypt-esc-trop/thomas/data_copy/"
        "MIKROBIOLOGI_IMM_Analyser_Covers_2019_2022_CoversPop20201227.massaged.utf8.csv",
        sep="|",
        encoding="UTF8",
        usecols=[
            'Alias',
            'Prdate',
            'QtText'
        ]
    )

    immunologen = immunologen[immunologen["Prdate"].notna()]
    immunologen["Analyssvar_ProvtagningDatum"] = immunologen["Prdate"]
    immunologen['Analyssvar_ProvtagningDatum'] = pd.to_datetime(immunologen['Analyssvar_ProvtagningDatum']).round("d")
    immunologen["Labanalys_Beskrivning"] = immunologen["QtText"]
    immunologen = immunologen[["Alias", "Analyssvar_ProvtagningDatum", "Labanalys_Beskrivning"]]
    immu_test_to_keep = [
        "IgA mot vävnadstransglutaminas (anti-tTG)", "S-IgA", "anti-CCP (Citrullin peptid)",
        "Reumatoid faktor (IgM-RF)", "ANA", "Antikroppar mot ENA", "Antikroppar mot dsDNA",
        "Lymfocyt konc", "Antikroppar mot proteinas 3",
        "Antikroppar mot myeloperoxidas (AMPO)", "Antikroppar mot kardiolipin (IgG)", "P-ANCA", "C-ANCA",
        "CD3 (T-celler)", "CD3 (T-celler), antal", "CD19 (B-celler)", "CD19 (B-celler), antal",
        "Konc. av C1q", "Komplementfunktion alternativ väg",
        "Komplementfunktion klassisk väg", "CD4 (T-hjälpceller)", "CD4 (T-hjälpceller), antal",
        "Total IgE", "Konc. av C4 Atellica", "Kalprotektin i faeces", "Konc. av C3 Atellica",
        "CD8 (T-cytotoxceller)", "CD8 (T-cytotoxceller), antal",
        "CD4/CD8 kvot", "Antikroppar mot ENA, SS-A (Ro60)", "Antikroppar mot ENA, Scl-70",
        "Antikroppar mot ENA, Jo-1", "Antikroppar mot ENA, Sm", "Antikroppar mot ENA, RNP",
        "CD16+CD56 (NK-celler)", "CD16+CD56 (NK-celler), antal",
        "Antikroppar mot glatt muskel", "Antikroppar mot mitokondrier", "CD20 (B-celler)",
        "CD20 (B-celler), antal", "Konc. av C4", "S-IgG Atellica", "Konc. av C3", "Konc. av C3dg",
        "S-IgA Atellica", "S-IgM Atellica", "S-IgG", "ASCA-IgG",
        "ASCA-IgA", "S-IgG4 Atellica", "Anti beta2-glykoprotein 1 (IgG)",
        "Antikroppar mot glomerulärt basalmembran (GBM)", "Tryptas", "S-IgG1 Atellica", "S-IgG2 Atellica",
        "S-IgG3 Atellica", "S-IgM", "Antikroppar mot C1q", "S-IgG4",
        "Antikroppar mot ENA, Ro52", "S-IgG3", "S-IgG1", "S-IgG2", "Penicillin V", "HLA-DQB1*02",
        "HLA-DQB1*03:02P (DQ8)", "HLA-DQA1*05", "Penicillin G", "Myositantikroppar Mi-2beta",
        "Myositantikroppar TIF1gamma", "Myositantikroppar SRP",
        "Myositantikroppar NXP2", "Myositantikroppar Mi-2alfa", "Myositantikroppar SAE1",
        "Myositantikroppar Ku", "Myositantikroppar PM-Scl100", "Myositantikroppar PM-Scl75",
        "Myositantikroppar Ro-52", "Myositantikroppar OJ",
        "Myositantikroppar EJ", "Myositantikroppar PL-12", "Myositantikroppar PL-7",
        "Myositantikroppar MDA5", "Myositantikroppar Jo-1", "HLA-DR (B-celler, aktiva T-celler)",
        "HLA-DR/CD3 (aktiva T-celler)", "HLA-DR (B-celler, aktiva T-celler), antal",
        "HLA-DR/CD3 (aktiva T-celler), antal", "Löslig interleukin-2receptor alfa i serum",
        "Yersinia enterocolitica IgA antikroppar", "Yersinia enterocolitica IgG antikroppar",
        "Campylobacter IgA-antikroppar", "Campylobacter IgG-antikroppar",
        "Konc. av properdin", "Cartilage Oligomeric Matrix Protein, COMP", "Salmonella IgG-antikroppar",
        "Salmonella IgM-antikroppar", "Kvant. C-funktion lektinväg", "C3d", "CD2",
        "PR3-ANCA capture teknik", "Antikroppar mot myositantigen",
        "sIL-2-receptor", "TARC/CCL17", "Autoimmun (limbisk) encefalit (ALE)"
    ]

    immunologen = immunologen[immunologen["Labanalys_Beskrivning"].isin(immu_test_to_keep)]
    ecg["Analyssvar_ProvtagningDatum"] = ecg["DBS_0_964"]
    ecg['Analyssvar_ProvtagningDatum'] = pd.to_datetime(ecg['Analyssvar_ProvtagningDatum']).round("d")
    ecg["Labanalys_Beskrivning"] = "ECG"
    ecg = ecg[["Alias", "Analyssvar_ProvtagningDatum", "Labanalys_Beskrivning"]]
    biokemi_value_count = biokemi["Labanalys_Beskrivning"].value_counts().head(200)
    biokemi = biokemi[biokemi["Labanalys_Beskrivning"].isin(biokemi_value_count.index)]
    biokemi["Alias"] = biokemi["Alias"].astype('Int64')
    biokemi['Analyssvar_ProvtagningDatum'] = pd.to_datetime(biokemi['Analyssvar_ProvtagningDatum']).round("d")
    biokemi = pd.concat([biokemi, ecg, immunologen])
    biokemi = biokemi[biokemi["Analyssvar_ProvtagningDatum"] > datetime(year=2020, month=5, day=25)]
    biokemi = biokemi.merge(befo_alias_series, how="inner")

    return biokemi


def make_vaccination_data(befo_alias_series):
    vaccinations = read_csv(
        "20220126_JB/FHM_NVR_Kahn_Skane.csv",
        usecols=[
            'Alias',
            'vaccination_date',
            'vaccine_product',
            'dose_number'
        ]
    )

    vaccinations['vaccination_date'] = pd.to_datetime(vaccinations['vaccination_date'])
    vaccinations = vaccinations.sort_values("vaccination_date").dropna()
    vaccinations = vaccinations.merge(befo_alias_series, how="inner")
    vaccinations["Dose_1"] = (vaccinations['dose_number'] == 1)*1
    vaccinations["Dose_2"] = (vaccinations['dose_number'] == 2)*1
    vaccinations['Dose_more_3'] = (vaccinations['dose_number'] >= 3)*1

    vaccinations = vaccinations[vaccinations.vaccine_product != "COVID-19 Vaccine Janssen"]

    vaccinations["Pfizer"] = (vaccinations['vaccine_product'] == "Comirnaty")*1
    vaccinations["AstraZeneca"] = (vaccinations['vaccine_product'] == "Spikevax")*1
    vaccinations['Moderna'] = (vaccinations['vaccine_product'] == "Vaxzevria")*1

    vaccinations["FödelseårOchMånad"] = pd.to_datetime(vaccinations["FödelseårOchMånad"], format="%Y%m")
    age = vaccinations["vaccination_date"] - vaccinations["FödelseårOchMånad"]
    vaccinations["Age"] = (age.dt.days / 365.25).astype(int)
    vaccinations = vaccinations.drop(["FödelseårOchMånad", "vaccine_product", "dose_number"], axis=1)
    return vaccinations


def make_rsvd_data(befo_alias_series):
    sva_use_cols = [
            'Alias',
            'Diagnos1',
            'Diagnos2',
            'Diagnos3',
            'Diagnos4',
            'Diagnos5',
            'Kva_kod1',
            'Kva_kod2',
            'Kva_kod3',
            'Kva_kod4',
            'Kva_kod5',
            'Kva_kod6',
            'Indatum'
        ]

    ova_use_cols = [
            'Alias',
            'Diagnos1',
            'Diagnos2',
            'Diagnos3',
            'Diagnos4',
            'Diagnos5',
            'Kva_kod1',
            'Kva_kod2',
            'Kva_kod3',
            'Kva_kod4',
            'Kva_kod5',
            'Kva_kod6',
            'Kontaktdatum',
            'Kontakttyp'
        ]

    sva19 = read_csv(
        "rsvd/EA_SVA19_vy_uppf_coverspop20201227.csv",
        usecols=sva_use_cols
    )

    sva20 = read_csv(
        "rsvd/EA_SVA20_vy_uppf_coverspop20201227.csv",
        usecols=sva_use_cols
    )

    sva21 = read_csv(
        "rsvd/EA_SVA21_vy_uppf_coverspop20201227.csv",
        usecols=sva_use_cols
    )

    sva22 = read_csv(
        "rsvd/EA_SVA22_vy_uppf_coverspop20201227.csv",
        usecols=sva_use_cols
    )
    sva = pd.concat([sva19, sva20, sva21, sva22])
    sva['Indatum'] = pd.to_datetime(sva['Indatum'])
    sva = sva.rename(columns={"Indatum": "Kontaktdatum"})

    lak19 = read_csv(
        "rsvd/EA_OVALAK19_vy_uppf_coverspop20201227.csv",
        usecols=ova_use_cols
    )

    lak20 = read_csv(
        "rsvd/EA_OVALAK20_vy_uppf_coverspop20201227.csv",
        usecols=ova_use_cols
    )

    lak21 = read_csv(
        "rsvd/EA_OVALAK21_vy_uppf_coverspop20201227.csv",
        usecols=ova_use_cols
    )

    lak22 = read_csv(
        "rsvd/EA_OVALAK22_vy_uppf_coverspop20201227.csv",
        usecols=ova_use_cols
    )
    lak = pd.concat([lak19, lak20, lak21, lak22])

    ovr19 = read_csv(
        "rsvd/EA_OVAOVR19_vy_uppf_coverspop20201227.csv",
        usecols=ova_use_cols
    )

    ovr20 = read_csv(
        "rsvd/EA_OVAOVR20_vy_uppf_coverspop20201227.csv",
        usecols=ova_use_cols
    )

    ovr21 = read_csv(
        "rsvd/EA_OVAOVR21_vy_uppf_coverspop20201227.csv",
        usecols=ova_use_cols
    )

    ovr22 = read_csv(
        "rsvd/EA_OVAOVR22_vy_uppf_coverspop20201227.csv",
        usecols=ova_use_cols
    )
    ovr = pd.concat([ovr19, ovr20, ovr21, ovr22])
    diag = pd.concat([sva, lak, ovr])
    diag = diag.merge(befo_alias_series, how="inner")
    diag['Kontaktdatum'] = pd.to_datetime(diag['Kontaktdatum'])

    return diag


def get_relevant_data(vaccinations, diag, biokemi, days):
    """
    As the datasets are large, they can be decreased by selecting the data around the vaccination,
    the data that is of interest.

    :param vaccinations: Dataframe
    :param days: Int
    :param diag: Dataframe
    :param biokemi: Dataframe
    :return: vacc_diag_lab: Dataframe
    """
    vaccinations_copy = vaccinations.copy()
    if days > 0:
        for i in range(1, days+1):
            vaccinations_copy["vaccination_date_" + str(i)] = vaccinations_copy["vaccination_date"] \
                                                          + pd.Timedelta(str(i) + " day")
    if days < 0:
        for i in range(1, np.abs(days)+1):
            vaccinations_copy["vaccination_date_" + str(i)] = vaccinations_copy["vaccination_date"] \
                                                          - pd.Timedelta(str(i)+" day")

    vacc_diag_lab = pd.DataFrame()

    for i in range(1, np.abs(days)+1):
        vacc_diag_lab_new = pd.merge(
            vaccinations_copy, biokemi,
            left_on=['Alias', 'vaccination_date_' + str(i)],
            right_on=['Alias', 'Analyssvar_ProvtagningDatum'],
            how="inner"
        )
        vacc_diag_lab = pd.concat([vacc_diag_lab, vacc_diag_lab_new])

    for i in range(1, np.abs(days)+1):
        vacc_diag_lab_new = pd.merge(
            vaccinations_copy, diag,
            left_on=['Alias', 'vaccination_date_' + str(i)],
            right_on=['Alias', 'Kontaktdatum'],
            how="inner")
        vacc_diag_lab = pd.concat([vacc_diag_lab, vacc_diag_lab_new])

    vacc_diag_lab = vacc_diag_lab.drop(columns=["vaccination_date_" + str(i) for i in range(1, 1+np.abs(days))])

    icd_map = {}
    
    icd_codes = pd.read_csv("~/AutoencoderProject/ICDCodes.csv")
    icd_codes["Letter"] = icd_codes['Codes'].str[:1]
    icd_codes["Start"] = icd_codes['Codes'].str[1:3]
    icd_codes["End"] = icd_codes['Codes'].str[5:7]
    icd_codes["Start"] = icd_codes["Start"].astype('int')
    icd_codes["End"] = icd_codes["End"].astype('int')

    def get_icd_range(diagnos):
        if diagnos in icd_map:
            return icd_map[diagnos]

        if type(diagnos) != str:
            return np.nan
        if diagnos == "U070" or diagnos == "U073" or diagnos == "U074" or diagnos == "U075" or diagnos == "U076" or \
                diagnos == "U077" or diagnos == "U078" or diagnos == "U079":
            return np.nan
        letter = diagnos[:1]
        try:
            numbers = int(diagnos[1:3])
        except ValueError:
            return np.nan
        icd_codes_copy = icd_codes[
            (icd_codes["Letter"] == letter) & (icd_codes["Start"] <= numbers) & (icd_codes["End"] >= numbers)]
        if icd_codes_copy.empty:
            return np.nan
        result = icd_codes_copy.Codes.iloc[0]
        icd_map[diagnos] = result
        return result

    def get_icd_range_first_4(diagnos):
        if pd.isna(diagnos):
            return np.nan
        return diagnos[0:3]
    vacc_diag_lab["Diagnos1"] = vacc_diag_lab["Diagnos1"].map(get_icd_range_first_4)
    vacc_diag_lab["Diagnos2"] = vacc_diag_lab["Diagnos2"].map(get_icd_range_first_4)
    vacc_diag_lab["Diagnos3"] = vacc_diag_lab["Diagnos3"].map(get_icd_range_first_4)
    vacc_diag_lab["Diagnos4"] = vacc_diag_lab["Diagnos4"].map(get_icd_range_first_4)
    vacc_diag_lab["Diagnos5"] = vacc_diag_lab["Diagnos5"].map(get_icd_range_first_4)
    return vacc_diag_lab


def make_index(vaccinations):
    return vaccinations[['Alias', 'vaccination_date']]\
        .sort_values(by=['Alias', 'vaccination_date'])\
        .set_index(["Alias", "vaccination_date"])


def make_lab_samples(vacc_diag_lab):
    vacc_diag_lab['temp'] = 1
    vacc_diag_lab = vacc_diag_lab[['Alias', 'Labanalys_Beskrivning', 'vaccination_date', 'temp']]\
        .dropna(subset=['Labanalys_Beskrivning'])\
        .sort_values(by=['Alias', 'vaccination_date'])

    vacc_diag_lab = vacc_diag_lab.reset_index(drop=True)

    lab_samples = vacc_diag_lab.pivot(columns='Labanalys_Beskrivning', values='temp')\
        .join(vacc_diag_lab[['Alias', 'vaccination_date']])\
        .groupby(by=['Alias', 'vaccination_date'])\
        .sum().astype(int)

    return lab_samples


def make_diag_samples(vacc_diag_lab):
    vacc_diag_lab['temp'] = 1

    diagnoser1 = vacc_diag_lab[['Alias', 'Diagnos1', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser1 = diagnoser1.reset_index(drop=True)
    diagnoser1 = diagnoser1.pivot(columns=['Diagnos1'], values='temp').join(
        diagnoser1[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    diagnoser2 = vacc_diag_lab[['Alias', 'Diagnos2', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser2 = diagnoser2.reset_index(drop=True)
    diagnoser2 = diagnoser2.pivot(columns=['Diagnos2'], values='temp').join(
        diagnoser2[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    diagnoser3 = vacc_diag_lab[['Alias', 'Diagnos3', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser3 = diagnoser3.reset_index(drop=True)
    diagnoser3 = diagnoser3.pivot(columns=['Diagnos3'], values='temp').join(
        diagnoser3[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    diagnoser4 = vacc_diag_lab[['Alias', 'Diagnos4', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser4 = diagnoser4.reset_index(drop=True)
    diagnoser4 = diagnoser4.pivot(columns=['Diagnos4'], values='temp').join(
        diagnoser4[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    diagnoser5 = vacc_diag_lab[['Alias', 'Diagnos5', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser5 = diagnoser5.reset_index(drop=True)
    diagnoser5 = diagnoser5.pivot(columns=['Diagnos5'], values='temp').join(
        diagnoser5[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    diagnos_df = pd.concat([diagnoser1, diagnoser2, diagnoser3, diagnoser4, diagnoser5]).groupby(
        by=['Alias', 'vaccination_date']).sum().astype(int)

    # diagnos_df = diagnoser1.groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    if np.nan in diagnos_df.columns:
        diagnos_df = diagnos_df.drop(columns=[np.nan])
    diagnos_codes = diagnos_df.sum()[diagnos_df.sum() > 30].index.tolist()
    diagnos_df = diagnos_df[diagnos_df.columns.intersection(diagnos_codes)]
    return diagnos_df


def make_history_diag_samples(rsvd_data, vaccinations, days):

    def get_icd_letter(diagnos):
        if type(diagnos) != str:
            return np.nan
        letter = diagnos[:1]
        if not letter.isupper():
            return np.nan
        if letter == "A" or letter == "B":
            return "A-B"
        if letter == "C" or letter == "D":
            return "C-D"
        if letter == "S" or letter == "T":
            return "S-T"
        if letter == "V" or letter == "X" or letter == "Y":
            return "V-Y"
        return letter
    diag_copy = rsvd_data.copy()
    diag_copy["Diagnos1"] = diag_copy["Diagnos1"].map(get_icd_letter)
    diag_copy["Diagnos2"] = diag_copy["Diagnos2"].map(get_icd_letter)
    diag_copy["Diagnos3"] = diag_copy["Diagnos3"].map(get_icd_letter)
    diag_copy["Diagnos4"] = diag_copy["Diagnos4"].map(get_icd_letter)
    diag_copy["Diagnos5"] = diag_copy["Diagnos5"].map(get_icd_letter)
    vaccinations_dates = vaccinations[["Alias", "vaccination_date"]]
    history_diag = pd.merge(diag_copy, vaccinations_dates, left_on="Alias", right_on="Alias")
    history_diag = history_diag[history_diag['Kontaktdatum'] <
                                history_diag['vaccination_date'] - pd.Timedelta(str(abs(days))+" days")]
    history_diag = history_diag[history_diag['Kontaktdatum'] >
                                history_diag['vaccination_date'] - pd.Timedelta("730 days")]

    history_diag['temp'] = 1

    diagnoser1 = history_diag[['Alias', 'Diagnos1', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser1 = diagnoser1.reset_index(drop=True)
    diagnoser1 = diagnoser1.pivot(columns=['Diagnos1'], values='temp').join(
        diagnoser1[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    diagnoser2 = history_diag[['Alias', 'Diagnos2', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser2 = diagnoser2.reset_index(drop=True)
    diagnoser2 = diagnoser2.pivot(columns=['Diagnos2'], values='temp').join(
        diagnoser2[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    diagnoser3 = history_diag[['Alias', 'Diagnos3', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser3 = diagnoser3.reset_index(drop=True)
    diagnoser3 = diagnoser3.pivot(columns=['Diagnos3'], values='temp').join(
        diagnoser3[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    diagnoser4 = history_diag[['Alias', 'Diagnos4', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser4 = diagnoser4.reset_index(drop=True)
    diagnoser4 = diagnoser4.pivot(columns=['Diagnos4'], values='temp').join(
        diagnoser4[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    diagnoser5 = history_diag[['Alias', 'Diagnos5', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    diagnoser5 = diagnoser5.reset_index(drop=True)
    diagnoser5 = diagnoser5.pivot(columns=['Diagnos5'], values='temp').join(
        diagnoser5[['Alias', 'vaccination_date']]).groupby(by=['Alias', 'vaccination_date']).sum().astype(int)

    history_diag_df = pd.concat([diagnoser1, diagnoser2, diagnoser3, diagnoser4, diagnoser5]).groupby(
        by=['Alias', 'vaccination_date']).sum().astype(int)
    if np.nan in history_diag_df.columns:
        history_diag_df = history_diag_df.drop(columns=[np.nan])

    return history_diag_df


def make_kva_samples(vacc_diag_lab):
    vacc_diag_lab['temp'] = 1
    kva_codes = ["ZV050", "ZV100", "AQ002", "AV034", "DR016", "DR014", "AV061", "AF021", "AC041", "AF034", "AC032",
                 "AC012", "AC034", "AL001", "DT001", "AF022", "AW999", "AF015", "AG051", "AG053", "AL019",
                 "AC026", "AM041", "AK047", "AG018", "AQ001", "DL006", "AV008", "DR029", "DL005", "AN006", "AF037",
                 "AV032", "AF040", "AQ004", "AL004", "AP032", "AF020", "SI310", "DQ017", "AG029", "AF023", "AG031",
                 "AB040", "DQ019", "AF012", "AQ012", "AC022", "AD003", "DV071", "AC019", "XS100"]

    kva1 = vacc_diag_lab[['Alias', 'Kva_kod1', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    kva1 = kva1.reset_index(drop=True)
    kva_samples1 = kva1.pivot(columns=['Kva_kod1'], values='temp').join(kva1[['Alias', 'vaccination_date']]).groupby(
        by=['Alias', 'vaccination_date']).sum().astype(int)

    kva2 = vacc_diag_lab[['Alias', 'Kva_kod2', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    kva2 = kva2.reset_index(drop=True)
    kva_samples2 = kva2.pivot(columns=['Kva_kod2'], values='temp').join(kva2[['Alias', 'vaccination_date']]).groupby(
        by=['Alias', 'vaccination_date']).sum().astype(int)

    kva3 = vacc_diag_lab[['Alias', 'Kva_kod3', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    kva3 = kva3.reset_index(drop=True)
    kva_samples3 = kva3.pivot(columns=['Kva_kod3'], values='temp').join(kva3[['Alias', 'vaccination_date']]).groupby(
        by=['Alias', 'vaccination_date']).sum().astype(int)

    kva4 = vacc_diag_lab[['Alias', 'Kva_kod4', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    kva4 = kva4.reset_index(drop=True)
    kva_samples4 = kva4.pivot(columns=['Kva_kod4'], values='temp').join(kva4[['Alias', 'vaccination_date']]).groupby(
        by=['Alias', 'vaccination_date']).sum().astype(int)

    kva5 = vacc_diag_lab[['Alias', 'Kva_kod5', 'vaccination_date', 'temp', 'Kontaktdatum']].dropna(
        subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    kva5 = kva5.reset_index(drop=True)
    kva_samples5 = kva5.pivot(columns=['Kva_kod5'], values='temp').join(kva4[['Alias', 'vaccination_date']]).groupby(
        by=['Alias', 'vaccination_date']).sum().astype(int)

    kva_df = pd.concat([kva_samples1, kva_samples2, kva_samples3, kva_samples4, kva_samples5]).groupby(
        by=['Alias', 'vaccination_date']).sum().astype(int)
    kva_df = kva_df[kva_df.columns.intersection(kva_codes)]
    return kva_df


def make_contact_samples(vacc_diag_lab):
    vacc_diag_lab['temp'] = 1
    contact = vacc_diag_lab[['Alias', 'Kontakttyp', 'vaccination_date', 'temp', 'Kontaktdatum']]\
        .dropna(subset=['Kontaktdatum']).sort_values(by=['Alias', 'vaccination_date'])
    contact = contact.reset_index(drop=True)
    contact_samples = contact.pivot(columns=['Kontakttyp'], values='temp').join(contact[['Alias', 'vaccination_date']])\
        .groupby(by=['Alias', 'vaccination_date']).sum().astype(int)
    if np.nan in contact_samples.columns:
        contact_samples = contact_samples.drop(columns=[np.nan])
    return contact_samples


def make_med_data(index, blood_samples, diag_samples, kva_samples, contact_samples, make_history_diag_samples):
    result = index.join([blood_samples, diag_samples, kva_samples, contact_samples, make_history_diag_samples])\
        .fillna(0).astype(int)
    return result


def merge_data(vaccination_data, med_data):
    med_vacc = pd.merge(vaccination_data, med_data, left_on=["Alias", "vaccination_date"],
                        right_on=["Alias", "vaccination_date"])
    return med_vacc


def make_data(days):
    log.debug("Starting to make dataset for " + str(days) + " days after/before vaccination.")
    log.debug("Demografic data extraction...")
    befo = make_befo_data()
    log.debug("Vaccination data extraction...")
    vaccination_data = make_vaccination_data(befo)
    log.debug("RSVD data extraction...")
    rsvd_data = make_rsvd_data(befo["Alias"])
    log.debug("Laboratory data extraction...")
    lab_data = make_lab_data(befo["Alias"])
    log.debug("Reducing data...")
    relevant_data = get_relevant_data(vaccination_data, rsvd_data, lab_data, days=days)
    log.debug("Pivoting data...")
    med_data = make_med_data(make_index(vaccination_data), make_lab_samples(relevant_data),
                             make_diag_samples(relevant_data), make_kva_samples(relevant_data),
                             make_contact_samples(relevant_data),
                             make_history_diag_samples(rsvd_data, vaccination_data, days=days))
    log.debug("Merging data...")
    out_data = merge_data(vaccination_data, med_data)
    return out_data
