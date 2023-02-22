from os.path import join


import pandas as pd
from datetime import datetime


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
    biokemi['Analyssvar_ProvtagningDatum'] = pd.to_datetime(biokemi['Analyssvar_ProvtagningDatum'])
    biokemi = pd.concat([biokemi, ecg, immunologen])
    biokemi = biokemi[biokemi["Analyssvar_ProvtagningDatum"] > datetime(year=2020, month=5, day=25)]
    biokemi = biokemi.merge(befo_alias_series, how="inner")

    return biokemi


def make_vaccination_data(befo_alias_series):
    vaccineringar = read_csv(
        "20220126_JB/FHM_NVR_Kahn_Skane.csv",
        usecols=[
            'Alias',
            'vaccination_date',
            'vaccine_product',
            'dose_number'
        ]
    )

    vaccineringar['vaccination_date'] = pd.to_datetime(vaccineringar['vaccination_date'])
    vaccineringar = vaccineringar.sort_values("vaccination_date").dropna()
    vaccineringar = vaccineringar.merge(befo_alias_series, how="inner")

    return vaccineringar


def make_rsvd_data():
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

    return diag


def add_dates_to_vaccinations(vaccineringar, days):
    if days == 0:
        return vaccineringar
    if days > 0:
        for i in range(days):
            vaccineringar["vaccination_date_" + str(i)] = vaccineringar["vaccination_date"] \
                                                          + pd.Timedelta(str(i) + " day")
        return vaccineringar
    if days < 0:
        for i in range(days):
            vaccineringar["vaccination_date_" + str(i)] = vaccineringar["vaccination_date"] \
                                                          - pd.Timedelta(str(i)+" day")
        return vaccineringar
