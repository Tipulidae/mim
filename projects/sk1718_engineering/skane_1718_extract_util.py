# -*- coding: utf-8 -*-
from datetime import timedelta
from tqdm import tqdm

import numpy as np
import pandas as pd

BLOOD_SAMPLE_COL_MAP = {
    "aB-Laktat": "blood-aB-Laktat",
    "B-Calciumjon": "blood-B-Calciumjon",
    "B-CRP (PNA)": "blood-B-CRP-PNA",
    "B-Hb (PNA)": "blood-B-Hb-PNA",
    "B-Hemoglobin (Hb)": "blood-B-Hb",
    "B-Leukocyter": "blood-B-Leukocyte",
    "B-Trombocyter": "blood-B-Platelet",
    "P-Calcium": "blood-P-Calcium",
    "P-CRP": "blood-P-CRP",
    "P-D-dimer": "blood-P-D-Dimer",
    "P-Glukos": "blood-P-Glukos",
    "P-Kalium": "blood-P-Kalium",
    "P-Kreatinin (enz)": "blood-P-Krea",
    "P-Laktat": "blood-P-Laktat",
    "P-Natrium": "blood-P-Natrium",
    "P-Troponin T": "blood-P-TnT",
    "P-NT-proBNP": "blood-P-NT-proBNP",
    "S-Calciumjon (pH7.4)": "blood-S-Calciumjon7.4"
}
BROKEN_BLOOD_SAMPLE_VALUES = [
    "FELTA", "HEMOL", "KOAG", "KOAGL", "KOMM", "OFYLL", "OTILL", "SAKNA"
]
SOS_SV_OP_COLS = ["OP"] + [f"OPD{i}" for i in range(1, 31)]
SOS_DIAG_COLS = ["hdia"] + [f"DIA{i}" for i in range(1, 31)]
DORS_ICD_COLS = ["ULORSAK"] + [f"MORSAK{i}" for i in range(1, 49)]
SOS_KVA_TABLE = pd.read_csv("/mnt/air-crypt/air-crypt-raw/andersb/resources/"
                            "sos_patientregister_kva/"
                            "kva-inkl-beskrivningstexter-2021-rev2020-12-23"
                            ".KVÅ-alla.csv").set_index("KOD")


def extract_previous_diagnoses_sos_ov(sos_df, liggar_df, icd_prefixes):
    sos_df = sos_df[sos_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, lopnr) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]].iterrows()
    ):
        pat_oppen = sos_df[(sos_df["LopNr"] == lopnr)]
        ov = pat_oppen[
            (pat_oppen["INDATUM"] < index_date) &
            (pat_oppen["INDATUM"] > index_date - timedelta(days=5 * 365))
            ]
        prev_icds = set(x for lst in ov[SOS_DIAG_COLS].values.tolist()
                        for x in lst if not pd.isna(x))
        new_row = [1 if any(icd.startswith(p) for icd in prev_icds) else 0
                   for p in icd_prefixes]
        out.append(new_row)

    colnames = [f"prev5y-{x}-OV" for x in icd_prefixes]
    df = pd.DataFrame(out, columns=colnames, index=liggar_df["Alias"])
    return df


def extract_previous_diagnoses_sos_sv(sos_df, liggar_df, icd_prefixes):
    sos_df = sos_df[sos_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, lopnr) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]
            ].iterrows()
    ):
        pat_sluten = sos_df[(sos_df["LopNr"] == lopnr)]

        ss = pat_sluten[(pat_sluten["UTDATUM"] < index_date) &
                        (pat_sluten["UTDATUM"] > index_date - timedelta(
                            days=5 * 365))]
        prev_icds = set(x
                        for lst in ss[SOS_DIAG_COLS].values.tolist()
                        for x in lst if not pd.isna(x)
                        )
        new_row = [1 if any(icd.startswith(p) for icd in prev_icds) else 0
                   for p in icd_prefixes]
        out.append(new_row)

    colnames = [f"prev5y-{x}-SV" for x in icd_prefixes]
    df = pd.DataFrame(out, columns=colnames, index=liggar_df["Alias"])
    return df


def extract_outcome_icds_sos(sos_df, liggar_df,
                             icd_prefixes, days_after_index_list,
                             colsuffix=""):
    sos_df = sos_df[sos_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, lopnr) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]
            ].iterrows()
    ):
        pat_sos = sos_df[(sos_df["LopNr"] == lopnr)]
        # Outcome
        ss = pat_sos[pat_sos["INDATUM"] > index_date + timedelta(days=-1)]
        new_row = []
        for days in days_after_index_list:
            s = ss[ss["INDATUM"] <= (index_date + timedelta(days=days))]
            outcome_icds = set(x
                               for lst in s[SOS_DIAG_COLS].values.tolist()
                               for x in lst if not pd.isna(x))
            # XXX: ICDs from DORS maybe should be added
            # if dod and index_date + timedelta(days=days) < dod + timedelta(
            #         hours=23, minutes=59):
            #     outcome_icds = outcome_icds.union(dors_icd)

            new_row += [1 if any(icd.startswith(p) for icd in outcome_icds)
                        else 0
                        for p in icd_prefixes]
        out.append(new_row)

    colnames = [f"outcome-{days}d-{icd}"
                for days in days_after_index_list
                for icd in icd_prefixes]
    if colsuffix:
        colnames = [f"{x}-{colsuffix}" for x in colnames]
    df = pd.DataFrame(out, columns=colnames, index=liggar_df["Alias"])
    return df


def extract_historical_atc_sos(sos_lakemedel_df, liggar_df,
                               atc_days_before_index_list,
                               atc_prefixes,
                               skip_first_n_days=0):
    lm_subset_df = sos_lakemedel_df
    lm_subset_df = lm_subset_df[lm_subset_df["Alias"].isin(liggar_df["Alias"])]

    def do_it(atc_days_before_index):
        out = []
        for _, (alias, index_date) in tqdm(
                liggar_df[["Alias", "Vardkontakt_InskrivningDatum"]].iterrows()
        ):
            lm = lm_subset_df[(lm_subset_df["Alias"] == alias)]
            lm = lm[
                (lm["EDATUM"] <=
                 index_date - timedelta(days=skip_first_n_days))
                &
                (lm["EDATUM"] >= index_date -
                 timedelta(days=atc_days_before_index))
                ]
            atcs = [1 if lm["ATC"].str.startswith(atc).sum() > 0
                    else 0 for atc in atc_prefixes]
            out.append(atcs)

        col_pf = f"med-{atc_days_before_index}d"
        if skip_first_n_days:
            col_pf += f"-skip{skip_first_n_days}d"
        colnames = [f"{col_pf}-{atc}" for atc in atc_prefixes]
        return out, colnames

    if type(atc_days_before_index_list) == list:
        ol = []
        colnames = []
        for days in atc_days_before_index_list:
            o, c = do_it(days)
            ol.append(np.array(o))
            colnames += c
        out = np.concatenate(ol, axis=1)
    else:
        out, colnames = do_it(atc_days_before_index_list)

    return pd.DataFrame(out, columns=colnames, index=liggar_df["Alias"])


def extract_index_blood_samples(labb_df, liggar_df,
                                blood_samples, max_hours_after_index):
    labb_df = labb_df[labb_df["Alias"].isin(liggar_df["Alias"])]
    labb_df = labb_df[
        ~labb_df["Analyssvar_Varde"].isin(BROKEN_BLOOD_SAMPLE_VALUES)
    ]

    out = []
    for _, (alias, index_date) in tqdm(
            liggar_df[["Alias", "Vardkontakt_InskrivningDatum"]].iterrows()
    ):
        lst = labb_df[(labb_df["Alias"] == alias)]
        lst = lst[
            (lst["Analyssvar_ProvtagningDatum"] >= index_date) &
            (lst["Analyssvar_ProvtagningDatum"] <= index_date +
             timedelta(hours=max_hours_after_index))
            ].sort_values(
            by=["Analyssvar_ProvtagningDatum", "Labanalys_Beskrivning"]
        )

        samples = []
        for bs in blood_samples:
            val = np.nan
            for ix, row in lst[lst["Labanalys_Beskrivning"] == bs].iterrows():
                val = row["Analyssvar_Varde"]
                break
            samples.append(val)

        out.append(samples)

    colnames = [BLOOD_SAMPLE_COL_MAP[b] for b in blood_samples]

    return pd.DataFrame(out, columns=colnames, index=liggar_df["Alias"])


def dump_sectra_to_annotate(alias, i, undersokningskod, undersokningsdatum,
                            undersokningsnummer, text, dump_root):
    filename = f"{dump_root}/{alias}_{i:02d}_" \
               f"{str(undersokningsdatum)[:10]}_" \
               f"{undersokningsnummer}_" \
               f"{undersokningskod}.txt"
    with open(filename, 'w', encoding='utf8') as f:
        f.write(text)


def extract_sectra_procedures(sectra_df, liggar_df,
                              sectra_codes, days_after,
                              days_before=1,
                              dump_text_root=None):
    sectra_df = sectra_df[sectra_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date) in tqdm(
            liggar_df[["Alias", "Vardkontakt_InskrivningDatum"]].iterrows()
    ):
        sectra_seen = []
        sec = sectra_df[(sectra_df["Alias"] == alias)]
        sec = sec[
            (sec["Undersökningsdatum"] >=
             (index_date - timedelta(days=days_before))) &
            (sec["Undersökningsdatum"] <=
             (index_date + timedelta(days=days_after))) &
            (sec["Undersökningskod"].isin(sectra_codes))
            ].sort_values(by="Undersökningsdatum")
        for i, (_, row) in enumerate(sec.iterrows()):
            sectra_seen.append(row["Undersökningskod"])
            if dump_text_root:
                dump_sectra_to_annotate(alias, i, row["Undersökningskod"],
                                        row["Undersökningsdatum"],
                                        row["Undersökningsnummer"],
                                        row["Fullständigt svar"],
                                        dump_text_root)

        out.append("" if len(sectra_seen) == 0 else ";".join(sectra_seen))
    df = pd.DataFrame(out, columns=[f"Sectra-{days_after}d"],
                      index=liggar_df["Alias"])
    return df


def extract_previous_surgery_sos_sv(sos_df, liggar_df,
                                    days_before_index):
    sos_df = sos_df[sos_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, lopnr) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]
            ].iterrows()
    ):
        surgery = 0
        pat_sluten = sos_df[(sos_df["LopNr"] == lopnr)]
        for _, row in pat_sluten[
            (pat_sluten["INDATUM"] < index_date) &
            (pat_sluten["INDATUM"] > index_date -
             timedelta(days=(days_before_index + 50)))
            # Add 50 days here to have some extra space backwards in time.
            # We will consider the actual OP date below anyway
        ][SOS_SV_OP_COLS].dropna(subset=["OP"]).iterrows():
            ops = row["OP"].split(" ")
            for i, o in enumerate(ops, start=1):
                d = pd.to_datetime(row[i])
                if index_date - timedelta(days=days_before_index) \
                        < d < index_date:
                    if o in SOS_KVA_TABLE.index and \
                            SOS_KVA_TABLE.loc[o, :]["KLASSDEL"]:
                        surgery = 1
                        break
        out.append(surgery)

    df = pd.DataFrame(out, columns=[f"surgery-{days_before_index}d-SV"],
                      index=liggar_df["Alias"])
    return df


def extract_previous_surgery_sos_ov(sos_df, liggar_df,
                                    days_before_index):
    sos_df = sos_df[sos_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, lopnr) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]
            ].iterrows()
    ):
        pat_oppen = sos_df[(sos_df["LopNr"] == lopnr)]
        surgery = 0
        for x in pat_oppen[
            (pat_oppen["INDATUM"] < index_date) &
            (pat_oppen["INDATUM"] > index_date -
             timedelta(days=days_before_index))].dropna(subset=["OP"])["OP"]:
            ops = x.split(" ")
            for o in ops:
                if o in SOS_KVA_TABLE.index and \
                        SOS_KVA_TABLE.loc[o, :]["KLASSDEL"]:
                    surgery = 1
                    break
        out.append(surgery)

    df = pd.DataFrame(out, columns=[f"surgery-{days_before_index}d-OV"],
                      index=liggar_df["Alias"])
    return df


def extract_outcome_kvas_sos_sv(sos_df, liggar_df, kva_strings,
                                days_after_index_list):
    sos_df = sos_df[sos_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, lopnr) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]
            ].iterrows()
    ):
        pat_sos = sos_df[(sos_df["LopNr"] == lopnr)]
        ss = pat_sos[pat_sos["INDATUM"] > index_date + timedelta(days=-1)]
        max_days = max(days_after_index_list)
        ss = ss[ss["INDATUM"] < index_date + timedelta(days=max_days)]
        seen = {}
        for _, row in ss[SOS_SV_OP_COLS].dropna(subset=["OP"]).iterrows():
            ops = row["OP"].split(" ")
            for i, o in enumerate(ops, start=1):
                if o not in kva_strings:
                    continue
                d = pd.to_datetime(row[i])
                if o in seen:
                    if seen[o] > d:
                        seen[o] = d
                else:
                    seen[o] = d

        new_row = []
        for days_after in days_after_index_list:
            ceiling = index_date + timedelta(days=days_after)
            o = [1 if k in seen and seen[k] < ceiling
                 else 0
                 for k in kva_strings]
            new_row += o
        out.append(new_row)

    colnames = [
        f"outcome-kva-{k}-{days}d-SV" for days in days_after_index_list
        for k in kva_strings
    ]
    df = pd.DataFrame(out, columns=colnames, index=liggar_df["Alias"])
    return df


def extract_outcome_kvas_sos_ov(sos_df, liggar_df, kva_strings,
                                days_after_index_list):
    sos_df = sos_df[sos_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, lopnr) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]
            ].iterrows()
    ):
        pat_oppen = sos_df[(sos_df["LopNr"] == lopnr)]
        pat_oppen = pat_oppen[
            pat_oppen["INDATUM"] > index_date + timedelta(days=-1)
        ]
        pat_oppen = pat_oppen[
            pat_oppen["INDATUM"] <
            index_date + timedelta(days=max(days_after_index_list))
        ]
        seen = {}
        for _, (op, d) in pat_oppen.dropna(subset=["OP"])[
            ["OP", "INDATUM"]
        ].iterrows():
            ops = op.split(" ")
            for o in (set(ops) & set(kva_strings)):
                if o in seen:
                    if seen[o] > d:
                        seen[o] = d
                else:
                    seen[o] = d

        new_row = []
        for days_after in days_after_index_list:
            ceiling = index_date + timedelta(days=days_after)
            o = [1 if k in seen and seen[k] < ceiling
                 else 0
                 for k in kva_strings]
            new_row += o
        out.append(new_row)

    colnames = [
        f"outcome-kva-{k}-{days}d-OV" for days in days_after_index_list
        for k in kva_strings
    ]
    df = pd.DataFrame(out, columns=colnames, index=liggar_df["Alias"])
    return df


def extract_death_sos_dors(sos_dors, liggar_df,
                           days_after_index_list):
    out = []
    for _, (alias, index_date) in tqdm(
            liggar_df[["Alias", "Vardkontakt_InskrivningDatum"]].iterrows()
    ):
        dors_icd = []
        dors = sos_dors[sos_dors["Alias"] == alias]
        dod = None
        if len(dors) > 0:
            row = dors.iloc[0, :]
            d = str(row["DODSDAT"])
            approx_dod = 0
            if d.endswith("0000"):  # Haxx
                d = d[:4] + "0630"
                approx_dod = 1
            elif d.endswith("00"):  # Haxx
                d = d[:6] + "15"
                approx_dod = 1
            dod = pd.to_datetime(d, format="%Y%m%d")
            dors_icd = [x for x in row[DORS_ICD_COLS] if not pd.isna(x)]
        if dod:
            death = [1 if (dod - timedelta(days=days)) < index_date else 0
                     for days in days_after_index_list]
            death.append(" ".join(dors_icd))
            death.append(str(row["DODSDAT"]))
            death.append(approx_dod)
        else:
            death = [0 for d in days_after_index_list]
            death.append("")
            death.append("")
            death.append(0)
        out.append(death)

    columns = [f"outcome-{days}d-DEATH" for days in days_after_index_list]
    columns.append("Death-ICD")
    columns.append("Date_Of_Death")
    columns.append("Death_Date_Approximate")

    df = pd.DataFrame(out, columns=columns,
                      index=liggar_df["Alias"])
    return df


def extract_liggare_future_visits(liggaren_full, liggaren_followup,
                                  subset_liggar_df, besoksorsaker_list,
                                  days_after_index_list):
    liggaren_full = liggaren_full[
        liggaren_full["Alias"].isin(subset_liggar_df["Alias"])
    ]
    liggaren_followup = liggaren_followup[
        liggaren_followup["Alias"].isin(subset_liggar_df["Alias"])
    ]

    lp = liggaren_full[
        ["Alias", "Vardkontakt_InskrivningDatum", "BesokOrsak_Kod"]
    ]
    lf = liggaren_followup[
        ["Alias", "Vardkontakt_InskrivningDatum", "BesokOrsak_Kod"]
    ]
    lc = pd.concat([lf, lp])

    out = []
    for _, (alias, index_date) in tqdm(
            subset_liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum"]
            ].iterrows()
    ):
        p_df = lc[lc["Alias"] == alias].copy()
        p_df["Vardkontakt_InskrivningDatum"] = pd.to_datetime(
            p_df["Vardkontakt_InskrivningDatum"]
        )
        p_df = p_df[p_df["Vardkontakt_InskrivningDatum"] > index_date]
        row = []
        for days in days_after_index_list:
            sok_orsaker = set(
                p_df[
                    p_df["Vardkontakt_InskrivningDatum"] <
                    index_date + timedelta(days=days)
                ]["BesokOrsak_Kod"]
            )
            row.append(1 if len(sok_orsaker) else 0)
            row += [1 if bo in sok_orsaker else 0 for bo in besoksorsaker_list]

        out.append(row)

    colnames = []
    for days in days_after_index_list:
        c = [f"outcome-revisit-{days}d"]
        c += [f"outcome-revisit-{bo}-{days}d" for bo in besoksorsaker_list]
        colnames += c

    df = pd.DataFrame(out, columns=colnames, index=subset_liggar_df["Alias"])
    return df


def combine_S_W_melior(out, melior_30day_icd_prefixes):
    S_COLS = [f"outcome-30d-{x}-Melior"
              for x in melior_30day_icd_prefixes if x[0] == "S"]
    W_COLS = [f"outcome-30d-{x}-Melior"
              for x in melior_30day_icd_prefixes if x[0] == "W"]

    def combine_S_W_melior_row(row):
        r = int(row[S_COLS].sum() > 0 and row[W_COLS].sum() > 1)
        return r

    s = out.apply(combine_S_W_melior_row, axis=1)
    s = s.rename("outcome-30d-S_and_W-Melior")
    return s


def extract_icd_outcomes_combine_sv_ov(sos_oppen, sos_sluten, liggar_df,
                                       TARGET_OUTCOME_ICD_PREFIXES,
                                       OUTCOME_DAYS_AFTER_INDEX):
    icd_outcomes_sv = extract_outcome_icds_sos(sos_sluten, liggar_df,
                                               TARGET_OUTCOME_ICD_PREFIXES,
                                               OUTCOME_DAYS_AFTER_INDEX, "SV")
    icd_outcomes_ov = extract_outcome_icds_sos(sos_oppen, liggar_df,
                                               TARGET_OUTCOME_ICD_PREFIXES,
                                               OUTCOME_DAYS_AFTER_INDEX, "OV")
    m = (icd_outcomes_ov.values + icd_outcomes_sv.values)
    m[m > 1] = 1
    colnames = [x[:-3] for x in list(icd_outcomes_ov)]
    df = pd.DataFrame(m, columns=colnames, index=liggar_df["Alias"])
    return df


def extract_kva_outcomes_combine_sv_ov(sos_oppen, sos_sluten, liggar_df,
                                       TARGET_OUTCOME_KVA_CODES,
                                       OUTCOME_DAYS_AFTER_INDEX):
    outcome_kva_sv = extract_outcome_kvas_sos_sv(sos_sluten, liggar_df,
                                                 TARGET_OUTCOME_KVA_CODES,
                                                 OUTCOME_DAYS_AFTER_INDEX)
    outcome_kva_ov = extract_outcome_kvas_sos_ov(sos_oppen, liggar_df,
                                                 TARGET_OUTCOME_KVA_CODES,
                                                 OUTCOME_DAYS_AFTER_INDEX)
    m = outcome_kva_ov.values + outcome_kva_sv.values
    m[m > 1] = 1
    colnames = [x[:-3] for x in list(outcome_kva_sv)]
    df = pd.DataFrame(m, columns=colnames, index=liggar_df["Alias"])
    return df


def extract_previous_diagnoses_combine_sv_ov(sos_oppen, sos_sluten, liggar_df,
                                             PREV_DISEASE_ICD_PREFIXES):
    prev_diseases_ov = extract_previous_diagnoses_sos_ov(
        sos_oppen, liggar_df, PREV_DISEASE_ICD_PREFIXES
    )
    prev_diseases_sv = extract_previous_diagnoses_sos_sv(
        sos_sluten, liggar_df, PREV_DISEASE_ICD_PREFIXES
    )
    m = (prev_diseases_ov.values + prev_diseases_sv.values)
    m[m > 1] = 1
    colnames = [x[:-3] for x in list(prev_diseases_ov)]
    df = pd.DataFrame(m, columns=colnames, index=liggar_df["Alias"])
    return df


def extract_melior_discharge_diags(melior_contact, liggar_df, icd_list):
    melior = melior_contact[melior_contact["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, kontaktid) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "KontaktId"]
            ].iterrows()
    ):
        kontakt_melior = melior[melior["KontaktId"] == kontaktid]
        row = [1 if kontakt_melior[
            "PatientDiagnos_Kod"
        ].str.contains(icd).any() else 0
               for icd in icd_list]
        out.append(row)

    colnames = [f"outcome-discharge-{icd}-Melior" for icd in icd_list]
    df = pd.DataFrame(out, columns=colnames, index=liggar_df["Alias"])
    return df


def expand_grouped_sos_icd_by_melior_historic(melior_diagnoser_5ar_fore,
                                              liggare_joined, expansion_dict):
    melior = melior_diagnoser_5ar_fore[
        melior_diagnoser_5ar_fore["Alias"].isin(liggare_joined["Alias"])
    ]

    out = []
    for _, l_row in tqdm(liggare_joined.iterrows()):
        alias = l_row["Alias"]
        inskrivning = l_row["Vardkontakt_InskrivningDatum"]
        row = []
        for key in sorted(expansion_dict.keys()):
            if l_row[key]:
                pat_melior = melior[melior["Alias"] == alias]
                pat_melior = pat_melior[
                    pat_melior["VårdtillfälleFörDiagnos_StartDatum"] <
                    inskrivning]
                vals = [1 if pat_melior["PatientDiagnos_Kod"].
                        str.fullmatch(icd).any() else 0
                        for icd in
                        expansion_dict[key]]
                row += vals
            else:
                row += [0 for _ in expansion_dict[key]]
        out.append(row)

    colnames = [f"prev5y-{icd}-Melior" for key in sorted(expansion_dict)
                for icd in expansion_dict[key]]
    df = pd.DataFrame(out, columns=colnames, index=liggare_joined["Alias"])
    return df


def get_index_start_stop(melior_contact, liggar_df):
    melior = melior_contact[
        melior_contact["Alias"].isin(liggar_df["Alias"])
    ].set_index("KontaktId")
    melior = melior[
        ["Alias", "VårdtillfälleFörDiagnos_StartDatum",
         "VårdtillfälleFörDiagnos_SlutDatum"]
    ].drop_duplicates()
    r = liggar_df[
        ["Alias", "KontaktId"]
    ].join(
        melior.drop("Alias", axis=1), on="KontaktId"
    ).drop("KontaktId", axis=1)
    r["VårdtillfälleFörDiagnos_SlutDatum"] = pd.to_datetime(
        r["VårdtillfälleFörDiagnos_SlutDatum"]
    )
    r["VårdtillfälleFörDiagnos_Längd_Timmar_AB"] = \
        (r["VårdtillfälleFörDiagnos_SlutDatum"] -
         r["VårdtillfälleFörDiagnos_StartDatum"]
         ).dt.total_seconds() / (60 * 60)
    return r.set_index("Alias")


def contact_and_30day_melior_outcomes(melior_kontakt, melior_30d,
                                      liggar_df, icd_prefixes):
    m1 = melior_kontakt[melior_kontakt["Alias"].isin(liggar_df["Alias"])]
    m2 = melior_30d[melior_30d["Alias"].isin(liggar_df["Alias"])]
    melior = pd.concat([m1, m2])
    data = []
    for _, (alias, index_date, kontaktid) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "KontaktId"]
            ].iterrows()
    ):
        m = melior[melior["Alias"] == alias]
        m = m[index_date <= m["VårdtillfälleFörDiagnos_StartDatum"]]
        m = m[
            index_date + timedelta(days=30) >=
            m["VårdtillfälleFörDiagnos_StartDatum"]
        ]
        row = [
            1 if m["PatientDiagnos_Kod"].str.contains(icd).sum() else 0
            for icd in icd_prefixes
        ]
        data.append(row)

    colnames = [f"outcome-30d-{icd}-Melior" for icd in icd_prefixes]
    df = pd.DataFrame(data, columns=colnames, index=liggar_df["Alias"])
    return df


def infer_newly_introduced_meds(liggare_joined, atc_prefixes,
                                short_days, long_days):
    out = []
    for atc in atc_prefixes:
        c1 = f"med-{short_days}d-{atc}"
        c2 = f"med-{long_days}d-skip{short_days}d-{atc}"
        n = f"med-newly-introduced-{short_days}d-{atc}"
        s = ((liggare_joined[c1]) & (~liggare_joined[c2]))
        s = s.rename(n)
        out.append(s)
    df = pd.concat(out, axis=1)
    return df


def find_first_stroke(sos_oppen, sos_sluten, out):
    stroke_cols = [f"outcome-365d-I{i}" for i in range(60, 70)]
    stroke_series = out[stroke_cols].any(axis=1).dropna()
    pos_stroke = stroke_series[stroke_series]
    sv = sos_sluten[sos_sluten["Alias"].isin(pos_stroke.index)]
    ov = sos_oppen[sos_oppen["Alias"].isin(pos_stroke.index)]

    def find_stroke_date(alias, inskrivningsdatum):
        s = sv[sv["Alias"] == alias]
        s = s[s["INDATUM"] >= inskrivningsdatum + timedelta(days=-1)]
        s = s[s["INDATUM"] <= inskrivningsdatum + timedelta(days=365)]
        s["HAS_STROKE_AB"] = s[SOS_DIAG_COLS].apply(
            lambda x: x.str.startswith("I6")
        ).any(axis=1)
        s = s[s["HAS_STROKE_AB"]]["INDATUM"]
        o = ov[ov["Alias"] == alias]
        o = o[o["INDATUM"] >= inskrivningsdatum + timedelta(days=-1)]
        o = o[o["INDATUM"] <= inskrivningsdatum + timedelta(days=365)]
        o["HAS_STROKE_AB"] = o[SOS_DIAG_COLS].apply(
            lambda x: x.str.startswith("I6")
        ).any(axis=1)
        o = o[o["HAS_STROKE_AB"]]["INDATUM"]
        date = pd.concat([o, s]).sort_values().iloc[0]
        return date

    r = []
    for alias, inskrivningsdatum in tqdm(
            zip(out.index, out["Vardkontakt_InskrivningDatum"])
    ):
        if alias not in pos_stroke.index:
            r.append([np.nan, np.nan])
            # print(f"{alias}")
            continue
        stroke_date = find_stroke_date(alias, inskrivningsdatum)
        row = [stroke_date]
        row.append((stroke_date - inskrivningsdatum).total_seconds() / (
                60 * 60 * 24) + 1)  # Add one to offset the timestamp/datestamp
        # issue. If stroke was on 2017-03-15, and contact in liggaren was
        # 2017-03-15 13:00, then the number of days between index and stroke
        # would be negative (between -1 and 0) -- this way everything is
        # positive (and can potentially be 365.<something> days,
        # which could be debated)
        r.append(row)
    df = pd.DataFrame(r,
                      columns=["Date_of_stroke", "Days_from_index_to_stroke"],
                      index=out.index)
    return df


def combine_cols_any(out, colnames, outcolname):
    s = out[colnames].any(axis=1)
    s = s.rename(outcolname)
    return s.astype(int)


##
# Lists taken from "New ICD-10 version of the Charlson Comorbidity Index
# predicted in-hospital mortality", by Sundararajan et al.,
# Journal of Clinical Epidemiology 57 (2004) 1288–1294
##

CHARLSON_KEY_ICD = dict(
    AcuteMyocardialInfarction=["I21", "I22", "I252"],
    CongestiveHeartFailure=["I50"],
    PeripheralVascularDisease=["I71", "I790", "I739", "R02", "Z958", "Z959"],
    CerebralVascularAccident=["I60", "I61", "I62", "I63", "I65", "I66", "G450",
                              "G451", "G452", "G458", "G459", "G46", "I64",
                              "G454", "I670", "I671", "I672", "I674", "I675",
                              "I676", "I677", "I678", "I679", "I681", "I682",
                              "I688", "I69"],
    Dementia=["F00", "F01", "F02", "F051"],
    PulmonaryDisease=["J40", "J41", "J42", "J44", "J43", "J45", "J46", "J47",
                      "J67", "J44", "J60", "J61", "J62", "J63", "J66", "J64",
                      "J65"],
    ConnectiveTissueDisorder=["M32", "M34", "M332", "M053", "M058", "M059",
                              "M060", "M063", "M069", "M050", "M052", "M051",
                              "M353"],
    PepticUlcer=["K25", "K26", "K27", "K28"],
    LiverDisease=["K702", "K703", "K73", "K717", "K740", "K742", "K746",
                  "K743", "K744", "K745"],
    Diabetes=["E109", "E119", "E139", "E149", "E101", "E111", "E131", "E141",
              "E105", "E115", "E135", "E145"],
    DiabetesComplications=["E102", "E112", "E132", "E142", "E103", "E113",
                           "E133", "E143", "E104", "E114", "E134", "E144"],
    Paraplegia=["G81", "G041", "G820", "G821", "G822"],
    RenalDisease=["N03", "N052", "N053", "N054", "N055", "N056", "N072",
                  "N073", "N074", "N01", "N18", "N19", "N25"],
    Cancer=["C0", "C1", "C2", "C3", "C40", "C41", "C43", "C45", "C46", "C47",
            "C48", "C49", "C5", "C6", "C70", "C71", "C72", "C73", "C74", "C75",
            "C76", "C80", "C81", "C82", "C83", "C84", "C85", "C883", "C887",
            "C889", "C900", "C901", "C91", "C92", "C93", "C940", "C941",
            "C942", "C943", "C9451", "C947", "C95", "C96"],
    MetastaticCancer=["C77", "C78", "C79", "C80"],
    SevereLiverDisease=["K729", "K766", "K767", "K721"],
    HIV=["B20", "B21", "B22", "B23", "B24"]
)


def _charlson_compute_score_df_row(row):
    # Count 1 for all
    zum = row.sum()

    # The following should get 2 points, hence add 1
    if row["Charlson-DiabetesComplications"]:
        zum += 1
    if row["Charlson-Paraplegia"]:
        zum += 1
    if row["Charlson-RenalDisease"]:
        zum += 1
    if row["Charlson-Cancer"]:
        zum += 1
    # The following should get 3 points, hence add 2
    if row["Charlson-MetastaticCancer"]:
        zum += 2
    if row["Charlson-SevereLiverDisease"]:
        zum += 2
    # The following should get 6 points, hence add 5
    if row["Charlson-HIV"]:
        zum += 5
    return zum


def compute_charlson_melior_5y(melior_diagnoser_5ar_fore, liggar_df):
    melior = melior_diagnoser_5ar_fore[
        melior_diagnoser_5ar_fore["Alias"].isin(liggar_df["Alias"])
    ]
    out = []
    for _, l_row in tqdm(liggar_df.iterrows()):
        alias = l_row["Alias"]
        inskrivning = l_row["Vardkontakt_InskrivningDatum"]
        pat_melior = melior[melior["Alias"] == alias]
        pat_melior = pat_melior[
            pat_melior["VårdtillfälleFörDiagnos_StartDatum"] < inskrivning
            ]
        pat_melior = pat_melior[
            pat_melior["VårdtillfälleFörDiagnos_StartDatum"] >
            inskrivning - timedelta(days=5 * 365)
            ]
        diags = set(pat_melior["PatientDiagnos_Kod"])
        chs = []
        for ch_key in sorted(CHARLSON_KEY_ICD.keys()):
            charlson_pf_list = CHARLSON_KEY_ICD[ch_key]
            t = any(
                (icd.startswith(pf) for icd in diags
                 for pf in charlson_pf_list)
            )
            chs.append(t)
        out.append(chs)
    colnames = [f"Charlson-{s}" for s in sorted(CHARLSON_KEY_ICD.keys())]
    df = pd.DataFrame(
        out, columns=colnames, index=liggar_df["Alias"]
    ).astype(int)
    scores = df.apply(
        _charlson_compute_score_df_row, axis=1
    ).rename("Charlson-SCORE")
    df = df.join(scores)
    return df
