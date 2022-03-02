# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
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
SOS_DIAG_COLS = ["hdia"] + [f"DIA{i}" for i in range(1,31)]
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
        ov = pat_oppen[(pat_oppen["INDATUM"] < index_date) &
                       (pat_oppen["INDATUM"] > index_date - timedelta(days=5*365))]
        prev_icds = set(x for l in ov[SOS_DIAG_COLS].values.tolist() for x in l if not pd.isna(x))
        new_row = [1 if any(icd.startswith(p) for icd in prev_icds) else 0 for p in icd_prefixes]
        out.append(new_row)

    colnames = [f"prev5y-{x}-OV" for x in icd_prefixes]
    df = pd.DataFrame(out, columns=colnames, index=liggar_df["Alias"])
    return df


def extract_previous_diagnoses_sos_sv(sos_df, liggar_df, icd_prefixes):
    sos_df = sos_df[sos_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, lopnr) in tqdm(
        liggar_df[["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]].iterrows()
    ):
        pat_sluten = sos_df[(sos_df["LopNr"] == lopnr)]

        ss = pat_sluten[(pat_sluten["UTDATUM"] < index_date) &
                        (pat_sluten["UTDATUM"] > index_date - timedelta(
                            days=5 * 365))]
        prev_icds = set(x
                        for l in ss[SOS_DIAG_COLS].values.tolist()
                        for x in l if not pd.isna(x)
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
        liggar_df[["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]].iterrows()
    ):
        pat_sos = sos_df[(sos_df["LopNr"] == lopnr)]
        # Outcome
        ss = pat_sos[pat_sos["INDATUM"] > index_date + timedelta(days=-1)]
        new_row = []
        for days in days_after_index_list:
            s = ss[ss["INDATUM"] <= (index_date + timedelta(days=days))]
            outcome_icds = set(
                x for l in s[SOS_DIAG_COLS].values.tolist() for x in l if
                not pd.isna(x))
            # XXX: ICDs from DORS maybe should be added
            # if dod and index_date + timedelta(days=days) < dod + timedelta(
            #         hours=23, minutes=59):
            #     outcome_icds = outcome_icds.union(dors_icd)

            new_row += [1 if any(icd.startswith(p) for icd in outcome_icds) else 0
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
                               atc_prefixes):

    lm_subset_df = sos_lakemedel_df
    lm_subset_df = lm_subset_df[lm_subset_df["Alias"].isin(liggar_df["Alias"])]

    def do_it(atc_days_before_index):
        out = []
        for _, (alias, index_date) in tqdm(
                liggar_df[["Alias", "Vardkontakt_InskrivningDatum"]].iterrows()
        ):
            lm = lm_subset_df[(lm_subset_df["Alias"] == alias)]
            lm = lm[(lm["EDATUM"] <= index_date) &
                    (lm["EDATUM"] >= index_date - timedelta(
                        days=atc_days_before_index))]
            atcs = [1 if lm["ATC"].str.startswith(atc).sum() > 0
                    else 0 for atc in atc_prefixes]
            out.append(atcs)

        colnames = [f"med-{atc_days_before_index}d-{atc}" for atc in atc_prefixes]
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
        l = labb_df[(labb_df["Alias"] == alias)]
        l = l[(l["Analyssvar_ProvtagningDatum"] >= index_date) &
              (l["Analyssvar_ProvtagningDatum"] <= index_date + timedelta(
                  hours=max_hours_after_index))
              ].sort_values(
            by=["Analyssvar_ProvtagningDatum", "Labanalys_Beskrivning"])

        samples = []
        for bs in blood_samples:
            val = np.nan
            for ix, row in l[l["Labanalys_Beskrivning"] == bs].iterrows():
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
            (sec["Undersökningsdatum"] >= (index_date - timedelta(days=days_before))) &
            (sec["Undersökningsdatum"] <= (index_date + timedelta(days=days_after))) &
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
        liggar_df[["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]].iterrows()
    ):
        surgery = 0
        pat_sluten = sos_df[(sos_df["LopNr"] == lopnr)]
        for _, row in pat_sluten[
            (pat_sluten["INDATUM"] < index_date) &
            (pat_sluten["INDATUM"] > index_date -
             timedelta(days=(days_before_index+50))) # Add 50 days here to have some extra space backwards in time. We will consider the actual OP date below anyway
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
        liggar_df[["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]].iterrows()
    ):
        pat_oppen = sos_df[(sos_df["LopNr"] == lopnr)]
        surgery = 0
        for x in pat_oppen[(pat_oppen["INDATUM"] < index_date) &
                           (pat_oppen["INDATUM"] > index_date -
                            timedelta(days=days_before_index))].dropna(
            subset=["OP"])["OP"]:
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
                if not o in kva_strings:
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


def extract_outcome_kvas_sos_ov(sos_df, liggar_df, kva_strings, days_after_index_list):
    sos_df = sos_df[sos_df["Alias"].isin(liggar_df["Alias"])]

    out = []
    for _, (alias, index_date, lopnr) in tqdm(
            liggar_df[
                ["Alias", "Vardkontakt_InskrivningDatum", "LopNr"]
            ].iterrows()
    ):
        pat_oppen = sos_df[(sos_df["LopNr"] == lopnr)]
        pat_oppen = pat_oppen[pat_oppen["INDATUM"] > index_date + timedelta(days=-1)]
        pat_oppen = pat_oppen[pat_oppen["INDATUM"] < index_date + timedelta(days=max(days_after_index_list))]
        seen = {}
        for _, (op, d) in pat_oppen.dropna(subset=["OP"])[["OP", "INDATUM"]].iterrows():
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
        liggar_df[["Alias","Vardkontakt_InskrivningDatum"]].iterrows()
    ):
        dors_icd = []
        dors = sos_dors[sos_dors["Alias"] == alias]
        dod = None
        if len(dors) > 0:
            row = dors.iloc[0, :]
            d = str(row["DODSDAT"])
            approx_dod = 0
            if d.endswith("0000"):  ## Haxx
                d = d[:4] + "0630"
                approx_dod = 1
            elif d.endswith("00"):    ## Haxx
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


def extract_liggare_future_visits(liggaren_full, subset_liggar_df, besoksorsaker_list, days_after_index_list):
    liggaren_full = liggaren_full[liggaren_full["Alias"].isin(subset_liggar_df["Alias"])]
    out = []
    for _, (alias, index_date) in tqdm(
            subset_liggar_df[["Alias", "Vardkontakt_InskrivningDatum"]].iterrows()
    ):
        p_df = liggaren_full[liggaren_full["Alias"] == alias].copy()
        p_df["Vardkontakt_InskrivningDatum"] = pd.to_datetime(p_df["Vardkontakt_InskrivningDatum"])
        p_df = p_df[p_df["Vardkontakt_InskrivningDatum"] > index_date]
        row = []
        for days in days_after_index_list:
            sok_orsaker = set(
                p_df[p_df["Vardkontakt_InskrivningDatum"] < index_date + timedelta(days=days)]["BesokOrsak_Kod"])
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
