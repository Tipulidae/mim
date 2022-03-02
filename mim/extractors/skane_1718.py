# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd

from mim.util.ab_util import parse_iso8601_datetime


###
# Quirks
#  -- MELIOR_LabanalyserInom24TimmarFrånAnkomst.csv
#  --- Some duplicates in general, just remove
#  --- Entries may be duplicated but with different KontaktId (in case the
#      patient came to ER twice within 24h)
#
#  -- MELIOR_OrdinationerSamtUtdelningarEttÅrFöreAnkomst.csv
#  --- Contains a spurious " on line 1265613, must be parsed with
#      pd.read_csv(..., quoting=3)
###

class BgColors(Enum):
    Melior = ""
    RED = "#dsgdsgsd"
    # etc, but too lazy to realize it now

class RowEvent:
    def __init__(self, row, timestamp):
        self.row = row
        self.timestamp = timestamp

    def row_prefix(self):
        s = f'<tr bgcolor="{self.BGCOLOR}">' \
            f'<td>{self.SHORTNAME}</td>'
        if "KontaktId" in self.row:
            s += f'<td>{self.timestamp} ({self.row["KontaktId"]})</td>'
        else:
            s += f'<td>{self.timestamp}</td>'

        return s


class MeliorDiagnosEvent(RowEvent):
    SHORTNAME = "Melior"
    BGCOLOR = "#99CCCC"

    def __init__(self, slutdatum, source, entries, **kwargs):
        super().__init__(**kwargs)
        self.slutdatum = slutdatum
        self.SHORTNAME += f"<br/>{source}"
        #        self.source = source
        self.entries = entries

    def to_html(self):
        s = self.row_prefix()
        s += f'<td>{self.slutdatum}</td>' \
             '<td colspan=3>'
        for e in self.entries:
            s += f'{e}<br/>'
        s += "</td>"
        return s


class SoSOppenEvent(RowEvent):
    SHORTNAME = "SoS_OV"
    BGCOLOR = "#66AA88"

    def to_html(self):
        diagnoses = [
            self.row[k] for k in [f"DIA{i}" for i in range(1, 31)]
            if not pd.isnull(self.row[k])
        ]
        gender = "Man" if self.row["KON"] == 1 else "Kvinna"
        if pd.isnull(self.row["OP"]):
            kva = ""
        else:
            kva = "<br/>".join(self.row["OP"].split(" "))

        s = self.row_prefix()
        s += f'<td>-----<br/>{gender}, ' \
             f'{self.row["ALDER"]} år</td>' \
             f'<td>Huvuddiagnos: {self.row["hdia"]}<br/>' \
             f'{", ".join(diagnoses)}</td>' \
             f'<td>{kva}</td>' \
             f'<td>Planerad Vård: {self.row["PVARD"]}<br/>' \
             f'Sjukhus: {self.row["SJUKHUS"]}<br/>' \
             f'MVO: {self.row["MVO"]}</td></tr>'

        return s


class SoSLakemedelEvent(RowEvent):
    SHORTNAME = "SoS_LM"
    BGCOLOR = "#9977BB"

    def to_html(self):

        s = self.row_prefix()
        s += f'<td></td>' \
             f'<td>Namn: {self.row["subnamn"]}<br/>' \
             f'ATC: {self.row["ATC"]}</td>' \
             f'<td></td><td></td></tr>'

        return s

class SoSSlutenEvent(RowEvent):
    SHORTNAME = "SoS_SV"
    BGCOLOR = "#77BB99"

    def to_html(self):
        diagnoses = [
            self.row[k] for k in [f"DIA{i}" for i in range(1, 31)]
            if not pd.isnull(self.row[k])
        ]
        gender = "Man" if self.row["KON"] == 1 else "Kvinna"
        if pd.isnull(self.row["OP"]):
            kva = ""
        else:
            kva_codes = self.row["OP"].split(" ")
            dates = [self.row[f'OPD{i}'] for i in range(1, len(kva_codes) + 1)]
            kva = "<br/>".join(
                [f'{k} ({d})' for k, d in zip(kva_codes, dates)])
        s = self.row_prefix()
        s += f'<td>{self.row["UTDATUM"]}<br/>{gender}, ' \
             f'{self.row["ALDER"]} år</td>' \
             f'<td>Huvuddiagnos: {self.row["hdia"]}<br/>' \
             f'{", ".join(diagnoses)}</td>' \
             f'<td>{kva}</td>' \
             f'<td>Planerad Vård: {self.row["PVARD"]}<br/>' \
             f'Sjukhus: {self.row["SJUKHUS"]}<br/>' \
             f'MVO: {self.row["MVO"]}</td></tr>'

        return s


class SoSDodsOrsakEvent(RowEvent):
    SHORTNAME = "SoS_DORS"
    BGCOLOR = "#CC6666"

    OP_CODES = {
        1: "Opererad inom 4v: Ja",
        2: "Opererad inom 4v: Nej",
        3: "Opererad inom 4v: Uppgift saknas"
    }
    DODSPL_CODES = {
        1: "Dödsplats: Sjukhus",
        2: "Dödsplats: Sjukhem eller SäBo",
        3: "Dödsplats: Privat bostad",
        4: "Dödsplats: Annan/okänd"
    }

    def to_html(self):
        dodsdat = str(self.row["DODSDAT"])
        dodsdat = f'<b>OKLART:</b> {dodsdat}<br/>' if dodsdat.endswith(
            "00") else ""
        gender = "Man" if self.row["KON"] == 1 else "Kvinna"
        age = self.row["alder"]
        ulorsak = self.row["ULORSAK"]
        morsak = [
            self.row[k] for k in [f"MORSAK{i}" for i in range(1, 24)]
            if not pd.isnull(self.row[k])
        ]
        opererad = self.OP_CODES[int(self.row["OPERERAD"])] if not pd.isnull(
            self.row["OPERERAD"]) else "Opererad inom 4v: ej angiven i intyg"
        dodspl = self.DODSPL_CODES[int(self.row["DODSPL"])] if not pd.isnull(
            self.row["DODSPL"]) else "Dödsplats: ej angiven i intyg"
        bullets = [dodspl, opererad]
        if not pd.isnull(self.row["DIABETES"]):
            bullets.append("Diabetiker")
        if not pd.isnull(self.row["ALKOHOL"]):
            bullets.append("Alkoholrelaterat dödsfall")
        if not pd.isnull(self.row["DODUTL"]):
            bullets.append("Död utomlands")
        bullets.append(f'Död kommun: {self.row["DOD_KOMMUN"]}')
        s = self.row_prefix()
        s += f'<td>{dodsdat}{gender}, {age} år</td>' \
             f'<td>ULORSAK: {ulorsak}<br/>MORSAK: ' \
             f'{", ".join(morsak)}'
        if not pd.isnull(self.row["KAP19"]):
            s += f'<br/>KAP19: {self.row["KAP19"]}'
        s += f'</td><td>{"<br/>".join(bullets)}</td>'
        s += "<td>" \
             f'Folkbokförd: {self.row["LK"]}<br/>' \
             f'Nationalitet: {self.row["NATION"]}<br/>' \
             f'Födelseland: {self.row["FODLAND"]}</td>'
        s += "</tr>"
        return s


class LabbEvent(RowEvent):
    SHORTNAME = "Labb"
    BGCOLOR = "#CC99CC"

    def to_html(self):
        s = self.row_prefix()
        s += f'<td></td><td>{self.row["Labanalys_Beskrivning"]}</td>' \
             f'<td>{self.row["Analyssvar_Varde"]} ' \
             f'{self.row["Analyssvar_Enhet"]}</td>' \
             f'<td>Referens (min-max): ' \
             f'{self.row["Analyssvar_ReferensvardeMin"]} - ' \
             f'{self.row["Analyssvar_ReferensvardeMax"]}</td>' \
             '</tr>'
        return s


class IndexEvent(RowEvent):
    SHORTNAME = "Liggaren"
    BGCOLOR = "#CCCC99"

    def __init__(self, preliminaries, **kwargs):
        super().__init__(**kwargs)
        self.preliminaries = preliminaries

    def to_html(self):
        gender = "Kvinna" if self.row["Patient_Kon"] == "F" else "Man"
        s = self.row_prefix()
        s += f'<td>{self.row["Vardkontakt_UtskrivningDatum"]}' \
             f'<br/>{gender}, ' \
             f'{self.row["Vardkontakt_PatientAlderVidInskrivning"]} år</td>'
        s += f'<td>{self.row["Sjukhus_Namn"]}<br/>' \
             f'<br/>Preliminär diagnos: '
        if len(self.preliminaries) > 0:
            for i, r in self.preliminaries.iterrows():
                s += f'<br/>{r["PatientDiagnos_Kod"]} ({r["Diagnostyp"]})'
        else:
            s += "n/a"

        s += f'<td>{self.row["BesokOrsak_Beskrivning"]}<br/>' \
             f'Processtext: {self.row["Process_text"]}<br/>' \
             f'UppföljningParameter: {self.row["UppföljningParameter_text"]}' \
             f'</td>'
        s += f'</td>' \
             f'<td>{self.row["Utskriven till"]}/' \
             f'{self.row["Inläggningsavdelning"]}<br/>' \
             f'Inlagd: {self.row["Inlagd"]}</td>' \
             '</tr>'
        return s


def extract_sos_dors(sos_dors, alias):
    x = sos_dors[sos_dors["Alias"] == alias]
    if len(x) == 0:
        return []

    row = x.iloc[0, :]

    d = str(row["DODSDAT"])
    if d.endswith("0000"):
        date = datetime(year=int(d[0:4]), month=12, day=31)
    elif d.endswith("00"):
        if d[4:6] == "12":
            date = datetime(year=int(d[0:4]) + 1, month=1, day=1)
        else:
            date = datetime(year=int(d[0:4]), month=int(d[4:6]) + 1, day=1)
        date = date + timedelta(days=-1)
    else:
        date = datetime(year=int(d[0:4]), month=int(d[4:6]), day=int(d[6:8]))

    date = date + timedelta(hours=23, minutes=59, seconds=59)

    return [SoSDodsOrsakEvent(row=row, timestamp=date)]


def extract_melior(dfs, alias):
    a = (
        dfs["melior_pre_5yr"][dfs["melior_pre_5yr"]["Alias"] == alias]
        .drop(["TermId", "KontaktId"], axis=1)
        .fillna({"VårdtillfälleFörDiagnos_SlutDatum": "n/a"})
    )
    b = dfs["melior_kontakt"][dfs["melior_kontakt"]["Alias"] == alias].drop(
        "KontaktId", axis=1).fillna(
        {"VårdtillfälleFörDiagnos_SlutDatum": "n/a"})
    c = dfs["melior_post_30dagar"][
        dfs["melior_post_30dagar"]["Alias"] == alias].drop(
        ["TermId", "KontaktId"], axis=1).fillna(
        {"VårdtillfälleFörDiagnos_SlutDatum": "n/a"})

    for df in [a, b, c]:
        df['vtkey'] = df["VårdtillfälleFörDiagnos_StartDatum"] + "~" + df[
            "VårdtillfälleFörDiagnos_SlutDatum"]

    vtkeys = set(pd.concat([a['vtkey'], b['vtkey'], c['vtkey']]))
    m_table = {}

    for vtkey in vtkeys:
        v = ""
        if (a['vtkey'] == vtkey).sum() > 0:
            v += "--Pre-5-years<br/>"
        if (b['vtkey'] == vtkey).sum() > 0:
            v += "--Contact<br/>"
        if (c['vtkey'] == vtkey).sum() > 0:
            v += "--Post 30d<br/>"
        m_table[vtkey] = v

    cat = pd.concat([a, b, c]).drop_duplicates()

    melior_events = []
    for vtkey in sorted(vtkeys):
        start_stop = vtkey.split("~")
        timestamp = parse_iso8601_datetime(start_stop[0])
        row = {}
        slutdatum = "n/a" if start_stop[
                                 1] == "n/a" else parse_iso8601_datetime(
            start_stop[1])
        entries = [
            f'{r["PatientDiagnos_ModifieradDatum"]}: '
            f'{r["PatientDiagnos_Kod"]} ({r["Diagnostyp"]})  '
            f'({r["PatientDiagnos_Beskrivning"]}) '
            f'({r["VårdtillfälleFörDiagnos_VardformText"]}) '
            f'({r["AktivitetTyp"]})'
            for k, r in cat[cat["vtkey"] == vtkey].sort_values(
                by=["PatientDiagnos_ModifieradDatum"]).iterrows()]
        melior_events.append(
            MeliorDiagnosEvent(slutdatum, m_table[vtkey], entries,
                               timestamp=timestamp, row=row))

    return melior_events


def generic_df_to_event_by_alias(df, alias, date_col, cls,
                                 drop_duplicates_kontaktid_removed=False):
    out = []
    items = df[df["Alias"] == alias]
    if drop_duplicates_kontaktid_removed:
        cols = list(items)
        cols.remove("KontaktId")
        items = items.drop_duplicates(subset=cols)

    for idx, r in items.iterrows():
        date = parse_iso8601_datetime(r[date_col])
        out.append(cls(r, date))
    return out


def index_visit_to_event(index_df, prelim_df, alias):
    out = []
    items = index_df[index_df["Alias"] == alias]
    for idx, r in items.iterrows():
        date = parse_iso8601_datetime(r["Vardkontakt_InskrivningDatum"])
        prelims = prelim_df[prelim_df["KontaktId"] == r["KontaktId"]]
        out.append(IndexEvent(prelims, row=r, timestamp=date))
    return out


def get_html_pre_table(alias):
    s = f'<html><head><title>{alias}</title>' \
        f'<meta charset="UTF-8"></head>\n<body>\n'
    s += "<table cellpadding=2 cellspacing=2><tr>" \
         "<th>Source</th><th>Timestamp (KontaktId)</th>" \
         "<th>Col1</th><th>Col2</th><tr/>\n"
    return s


def write_patient_html(alias, filename, dfs):
    events = []
    # events += generic_df_to_event_by_alias(dfs["liggaren"], alias,
    #                                       "Vardkontakt_InskrivningDatum",
    #                                       IndexEvent)
    events += index_visit_to_event(dfs["liggaren"], dfs["prelim"], alias)
    events += generic_df_to_event_by_alias(dfs["labb"], alias,
                                           "Analyssvar_ProvtagningDatum",
                                           LabbEvent, True)
    events += generic_df_to_event_by_alias(dfs["sos_sluten"], alias,
                                           "INDATUM", SoSSlutenEvent, False)
    events += generic_df_to_event_by_alias(dfs["sos_oppen"], alias,
                                           "INDATUM", SoSOppenEvent, False)
    events += extract_melior(dfs, alias)
    events += extract_sos_dors(dfs["sos_dors"], alias)
    events += generic_df_to_event_by_alias(dfs["sos_lm"], alias, "EDATUM",
                                           SoSLakemedelEvent, False)

    with open(filename, "w") as fid:
        fid.write(get_html_pre_table(alias))
        for e in sorted(events, key=lambda x: x.timestamp):
            fid.write(e.to_html())
            fid.write("\n")
            #print(e.to_html())
