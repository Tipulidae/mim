import re

from mim.util.util import take


def select_multihot_columns(columns, source="", diagnoses=0,
                            interventions=0, meds=0, **kwargs):
    """
    Select a subset of columns based on their names.

    :param columns: Iterable of column names to choose from
    :param source: The "source", either SV or OV, from which to choose
    diagnosis and intervention codes. If None, uses both sources.
    :param diagnoses: How many diagnoses (ICD) codes to return from each
    source. If negative, returns all diagnoses.
    :param interventions: How many intervention (KVÅ) codes to return from
    each source. If negative, returns all intervention codes.
    :param meds: How many medication (ATC) codes to return. If negative,
    return all medication codes.
    :return: Subset of input names, as a list. Any columns that do not match
    the patterns (like Alias, date, etc.) are ignored and not returned.
    """
    output = []
    if diagnoses:
        p = re.compile(f"{source}_ICD_", re.IGNORECASE)
        output += take(diagnoses, filter(p.search, columns))

    if interventions:
        p = re.compile(f"{source}_OP_", re.IGNORECASE)
        output += take(interventions, filter(p.search, columns))

    if meds:
        p = re.compile("ATC_", re.IGNORECASE)
        output += take(meds, filter(p.search, columns))

    return output
