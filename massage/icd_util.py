import re

# The ?P<name> syntax gives a name to the group, allowing me to tell which
# part of the pattern was matched against. This way, I can describe all the
# chapters in a single expression.
icd_pattern = re.compile(
    "(?P<I>^[AB])|"
    "(?P<II>^(C)|(D[0-4]))|"
    "(?P<III>^D[5-8])|"
    "(?P<IV>^E)|"
    "(?P<V>^F)|"
    "(?P<VI>^G)|"
    "(?P<VII>^H[0-5])|"
    "(?P<VIII>^H[6-9])|"
    "(?P<IX>^I)|"
    "(?P<X>^J)|"
    "(?P<XI>^K)|"
    "(?P<XII>^L)|"
    "(?P<XIII>^M)|"
    "(?P<XIV>^N)|"
    "(?P<XV>^O)|"
    "(?P<XVI>^P)"
)

icd_levels = ['chapter', 'section', 'category', 'full']
icd_chapters = [
    'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII',
    'XIII', 'XIV', 'XV', 'XVI'
]


def round_icd_to_chapter(code):
    if match := icd_pattern.match(code):
        return match.lastgroup
    return "XXX"


def round_icd_to_section(code):
    raise NotImplementedError


def round_icd_to_category(code):
    raise NotImplementedError


# 1 character, then 2 digits, then maybe 2 characters maybe followed by
# 2 digits
atc_pattern = re.compile("^[ABCDGHJLMNPRSV]\\d{2}([A-X]{0,2}(\\d{2})?)?")
atc_levels = ['anatomical', 'therapeutic', 'pharmacological', 'chemical',
              'full']


def atc_to_level_rounder(level=None):
    if level is None:
        level = 'full'

    level = level.lower()
    assert level in atc_levels
    if level == 'anatomical':
        return round_atc_to_anatomical
    elif level == 'therapeutic':
        return round_atc_to_therapeutic
    elif level == 'pharmacological':
        return round_atc_to_pharmacological
    elif level == 'full':
        return round_atc_to_chemical
    else:
        return lambda x: x


def round_atc_to_anatomical(code):
    return code[0]


def round_atc_to_therapeutic(code):
    return code[:3]


def round_atc_to_pharmacological(code):
    return code[:4]


def round_atc_to_chemical(code):
    return code[:5]
