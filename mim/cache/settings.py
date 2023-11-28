from mim.util.metadata import Validator


STUPID = Validator(
    allow_uncommitted=True,
    allow_different_commits=True,
    allow_different_branches=True,
    allow_different_files=True,
    allow_different_environments=True,
    max_age_difference=None
)

UNSAFE = Validator(
    allow_uncommitted=False,
    allow_different_commits=True,
    allow_different_branches=True,
    allow_different_files=False,
    allow_different_environments=True,
    max_age_difference=None
)

SAFE = Validator(
    allow_uncommitted=False,
    allow_different_commits=False,
    allow_different_branches=False,
    allow_different_files=False,
    allow_different_environments=False,
    max_age_difference=None
)

enabled = False
validator = SAFE


def set_level(level):
    global validator
    assert level.lower() in ['stupid', 'unsafe', 'safe']

    if level == 'stupid':
        validator = STUPID
    elif level == 'unsafe':
        validator = UNSAFE
    else:
        validator = SAFE
