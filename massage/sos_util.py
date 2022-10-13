def fix_dors_date(s):
    if s[-4:] == '0000':
        return s[:-4] + '1201'
    elif s[-2:] == '00':
        return s[:-2] + '01'
    else:
        return s
