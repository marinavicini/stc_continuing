import unidecode

def remove_accent(acc_string):
    '''Return string without accents'''
    unaccented_string = unidecode.unidecode(acc_string)
    return unaccented_string