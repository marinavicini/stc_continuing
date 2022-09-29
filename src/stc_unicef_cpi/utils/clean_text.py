import unidecode
import re

def remove_accent(acc_string):
    '''Return string without accents'''
    unaccented_string = unidecode.unidecode(acc_string)
    return unaccented_string


def remove_apostrophe(string):
    '''Return string without apostrophe'''
    return re.sub("'", "", string)


def clean_string_gee(string):
    '''Format string to pass as the name for the configuration'''
    string = remove_accent(string)
    string = remove_apostrophe(string)
    return string