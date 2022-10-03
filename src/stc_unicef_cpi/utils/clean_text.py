import unidecode
import stc_unicef_cpi.utils.constants as c

def remove_accent(acc_string):
    '''Return string without accents'''
    unaccented_string = unidecode.unidecode(acc_string)
    return unaccented_string


def get_country_name_gaul(name):
    
    if name in c.dic_pycountry_to_gaul.keys():
        return c.dic_pycountry_to_gaul[name]
    else:
        return name