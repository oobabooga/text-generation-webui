def get_alpha_value(alpha, base):
    '''
    Gets alpha_value from alpha_value and rope_freq_base
    '''
    if base > 0:
        return (base/10000.) ** (63/64.)
    else:
        return alpha


def get_rope_freq_base(alpha, base):
    '''
    Gets rope_freq_base from alpha_value and rope_freq_base
    '''
    if base > 0:
        return base
    else:
        return 10000 * alpha ** (64/63.)
