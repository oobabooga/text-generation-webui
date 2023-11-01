import copy


def get_default_generate_params():
    return copy.deepcopy(default_generate_params)


def default(dic, key, default):
    '''
    little helper to get defaults if arg is present but None and should be the same type as default.
    '''
    val = dic.get(key, default)
    if not isinstance(val, type(default)):
        # maybe it's just something like 1 instead of 1.0
        try:
            v = type(default)(val)
            if type(val)(v) == val:  # if it's the same value passed in, it's ok.
                return v
        except:
            pass

        val = default

    return val


def clamp(value, minvalue, maxvalue):
    return max(minvalue, min(value, maxvalue))
