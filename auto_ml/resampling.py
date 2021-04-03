

def initial_resample(x, y, res_type='all'):
    """
    function takes x and y
    then trying to find single best resample
    return res_x, res_y
    """
    from imblearn.combine import SMOTEENN, SMOTETomek
    resamplers = []
    rand = 42

    # compare with None ???
    if   res_type == 'under':
        pass
    elif res_type == 'over':
        pass
    elif res_type == 'combined':
        resamplers.append(('SMOTEENN', SMOTEENN(random_state=rand)))
        resamplers.append(('SMOTETomek', SMOTETomek(random_state=rand)))
    elif res_type == 'all':
        pass

    x,y = 1,1
    return x ,y



