def contagem(l: list) -> int:
    """[summary]

    Args:
        l (list): [description]

    Returns:
        int: [description]
    """
    cont = 0
    for i in l:
        cont += 1
    return cont

def soma_list(l: list) -> float:
    soma = 0
    for i in l:
        soma += i
    return soma

def media_list(l: list) -> float:
    media = soma_list(l) / contagem(l)
    return media

def sample_variance(l: list) -> float:
    media   = media_list(l)
    tamanho = contagem  (l)

    diff_quad = [(x-media)**2 for x in l]
    soma = soma_list(diff_quad)
    var = soma / (tamanho-1)
    return var

def sample_covariance(feature: list, target: list) -> float:
    tamanho = contagem(feature)
    med_feat = media_list(feature)
    med_targ = media_list(target)

    diff = [(x-med_feat)*(y-med_targ) for x,y in zip(feature,target)]
    soma = soma_list(diff)
    cov  = soma / (tamanho-1)

    return cov

def FNC_fit(feature: list, target: list) -> float:
    med_feat = media_list(feature)
    med_targ = media_list(target)

    beta_num = sample_covariance(feature, target)
    beta_den = sample_variance(feature)

    beta = beta_num / beta_den

    alfa = med_targ - beta * med_feat

    return alfa, beta

def FNC_predict(feature: list, target: list) -> list:
    alfa, beta = FNC_fit(feature, target)
    target_pred = [(alfa + beta*x) for x in feature]
    
    return target_pred