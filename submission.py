## import modules here

from sklearn.metrics import f1_score
import nltk
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from collections import Counter

################# training #################
def ngram_Pr(syl_list, length):
    return tuple(zip(*(syl_list[i:] for i in range(length))))

def get_ngrams(syl_list):
    ngrams = set()

    leng = len(syl_list)
    for i in range(2,leng + 1):
        ngrams.update(ngram_Pr(syl_list,i))
    return ngrams
def upper(s):
    return {x.upper() for x in s}
def check_prefix(word,prefix_set):
    for i in range(len(word) + 1):
        if word[:i] in prefix_set:
            return word[:i]
    return False
def check_suffix(word,suffix_set):
    word_len = len(word)
    for i in range(word_len + 1):
        if word[abs(i - word_len):] in suffix_set:
            return word[abs(i - word_len):]
    return False
def train(data, classifier_file):  # do not change the heading of the function
    data_list = data
    model_x=[]
    model_y=[]
    vo_list={'IH', 'UW', 'OY', 'AH', 'ER', 'EY', 'AO', 'AW', 'AY', 'EH', 'AE', 'UH', 'IY', 'AA', 'OW'}
    co_list={'W', 'K', 'HH', 'G', 'JH', 'Z', 'Y', 'N', 'V', 'SH', 'L', 'NG', 'S', 'CH', 'R', 'D', 'B', 'TH', 'F', 'DH', 'T', 'P', 'M', 'ZH'}
    strong_suffixes = {'al', 'ance', 'ancy', 'ant', 'ard', 'ary', 'àte', 'auto', 'ence', 'ency', 'ent','ery', 'est', 'ial', 'ian', 'iana', 'en', 'ésce', 'ic', 'ify', 'ine', 'ion', 'tion',
                           'ity', 'ive', 'ory', 'ous', 'ual', 'ure', 'wide', 'y', 'se', 'ade', 'e', 'ee', 'een','eer', 'ese', 'esque', 'ette', 'eur', 'ier', 'oon', 'que'}

    strong_prefixes = {'ad', 'co', 'con', 'counter', 'de', 'di', 'dis', 'e', 'en', 'ex', 'in', 'mid', 'ob', 'para',
                           'pre', 're', 'sub',
                           'a', 'be', 'with', 'for'}

    neutral_prefixes = {'down', 'fore', 'mis', 'over', 'out', 'un', 'under', 'up', 'anti', 'bi', 'non', 'pro', 'tri',
                           'contra', 'counta',
                           'de', 'dis', 'extra', 'inter', 'intro', 'multi', 'non', 'post', 'retro', 'super', 'trans',
                           'ultra'}

    neutral_suffixes = {'able', 'age', 'al', 'ate', 'ed', 'en', 'er', 'est', 'ful', 'hood', 'ible', 'ing', 'ile', 'ish', 'ism',
         'ist', 'ize', 'less', 'like', 'ly''man', 'ment', 'most', 'ness', 'old', 's', 'ship', 'some', 'th', 'ward',
         'wise', 'y'}

    suffixes = {
        'inal', 'ain', 'tion', 'sion', 'osis', 'oon', 'sce', 'que', 'ette', 'eer', 'ee', 'aire', 'able', 'ible', 'acy',
        'cy', 'ade',
        'age', 'al', 'al', 'ial', 'ical', 'an', 'ance', 'ence',
        'ancy', 'ency', 'ant', 'ent', 'ant', 'ent', 'ient', 'ar', 'ary', 'ard', 'art', 'ate', 'ate', 'ate', 'ation',
        'cade',
        'drome', 'ed', 'ed', 'en', 'en', 'ence', 'ency', 'er', 'ier',
        'er', 'or', 'er', 'or', 'ery', 'es', 'ese', 'ies', 'es', 'ies', 'ess', 'est', 'iest', 'fold', 'ful', 'ful',
        'fy', 'ia',
        'ian', 'iatry', 'ic', 'ic', 'ice', 'ify', 'ile',
        'ing', 'ion', 'ish', 'ism', 'ist', 'ite', 'ity', 'ive', 'ive', 'ative', 'itive', 'ize', 'less', 'ly', 'ment',
        'ness',
        'or', 'ory', 'ous', 'eous', 'ose', 'ious', 'ship', 'ster',
        'ure', 'ward', 'wise', 'ize', 'phy', 'ogy'}

    prefixes = {
        'ac', 'ad', 'af', 'ag', 'al', 'an', 'ap', 'as', 'at', 'an', 'ab', 'abs', 'acer', 'acid', 'acri', 'act', 'ag',
        'acu',
        'aer', 'aero', 'ag', 'agi',
        'ig', 'act', 'agri', 'agro', 'alb', 'albo', 'ali', 'allo', 'alter', 'alt', 'am', 'ami', 'amor', 'ambi', 'ambul',
        'ana',
        'ano', 'andr', 'andro', 'ang',
        'anim', 'ann', 'annu', 'enni', 'ante', 'anthrop', 'anti', 'ant', 'anti', 'antico', 'apo', 'ap', 'aph', 'aqu',
        'arch',
        'aster', 'astr', 'auc', 'aug',
        'aut', 'aud', 'audi', 'aur', 'aus', 'aug', 'auc', 'aut', 'auto', 'bar', 'be', 'belli', 'bene', 'bi', 'bine',
        'bibl',
        'bibli', 'biblio', 'bio', 'bi',
        'brev', 'cad', 'cap', 'cas', 'ceiv', 'cept', 'capt', 'cid', 'cip', 'cad', 'cas', 'calor', 'capit', 'capt',
        'carn',
        'cat', 'cata', 'cath', 'caus', 'caut'
        , 'cause', 'cuse', 'cus', 'ceas', 'ced', 'cede', 'ceed', 'cess', 'cent', 'centr', 'centri', 'chrom', 'chron',
        'cide',
        'cis', 'cise', 'circum', 'cit',
        'civ', 'clam', 'claim', 'clin', 'clud', 'clus claus', 'co', 'cog', 'col', 'coll', 'con', 'com', 'cor', 'cogn',
        'gnos',
        'com', 'con', 'contr', 'contra',
        'counter', 'cord', 'cor', 'cardi', 'corp', 'cort', 'cosm', 'cour', 'cur', 'curr', 'curs', 'crat', 'cracy',
        'cre',
        'cresc', 'cret', 'crease', 'crea',
        'cred', 'cresc', 'cret', 'crease', 'cru', 'crit', 'cur', 'curs', 'cura', 'cycl', 'cyclo', 'de', 'dec', 'deca',
        'dec',
        'dign', 'dei', 'div', 'dem', 'demo',
        'dent', 'dont', 'derm', 'di', 'dy', 'dia', 'dic', 'dict', 'dit', 'dis', 'dif', 'dit', 'doc', 'doct', 'domin',
        'don',
        'dorm', 'dox', 'duc', 'duct', 'dura',
        'dynam', 'dys', 'ec', 'eco', 'ecto', 'en', 'em', 'end', 'epi', 'equi', 'erg', 'ev', 'et', 'ex', 'exter',
        'extra',
        'extro', 'fa', 'fess', 'fac', 'fact',
        'fec', 'fect', 'fic', 'fas', 'fea', 'fall', 'fals', 'femto', 'fer', 'fic', 'feign', 'fain', 'fit', 'feat',
        'fid', 'fid',
        'fide', 'feder', 'fig', 'fila',
        'fili', 'fin', 'fix', 'flex', 'flect', 'flict', 'flu', 'fluc', 'fluv', 'flux', 'for', 'fore', 'forc', 'fort',
        'form',
        'fract', 'frag',
        'frai', 'fuge', 'fuse', 'gam', 'gastr', 'gastro', 'gen', 'gen', 'geo', 'germ', 'gest', 'giga', 'gin', 'gloss',
        'glot',
        'glu', 'glo', 'gor', 'grad', 'gress',
        'gree', 'graph', 'gram', 'graf', 'grat', 'grav', 'greg', 'hale', 'heal', 'helio', 'hema', 'hemo', 'her', 'here',
        'hes',
        'hetero', 'hex', 'ses', 'sex', 'homo',
        'hum', 'human', 'hydr', 'hydra', 'hydro', 'hyper', 'hypn', 'an', 'ics', 'ignis', 'in', 'im', 'in', 'im', 'il',
        'ir',
        'infra', 'inter', 'intra', 'intro', 'ty',
        'jac', 'ject', 'join', 'junct', 'judice', 'jug', 'junct', 'just', 'juven', 'labor', 'lau', 'lav', 'lot', 'lut',
        'lect',
        'leg', 'lig', 'leg', 'levi', 'lex',
        'leag', 'leg', 'liber', 'liver', 'lide', 'liter', 'loc', 'loco', 'log', 'logo', 'ology', 'loqu', 'locut', 'luc',
        'lum',
        'lun', 'lus', 'lust', 'lude', 'macr',
        'macer', 'magn', 'main', 'mal', 'man', 'manu', 'mand', 'mania', 'mar', 'mari', 'mer', 'matri', 'medi', 'mega',
        'mem',
        'ment', 'meso', 'meta', 'meter', 'metr',
        'micro', 'migra', 'mill', 'kilo', 'milli', 'min', 'mis', 'mit', 'miss', 'mob', 'mov', 'mot', 'mon', 'mono',
        'mor',
        'mort', 'morph', 'multi', 'nano', 'nasc',
        'nat', 'gnant', 'nai', 'nat', 'nasc', 'neo', 'neur', 'nom', 'nom', 'nym', 'nomen', 'nomin', 'non', 'non', 'nov',
        'nox',
        'noc', 'numer', 'numisma', 'ob', 'oc',
        'of', 'op', 'oct', 'oligo', 'omni', 'onym', 'oper', 'ortho', 'over', 'pac', 'pair', 'pare', 'paleo', 'pan',
        'para',
        'pat', 'pass', 'path', 'pater', 'patr',
        'path', 'pathy', 'ped', 'pod', 'pedo', 'pel', 'puls', 'pend', 'pens', 'pond', 'per', 'peri', 'phage', 'phan',
        'phas',
        'phen', 'fan', 'phant', 'fant', 'phe',
        'phil', 'phlegma', 'phobia', 'phobos', 'phon', 'phot', 'photo', 'pico', 'pict', 'plac', 'plais', 'pli', 'ply',
        'plore',
        'plu', 'plur', 'plus', 'pneuma',
        'pneumon', 'pod', 'poli', 'poly', 'pon', 'pos', 'pound', 'pop', 'port', 'portion', 'post', 'pot', 'pre', 'pur',
        'prehendere', 'prin', 'prim', 'prime',
        'pro', 'proto', 'psych', 'punct', 'pute', 'quat', 'quad', 'quint', 'penta', 'quip', 'quir', 'quis', 'quest',
        'quer',
        're', 'reg', 'recti', 'retro', 'ri', 'ridi',
        'risi', 'rog', 'roga', 'rupt', 'sacr', 'sanc', 'secr', 'salv', 'salu', 'sanct', 'sat', 'satis', 'sci', 'scio',
        'scientia', 'scope', 'scrib', 'script', 'se',
        'sect', 'sec', 'sed', 'sess', 'sid', 'semi', 'sen', 'scen', 'sent', 'sens', 'sept', 'sequ', 'secu', 'sue',
        'serv',
        'sign', 'signi', 'simil', 'simul', 'sist', 'sta',
        'stit', 'soci', 'sol', 'solus', 'solv', 'solu', 'solut', 'somn', 'soph', 'spec', 'spect', 'spi', 'spic', 'sper',
        'sphere', 'spir', 'stand', 'stant', 'stab',
        'stat', 'stan', 'sti', 'sta', 'st', 'stead', 'strain', 'strict', 'string', 'stige', 'stru', 'struct', 'stroy',
        'stry',
        'sub', 'suc', 'suf', 'sup', 'sur', 'sus',
        'sume', 'sump', 'super', 'supra', 'syn', 'sym', 'tact', 'tang', 'tag', 'tig', 'ting', 'tain', 'ten', 'tent',
        'tin',
        'tect', 'teg', 'tele', 'tem', 'tempo', 'ten',
        'tin', 'tain', 'tend', 'tent', 'tens', 'tera', 'term', 'terr', 'terra', 'test', 'the', 'theo', 'therm',
        'thesis',
        'thet', 'tire', 'tom', 'tor', 'tors', 'tort'
        , 'tox', 'tract', 'tra', 'trai', 'treat', 'trans', 'tri', 'trib', 'tribute', 'turbo', 'typ', 'ultima', 'umber',
        'umbraticum', 'un', 'uni', 'vac', 'vade', 'vale',
        'vali', 'valu', 'veh', 'vect', 'ven', 'vent', 'ver', 'veri', 'verb', 'verv', 'vert', 'vers', 'vi', 'vic',
        'vicis',
        'vict', 'vinc', 'vid', 'vis', 'viv', 'vita', 'vivi'
        , 'voc', 'voke', 'vol', 'volcan', 'volv', 'volt', 'vol', 'vor', 'with', 'zo'}
    neutral_prefixes = upper(neutral_prefixes)
    neutral_suffixes = upper(neutral_suffixes)
    strong_prefixes = upper(strong_prefixes)
    strong_suffixes = upper(strong_suffixes)
    full_suffixes_set = upper(suffixes)
    full_prefixes_set = upper(prefixes)
    suffix={"1","2","0"}
    for line in data_list:
        dict={}
        vow_index=[]
        vowelCount =0
        pattern=""
        y=""
        dict["pos"]=nltk.pos_tag([line.split(":")[0]])[0][1]
        word=line.split(":")[0]
        temp= check_prefix(word, neutral_prefixes)
        if temp:
            dict['neu_pre']=temp
        temp = check_suffix(word, neutral_suffixes)
        if temp:
            dict['neu_suf']=temp
        temp = check_prefix(word,strong_prefixes)
        if temp:
            dict['str_pre']=temp
        temp = check_suffix(word,strong_suffixes)
        if temp:
            dict['str_suf']=temp
        temp = check_prefix(word,full_suffixes_set)
        if temp:
            dict['ful_pre']=temp
        temp = check_suffix(word,full_prefixes_set)
        if temp:
            dict['ful_suf']=temp
        line = line.split(":")[1].strip()

        syllables = line.split(" ")
        l=[]
        for i in syllables:
            l.append(i if not(i[-1].isdigit()) else i[:-1] )
        dict.update(Counter({''.join(i) for i in get_ngrams(l)}))
        dict['len'] = len(syllables)
        out=''
        for i in range(len(syllables)):
            syl=syllables[i]

            if syl[-1] in suffix:
                vowelCount+=1
                vow_index.append(i)
                out+=syl[-1]
                # if syl[-1]=="1":
                #     model_y.append(vowelCount)
                pattern+="V"
            else:
                pattern+="C"

        model_y.append(out)
        vowelCount=0
        dict["pattern"]=pattern
        dict['vow_len']=len(vow_index)
        for i in vow_index:
            vowelCount += 1
            if i - 1 >= 0:
                dict["onset2_" + str(vowelCount)] = syllables[i - 1]
            if i + 1 < len(syllables):
                dict["coda1_" + str(vowelCount)] = syllables[i + 1]
            dict["nucleus_" + str(vowelCount)] = syllables[i][:-1]
        model_x.append(dict)
    # print(pd.DataFrame(model_x))
    # print(model_y)
    v = DictVectorizer(sparse=True)


    X = v.fit_transform(model_x)
    classifier = LogisticRegression(penalty='l2', class_weight='balanced')

    classifier.fit(X, model_y)
    with open(classifier_file, 'wb') as f:
        pickle.dump(classifier, f)
        pickle.dump(v, f)




def test(data, classifier_file):  # do not change the heading of the function
    data_list = data
    with open(classifier_file, 'rb') as f:
        clf = pickle.load(f)
        v = pickle.load(f)
    model_x=[]
    model_y=[]
    vo_list={'IH', 'UW', 'OY', 'AH', 'ER', 'EY', 'AO', 'AW', 'AY', 'EH', 'AE', 'UH', 'IY', 'AA', 'OW'}
    co_list={'W', 'K', 'HH', 'G', 'JH', 'Z', 'Y', 'N', 'V', 'SH', 'L', 'NG', 'S', 'CH', 'R', 'D', 'B', 'TH', 'F', 'DH', 'T', 'P', 'M', 'ZH'}
    strong_suffixes = {'al', 'ance', 'ancy', 'ant', 'ard', 'ary', 'àte', 'auto', 'ence', 'ency', 'ent',
                           'ery', 'est', 'ial', 'ian', 'iana', 'en', 'ésce', 'ic', 'ify', 'ine', 'ion', 'tion',
                           'ity', 'ive', 'ory', 'ous', 'ual', 'ure', 'wide', 'y', 'se', 'ade', 'e', 'ee', 'een',
                           'eer', 'ese', 'esque', 'ette', 'eur', 'ier', 'oon', 'que'}

    strong_prefixes = {'ad', 'co', 'con', 'counter', 'de', 'di', 'dis', 'e', 'en', 'ex', 'in', 'mid', 'ob', 'para',
                           'pre', 're', 'sub',
                           'a', 'be', 'with', 'for'}

    neutral_prefixes = {'down', 'fore', 'mis', 'over', 'out', 'un', 'under', 'up', 'anti', 'bi', 'non', 'pro', 'tri',
                           'contra', 'counta',
                           'de', 'dis', 'extra', 'inter', 'intro', 'multi', 'non', 'post', 'retro', 'super', 'trans',
                           'ultra'}

    neutral_suffixes = {'able', 'age', 'al', 'ate', 'ed', 'en', 'er', 'est', 'ful', 'hood', 'ible', 'ing', 'ile', 'ish', 'ism',
         'ist', 'ize', 'less', 'like', 'ly''man', 'ment', 'most', 'ness', 'old', 's', 'ship', 'some', 'th', 'ward',
         'wise', 'y'}

    suffixes = {
        'inal', 'ain', 'tion', 'sion', 'osis', 'oon', 'sce', 'que', 'ette', 'eer', 'ee', 'aire', 'able', 'ible', 'acy',
        'cy', 'ade',
        'age', 'al', 'al', 'ial', 'ical', 'an', 'ance', 'ence',
        'ancy', 'ency', 'ant', 'ent', 'ant', 'ent', 'ient', 'ar', 'ary', 'ard', 'art', 'ate', 'ate', 'ate', 'ation',
        'cade',
        'drome', 'ed', 'ed', 'en', 'en', 'ence', 'ency', 'er', 'ier',
        'er', 'or', 'er', 'or', 'ery', 'es', 'ese', 'ies', 'es', 'ies', 'ess', 'est', 'iest', 'fold', 'ful', 'ful',
        'fy', 'ia',
        'ian', 'iatry', 'ic', 'ic', 'ice', 'ify', 'ile',
        'ing', 'ion', 'ish', 'ism', 'ist', 'ite', 'ity', 'ive', 'ive', 'ative', 'itive', 'ize', 'less', 'ly', 'ment',
        'ness',
        'or', 'ory', 'ous', 'eous', 'ose', 'ious', 'ship', 'ster',
        'ure', 'ward', 'wise', 'ize', 'phy', 'ogy'}

    prefixes = {
        'ac', 'ad', 'af', 'ag', 'al', 'an', 'ap', 'as', 'at', 'an', 'ab', 'abs', 'acer', 'acid', 'acri', 'act', 'ag',
        'acu',
        'aer', 'aero', 'ag', 'agi',
        'ig', 'act', 'agri', 'agro', 'alb', 'albo', 'ali', 'allo', 'alter', 'alt', 'am', 'ami', 'amor', 'ambi', 'ambul',
        'ana',
        'ano', 'andr', 'andro', 'ang',
        'anim', 'ann', 'annu', 'enni', 'ante', 'anthrop', 'anti', 'ant', 'anti', 'antico', 'apo', 'ap', 'aph', 'aqu',
        'arch',
        'aster', 'astr', 'auc', 'aug',
        'aut', 'aud', 'audi', 'aur', 'aus', 'aug', 'auc', 'aut', 'auto', 'bar', 'be', 'belli', 'bene', 'bi', 'bine',
        'bibl',
        'bibli', 'biblio', 'bio', 'bi',
        'brev', 'cad', 'cap', 'cas', 'ceiv', 'cept', 'capt', 'cid', 'cip', 'cad', 'cas', 'calor', 'capit', 'capt',
        'carn',
        'cat', 'cata', 'cath', 'caus', 'caut'
        , 'cause', 'cuse', 'cus', 'ceas', 'ced', 'cede', 'ceed', 'cess', 'cent', 'centr', 'centri', 'chrom', 'chron',
        'cide',
        'cis', 'cise', 'circum', 'cit',
        'civ', 'clam', 'claim', 'clin', 'clud', 'clus claus', 'co', 'cog', 'col', 'coll', 'con', 'com', 'cor', 'cogn',
        'gnos',
        'com', 'con', 'contr', 'contra',
        'counter', 'cord', 'cor', 'cardi', 'corp', 'cort', 'cosm', 'cour', 'cur', 'curr', 'curs', 'crat', 'cracy',
        'cre',
        'cresc', 'cret', 'crease', 'crea',
        'cred', 'cresc', 'cret', 'crease', 'cru', 'crit', 'cur', 'curs', 'cura', 'cycl', 'cyclo', 'de', 'dec', 'deca',
        'dec',
        'dign', 'dei', 'div', 'dem', 'demo',
        'dent', 'dont', 'derm', 'di', 'dy', 'dia', 'dic', 'dict', 'dit', 'dis', 'dif', 'dit', 'doc', 'doct', 'domin',
        'don',
        'dorm', 'dox', 'duc', 'duct', 'dura',
        'dynam', 'dys', 'ec', 'eco', 'ecto', 'en', 'em', 'end', 'epi', 'equi', 'erg', 'ev', 'et', 'ex', 'exter',
        'extra',
        'extro', 'fa', 'fess', 'fac', 'fact',
        'fec', 'fect', 'fic', 'fas', 'fea', 'fall', 'fals', 'femto', 'fer', 'fic', 'feign', 'fain', 'fit', 'feat',
        'fid', 'fid',
        'fide', 'feder', 'fig', 'fila',
        'fili', 'fin', 'fix', 'flex', 'flect', 'flict', 'flu', 'fluc', 'fluv', 'flux', 'for', 'fore', 'forc', 'fort',
        'form',
        'fract', 'frag',
        'frai', 'fuge', 'fuse', 'gam', 'gastr', 'gastro', 'gen', 'gen', 'geo', 'germ', 'gest', 'giga', 'gin', 'gloss',
        'glot',
        'glu', 'glo', 'gor', 'grad', 'gress',
        'gree', 'graph', 'gram', 'graf', 'grat', 'grav', 'greg', 'hale', 'heal', 'helio', 'hema', 'hemo', 'her', 'here',
        'hes',
        'hetero', 'hex', 'ses', 'sex', 'homo',
        'hum', 'human', 'hydr', 'hydra', 'hydro', 'hyper', 'hypn', 'an', 'ics', 'ignis', 'in', 'im', 'in', 'im', 'il',
        'ir',
        'infra', 'inter', 'intra', 'intro', 'ty',
        'jac', 'ject', 'join', 'junct', 'judice', 'jug', 'junct', 'just', 'juven', 'labor', 'lau', 'lav', 'lot', 'lut',
        'lect',
        'leg', 'lig', 'leg', 'levi', 'lex',
        'leag', 'leg', 'liber', 'liver', 'lide', 'liter', 'loc', 'loco', 'log', 'logo', 'ology', 'loqu', 'locut', 'luc',
        'lum',
        'lun', 'lus', 'lust', 'lude', 'macr',
        'macer', 'magn', 'main', 'mal', 'man', 'manu', 'mand', 'mania', 'mar', 'mari', 'mer', 'matri', 'medi', 'mega',
        'mem',
        'ment', 'meso', 'meta', 'meter', 'metr',
        'micro', 'migra', 'mill', 'kilo', 'milli', 'min', 'mis', 'mit', 'miss', 'mob', 'mov', 'mot', 'mon', 'mono',
        'mor',
        'mort', 'morph', 'multi', 'nano', 'nasc',
        'nat', 'gnant', 'nai', 'nat', 'nasc', 'neo', 'neur', 'nom', 'nom', 'nym', 'nomen', 'nomin', 'non', 'non', 'nov',
        'nox',
        'noc', 'numer', 'numisma', 'ob', 'oc',
        'of', 'op', 'oct', 'oligo', 'omni', 'onym', 'oper', 'ortho', 'over', 'pac', 'pair', 'pare', 'paleo', 'pan',
        'para',
        'pat', 'pass', 'path', 'pater', 'patr',
        'path', 'pathy', 'ped', 'pod', 'pedo', 'pel', 'puls', 'pend', 'pens', 'pond', 'per', 'peri', 'phage', 'phan',
        'phas',
        'phen', 'fan', 'phant', 'fant', 'phe',
        'phil', 'phlegma', 'phobia', 'phobos', 'phon', 'phot', 'photo', 'pico', 'pict', 'plac', 'plais', 'pli', 'ply',
        'plore',
        'plu', 'plur', 'plus', 'pneuma',
        'pneumon', 'pod', 'poli', 'poly', 'pon', 'pos', 'pound', 'pop', 'port', 'portion', 'post', 'pot', 'pre', 'pur',
        'prehendere', 'prin', 'prim', 'prime',
        'pro', 'proto', 'psych', 'punct', 'pute', 'quat', 'quad', 'quint', 'penta', 'quip', 'quir', 'quis', 'quest',
        'quer',
        're', 'reg', 'recti', 'retro', 'ri', 'ridi',
        'risi', 'rog', 'roga', 'rupt', 'sacr', 'sanc', 'secr', 'salv', 'salu', 'sanct', 'sat', 'satis', 'sci', 'scio',
        'scientia', 'scope', 'scrib', 'script', 'se',
        'sect', 'sec', 'sed', 'sess', 'sid', 'semi', 'sen', 'scen', 'sent', 'sens', 'sept', 'sequ', 'secu', 'sue',
        'serv',
        'sign', 'signi', 'simil', 'simul', 'sist', 'sta',
        'stit', 'soci', 'sol', 'solus', 'solv', 'solu', 'solut', 'somn', 'soph', 'spec', 'spect', 'spi', 'spic', 'sper',
        'sphere', 'spir', 'stand', 'stant', 'stab',
        'stat', 'stan', 'sti', 'sta', 'st', 'stead', 'strain', 'strict', 'string', 'stige', 'stru', 'struct', 'stroy',
        'stry',
        'sub', 'suc', 'suf', 'sup', 'sur', 'sus',
        'sume', 'sump', 'super', 'supra', 'syn', 'sym', 'tact', 'tang', 'tag', 'tig', 'ting', 'tain', 'ten', 'tent',
        'tin',
        'tect', 'teg', 'tele', 'tem', 'tempo', 'ten',
        'tin', 'tain', 'tend', 'tent', 'tens', 'tera', 'term', 'terr', 'terra', 'test', 'the', 'theo', 'therm',
        'thesis',
        'thet', 'tire', 'tom', 'tor', 'tors', 'tort'
        , 'tox', 'tract', 'tra', 'trai', 'treat', 'trans', 'tri', 'trib', 'tribute', 'turbo', 'typ', 'ultima', 'umber',
        'umbraticum', 'un', 'uni', 'vac', 'vade', 'vale',
        'vali', 'valu', 'veh', 'vect', 'ven', 'vent', 'ver', 'veri', 'verb', 'verv', 'vert', 'vers', 'vi', 'vic',
        'vicis',
        'vict', 'vinc', 'vid', 'vis', 'viv', 'vita', 'vivi'
        , 'voc', 'voke', 'vol', 'volcan', 'volv', 'volt', 'vol', 'vor', 'with', 'zo'}
    neutral_prefixes = upper(neutral_prefixes)
    neutral_suffixes = upper(neutral_suffixes)
    strong_prefixes = upper(strong_prefixes)
    strong_suffixes = upper(strong_suffixes)
    full_suffixes_set = upper(suffixes)
    full_prefixes_set = upper(prefixes)


    for line in data_list:
        dict={}
        vow_index=[]
        vowelCount =0
        pattern=""
        dict["pos"]=nltk.pos_tag([line.split(":")[0]])[0][1]
        word=line.split(":")[0]
        temp= check_prefix(word, neutral_prefixes)
        if temp:
            dict['neu_pre']=temp
        temp = check_suffix(word, neutral_suffixes)
        if temp:
            dict['neu_suf']=temp
        temp = check_prefix(word,strong_prefixes)
        if temp:
            dict['str_pre']=temp
        temp = check_suffix(word,strong_suffixes)
        if temp:
            dict['str_suf']=temp
        temp = check_prefix(word,full_suffixes_set)
        if temp:
            dict['ful_pre']=temp
        temp = check_suffix(word,full_prefixes_set)
        if temp:
            dict['ful_suf']=temp
        line = line.split(":")[1].strip()

        syllables = line.split(" ")
        l=[]
        for i in syllables:
            l.append(i if not (i[-1].isdigit()) else i[:-1])
        dict.update(Counter({''.join(i) for i in get_ngrams(l)}))
        dict['len'] = len(syllables)
        for i in range(len(syllables)):
            syl=syllables[i]

            if syl in vo_list:
                vowelCount+=1
                vow_index.append(i)

                pattern+="V"
            else:
                pattern+="C"



        vowelCount=0
        dict['vow_len']=len(vow_index)
        dict["pattern"]=pattern
        for i in vow_index:
            vowelCount += 1
            if i - 1 >= 0:
                dict["onset2_" + str(vowelCount)] = syllables[i - 1]
            if i + 1 < len(syllables):
                dict["coda1_" + str(vowelCount)] = syllables[i + 1]
            dict["nucleus_" + str(vowelCount)] = syllables[i]
        model_x.append(dict)
    # print(pd.DataFrame(model_x))
    # print(model_y)
    x=v.transform(model_x)
    re = clf.predict(x).tolist()
    res = []
    for i in re:
        c = 0
        for j in i:
            c += 1
            if j == "1":
                res.append(c)
    return res

# import helper
# train(helper.read_data("training_data.txt"),"dump.dat")
# res=test(helper.read_data("syl.txt"),"dump.dat")
#
# l=[]
# with open("res.txt")as f:
# 	for line in f:
# 		line=line.strip()
# 		l.append(int(line))
#
# print(f1_score(l,res,average='macro'))