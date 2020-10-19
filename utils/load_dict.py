def load_dict(dictFile):
    '''
    read lexicon from files
    args:
        dictFile: dictionary file with key-value pairs
    returns:
        lexicon: dict with key-value pairs
    '''
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])
    print('total words/phones',len(lexicon))
    return lexicon
