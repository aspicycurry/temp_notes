import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def check_attr(soup, attr_ls, attr_loc='span', attr_term='style'):
    '''
    INPUT:
        soup -> bs4 object
        
    '''
    attr = set()
    for found in soup.findAll(attr_loc):
        if attr_term in found:
            for style in found[attr_term].split(';'):
                attr.add(style.split(':')[0].strip())
    return dict(zip(attr_ls, [True if x in attr else False for x in attr_ls]))  

def process_html(text, attr_ls):
    soup = BeautifulSoup(text)

    if attr_ls is not None:
        attr = check_attr(soup, attr_ls)
    else:
        attr = dict()

    content = soup.text
    attr['content'] = content
    attr['word_count'] = len(content.split(' '))
    attr['char_length'] = len(content)
    return attr

def html_features(df, col, attr_ls=None):
    #['font-size', 'color', 'backgroud-color']
    df = df.copy()
    series = df.pop(col)
    data = defaultdict(list)
    for row in series:
        attr = process_html(row, attr_ls)
        for k, v in attr.iteritems():
            header = '{0}_{1}'.format(col, k)
            data[header].append(v)
    for k, v in data.iteritems():
        df[k] = v
    return df