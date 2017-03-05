import pandas as pd
import numpy as np

#def impute_into_df(df, country_codes, currencies, all_countries=False):
def impute_into_df(df, all_countries=False):
    '''
    INPUT: Pandas dataframe
           country_codes    list of 2-letter country codes spanning all
                                /expected/ cases.
           all_countries    bool, creates categories for all of 239 country
                codes, rather than just those in the
                country_codes list.
    OUTPUT: dataframe
    Returns dataframe with values imputed for missing:
        country --> 'MI'
        currency --> 'Missing'
        delivery_method --> -1.0
        event_published --> event_created - 1 day
        has_header --> -1.0
        org_facebook --> -1.0
        org_twitter --> -1.0
        sale_duration --> -1.0
        sale_duration2 --> -1
        venue_country --> 'MI'
        venue_latitude --> -90.0
        venue_longigute --> 0.0
        venue_name --> 'Missing'
        venue_state --> 'Missing'
    '''

    # Bad, hard-coded stuff:
    country_codes = [ 'BE', 'BG', 'BB', 'JP', 'JM', 'BR', 'BS', 'JE', 'RU',
                      'RS', 'RE', 'RO', 'GR', 'GB', 'GH', 'OM', 'HR', 'HT',
                      'HU', 'HK', 'PR', 'PS', 'PT', 'UY', 'PE', 'PK', 'PH',
                      'PL', 'ZA', 'EC', 'ES', 'MA', 'MC', 'MI', 'US', 'IM',
                      'MY', 'MX', 'IL', 'FR', 'FI', 'NI', 'NL', 'NO', 'NA',
                      'NG', 'NZ', 'CI', 'CH', 'CO', 'CN', 'CM', 'CA', 'CZ',
                      'CY', 'CR', 'KE', 'KH', 'SK', 'SI', 'SG', 'SE', 'DO',
                      'DK', 'DE', 'DZ', 'LB', 'TT', 'TR', 'A1', 'LU', 'TJ',
                      'TH', 'AE', 'VE', 'VI', 'IS', 'IT', 'VN', 'AR', 'AU',
                      'AT', 'IN', 'IE', 'ID', 'QA']

    currencies = ['AUD', 'CAD', 'EUR', 'GBP', 'MXN', 'NZD', 'USD']

    # End bad

    df['country'] = df['country'].apply(lambda x: 'MI' if (x == '' or x is None) else x)
    df['currency'] = df['currency'].apply(lambda x: 'Missing' if (x == '' or x is None) else x)
    df['delivery_method'] = df['delivery_method'].fillna(-1.0)
    df['has_header'] = df['has_header'].fillna(-1.0)
    df['org_facebook'] = df['org_facebook'].fillna(-1.0)
    df['org_twitter'] = df['org_twitter'].fillna(-1.0)

    # Should remove negative sale_duration and sale_duration2 values? Or set
    # all to -1?
    df['sale_duration'] = df['sale_duration'].fillna(-1.0)
    df['sale_duration2'] = df['sale_duration2'].apply(lambda x: -1 if x < 0 else x)
    df['venue_country'] = df['venue_country'].apply(lambda x: 'MI' if (x == '' or x is None) else x)
    df['venue_latitude'] = df['venue_latitude'].fillna(-90.0)
    df['venue_longitude'] = df['venue_longitude'].fillna(0.0)
    df['venue_name'] = df['venue_name'].apply(lambda x: 'MI' if (x == '' or x is None) else x)
    df['venue_state'] = df['venue_state'].apply(lambda x: 'missing' if (x == '' or x is None) else x)

    df['event_published'] = df.apply(lambda x: x['event_created'] - 86400.0 if (np.isnan(x['event_published']) or x['event_published'] == 0.0) else x['event_published'], axis=1)

    # Change countries to dummy variables
    all_codes = ['AF', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG',
                 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BS', 'BH', 'BD',
                 'BB', 'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BA',
                 'BW', 'BV', 'BR', 'IO', 'BN', 'BG', 'BF', 'BI', 'KH',
                 'CM', 'CA', 'CV', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX',
                 'CC', 'CO', 'KM', 'CG', 'CK', 'CR', 'CI', 'HR', 'CU',
                 'CY', 'CZ', 'DK', 'DJ', 'DM', 'DO', 'TL', 'EC', 'EG',
                 'SV', 'GQ', 'ER', 'EE', 'ET', 'FK', 'FO', 'FJ', 'FI',
                 'FR', 'FX', 'GF', 'PF', 'TF', 'GA', 'GM', 'GE', 'DE',
                 'GH', 'GI', 'GR', 'GL', 'GD', 'GP', 'GU', 'GT', 'GN',
                 'GW', 'GY', 'HT', 'HM', 'HN', 'HK', 'HU', 'IS', 'IN',
                 'ID', 'IR', 'IQ', 'IE', 'IL', 'IT', 'JM', 'JP', 'JO',
                 'KZ', 'KE', 'KI', 'KP', 'KR', 'KW', 'KG', 'LA', 'LV',
                 'LB', 'LS', 'LR', 'LY', 'LI', 'LT', 'LU', 'MO', 'MK',
                 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH', 'MQ', 'MR',
                 'MU', 'YT', 'MX', 'FM', 'MD', 'MC', 'MN', 'MS', 'MA',
                 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL', 'AN', 'NC', 'NZ',
                 'NI', 'NE', 'NG', 'NU', 'NF', 'MP', 'NO', 'OM', 'PK',
                 'PW', 'PA', 'PG', 'PY', 'PE', 'PH', 'PN', 'PL', 'PT',
                 'PR', 'QA', 'RE', 'RO', 'RU', 'RW', 'KN', 'LC', 'VC',
                 'WS', 'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL', 'SG',
                 'SK', 'SI', 'SB', 'SO', 'ZA', 'ES', 'LK', 'SH', 'PM',
                 'SD', 'SR', 'SJ', 'SZ', 'SE', 'CH', 'SY', 'TW', 'TJ',
                 'TZ', 'TH', 'TG', 'TK', 'TO', 'TT', 'TN', 'TR', 'TM',
                 'TC', 'TV', 'UG', 'UA', 'AE', 'GB', 'US', 'UM', 'UY',
                 'UZ', 'VU', 'VA', 'VE', 'VN', 'VG', 'VI', 'WF', 'EH',
                 'YE', 'YU', 'ZR', 'ZM', 'ZW']

#   all_ccs = set(list(set(df.country)) + list(set(df.venue_country)))

    # First dummify for user's country:
#   df = df.join(pd.get_dummies(df['country']))
    df = pd.concat([df, pd.get_dummies(df['country'])], axis=1)
    cc_rename = {}
    for c in list(country_codes):
        cc_rename[c] = 'u_' + c
        if c not in df:
            df[c] = pd.Series([0 for x in range(len(df.index))])

    if all_countries:
        for cc in all_codes:
            if cc not in df:
                cc_rename[c] = 'u_' + cc
                df[cc] = pd.Series([0 for x in range(len(df.index))])

    df.rename(columns = cc_rename, inplace=True)
    del df['country']

    # Repeat for venue_country:
#   df = df.join(pd.get_dummies(df['venue_country']))
    df = pd.concat([df, pd.get_dummies(df['venue_country'])], axis=1)
    cc_rename = {}
    for c in list(country_codes):
        cc_rename[c] = 'v_' + c
        if c not in df:
            df[c] = pd.Series([0 for x in range(len(df.index))])

    if all_countries:
        for cc in all_codes:
            if cc not in df:
                cc_rename[c] = 'v_' + cc
                df[cc] = pd.Series([0 for x in range(len(df.index))])

    df.rename(columns = cc_rename, inplace=True)

    # Convert currencies to dummy variables:
    del df['venue_country']

    if 'currency' in df:
        df = pd.concat([df, pd.get_dummies(df['currency'])], axis=1)
        del df['currency']

    cc_rename = {}
    for c in currencies:
        cc_rename[c] = 'cur_' + c
        if c not in df:
            df[c] = pd.Series([0 for x in range(len(df.index))])

    df.rename(columns = cc_rename, inplace=True)

    return df

if __name__ == "__main__":
    df = pd.read_json('../../fraud_data/train_new.json')
#   df = impute_into_df(df, all_countries=True)
    df = impute_into_df(df)
    print df.describe()
    cols = list(df.columns)
