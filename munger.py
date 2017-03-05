import pandas as pd
import numpy as np
import datetime as dt

import impute_into_df

def munge_dates(df):
    '''
    INPUT dataframe
    OUTPUT dataframe
    creates new fields (columns) indicating day of week and month of year in
    which events are created, etc, as well as time differences between dates
    posted, dates of event, and approx_payout_date.
    New fields created:
    user_create_dow         day of week user created
    user_create_moy         month of year user created
    event_create_dow            day of week event created
    event_create_moy            month of year event created
    event_pub_dow           day of week event published
    event_pub_moy           month of year event published
    diff_payout_event_pub       approx_payout_date - event_published
    diff_payout_event_create        approx_payout_date - event_created
    diff_event_created_user_create  event_created - user_created
    diff_event_pub_user_create      event_published - user_created
    diff_event_pub_event_create     event_published - event_crfeated
    '''

    dt1970 = dt.datetime(1970, 1, 1)

    # Hard-coded stuffs:
    days_of_week = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    months_of_year = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # First the day of week and month of year columns:

    tsec = df['user_created']
    tdatetimes = [dt1970 + dt.timedelta(days = int(tsec/86400.0))
                 for tsec in tsec]
    df['user_create_dow'] = [d.strftime('%a') for d in tdatetimes]
    df['user_create_moy'] = [d.strftime('%b') for d in tdatetimes]

    tsec = df['event_created']
    tdatetimes = [dt1970 + dt.timedelta(days = int(tsec/86400.0))
                 for tsec in tsec]
    df['event_create_dow'] = [d.strftime('%a') for d in tdatetimes]
    df['event_create_moy'] = [d.strftime('%b') for d in tdatetimes]

    tsec = df['event_published']
    tdatetimes = [dt1970 + dt.timedelta(days = int(tsec/86400.0))
                 for tsec in tsec]
    df['event_pub_dow'] = [d.strftime('%a') for d in tdatetimes]
    df['event_pub_moy'] = [d.strftime('%b') for d in tdatetimes]

    # Now the differences:

    df['diff_payout_event_pub'] = [int(a + 0.5) for a in
                                   (df['approx_payout_date'].astype(float)
                                    - df['event_published'])/86400.0]
    df['diff_payout_event_create'] = [int(a + 0.5) for a in
                                      (df['approx_payout_date'].astype(float)
                                       - df['event_created'])/86400.0]
    df['diff_event_create_user_create'] = [int(a + 0.5) for a in
                                           (df['event_created']
                                            - df['user_created'])/86400.0]
    df['diff_event_pub_user_create'] = [int(a + 0.5) for a in
                                        (df['event_published']
                                         - df['user_created'])/86400.0]
    df['diff_event_pub_event_create'] = [int(a + 0.5) for a in
                                         (df['event_published']
                                          - df['event_created'])/86400.0]

    # Dummify days of week and months of year variables:

    df = pd.concat([df, pd.get_dummies(df['user_create_dow'])], axis=1)

    cc_rename = {}
    for c in days_of_week:
        cc_rename[c] = 'uc_' + c
        if c not in df:
            df[c] = pd.Series([0 for x in range(len(df.index))])
    df.rename(columns = cc_rename, inplace=True)
    del df['user_create_dow']

    df = pd.concat([df, pd.get_dummies(df['user_create_moy'])], axis=1)

    cc_rename = {}
    for c in months_of_year:
        cc_rename[c] = 'uc_' + c
        if c not in df:
            df[c] = pd.Series([0 for x in range(len(df.index))])
    df.rename(columns = cc_rename, inplace=True)
    del df['user_create_moy']

    df = pd.concat([df, pd.get_dummies(df['event_create_dow'])], axis=1)

    cc_rename = {}
    for c in days_of_week:
        cc_rename[c] = 'ec_' + c
        if c not in df:
            df[c] = pd.Series([0 for x in range(len(df.index))])
    df.rename(columns = cc_rename, inplace=True)
    del df['event_create_dow']

    df = pd.concat([df, pd.get_dummies(df['event_create_moy'])], axis=1)

    cc_rename = {}
    for c in months_of_year:
        cc_rename[c] = 'ec_' + c
        if c not in df:
            df[c] = pd.Series([0 for x in range(len(df.index))])
    df.rename(columns = cc_rename, inplace=True)
    del df['event_create_moy']

    df = pd.concat([df, pd.get_dummies(df['event_pub_dow'])], axis=1)

    cc_rename = {}
    for c in days_of_week:
        cc_rename[c] = 'ep_' + c
        if c not in df:
            df[c] = pd.Series([0 for x in range(len(df.index))])
    df.rename(columns = cc_rename, inplace=True)
    del df['event_pub_dow']

    df = pd.concat([df, pd.get_dummies(df['event_pub_moy'])], axis=1)

    cc_rename = {}
    for c in months_of_year:
        cc_rename[c] = 'ep_' + c
        if c not in df:
            df[c] = pd.Series([0 for x in range(len(df.index))])
    df.rename(columns = cc_rename, inplace=True)
    del df['event_pub_moy']

    return df

if __name__ == "__main__":
    df = pd.read_json('../../fraud_data/train_new.json')
    df = impute_into_df.impute_into_df(df)
    df = munge_dates(df)
    print df.describe().T
