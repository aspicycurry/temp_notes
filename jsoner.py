import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from collections import defaultdict
from pandas.tools.plotting import scatter_matrix

def ticket_types_summary(row):
    num = len(row)
    tickets_sold = 0
    quantity_total = 0

    event_id = False
    if len(row) > 0:
    	event_id = row[0]['event_id']
    costs = []
    for tier in row:
        tickets_sold += tier['quantity_sold']
        quantity_total += tier['quantity_total']
        costs.append(tier['cost'] * tier['quantity_sold'])
    costs = np.array(costs)
    return {'num_tiers': num, 'revenue': costs.sum(), 'tickets_sold': tickets_sold, 'quantity_total': quantity_total, 'event_id': event_id}
   

def process_prev_payments(row, att_list = ['num_prev_events', 'user_id', 'name_present','state_present']):
#      """
#      INPUT: An element from the previous_payment columm
#      OUTPUT: A dictionary of features about user and user activity

#      This function returns a dict of features about user activity.
#      """
    # Features
    name_present = False
    state_present = False
    user_id = False
    if len(row) > 0:
        user_id = row[0]['uid']
    num_prev_events = 0

    # Discover features
    if len(row) > 1:
        num_prev_events = len(row)
        for event in  row:
            if event['name'] != '':
                name_present = True
            if event['state'] != '':
                state_present = True
    return {'prev_name_present': name_present, 'prev_state_present': state_present, 'user_id': user_id, 'num_prev_events': num_prev_events}

def add_to_dict(col, data, func):
	for i in col:
		dic = func(i)
		for k,v in dic.iteritems():
			data[k].append(v)
	return data		

def add_features(df):
	df = df.copy()

	previous = df.pop('previous_payouts')
	tickets = df.pop('ticket_types')
	data = defaultdict(list)

	for i,j in zip([previous, tickets],[process_prev_payments, ticket_types_summary]):
		data = add_to_dict(i, data, j)
	for k,v in data.iteritems():
		df[k] = v

	return df