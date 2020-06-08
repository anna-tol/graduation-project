'''
Prototype "NICU Amsterdam UMC Nursing Dashboard"

As part of thesis "Predicting and preventing nursing shortages within hospital departments using k-nearest neighbors and data visualization"

BY ANNA TOL

First examiner: Marieke Welle Donker-Kuijer
Second examiner: Aletta Smits

Master Data-Driven Design
University of Applied Science Utrecht

June, 2020
'''

# import plotly
import plotly.graph_objects as go

# import dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

# import time related
from datetime import datetime
import calendar
from calendar import monthrange

# import data related
import os
import pandas as pd
import numpy as np

# import model related
from sklearn.neighbors import KNeighborsRegressor

# find data function
def find_data(file_name):
    return os.path.dirname(__file__) + '/data/' + file_name

# data
employees_future = pd.read_csv(find_data('employees future.csv'))
employees_future['in'] = pd.to_datetime(employees_future['in'], dayfirst=True)

# standard parameters
clicks = 0
beds = 25
months_training = 20
months_forecast = 22
fte_hours_per_week = 36
overall_absenteeism = 1 - 190 / 234
unexpected_absenteeism = 0.07
expected_absenteeism = overall_absenteeism - unexpected_absenteeism
start_month_past = pd.Timestamp('2017-01-01')
start_month_future = pd.Timestamp('2020-04-01')

# colors
nurse_color = '#58b9d4'
student_color = '#5490c9'
support_color = '#526db3'
care_color = '#f4982b'
demand_color = '#dd6b1e'
capacity_color = '#e83a4e'

# ------------------------------------------------ general functions ------------------------------------------------

# load main dataset
def load_data():
    dataset  = pd.read_excel(find_data('dataset.xlsx'))
    dataset = dataset.set_index('month')
    dataset.index = pd.to_datetime(dataset.index)
    return dataset

# calculate number of weeks in month
def weeks_per_month(start, end):
    
    if start > end:
        months = [i for i in pd.date_range(start = end, end = start, freq='MS')]
    else:
        months = [i for i in pd.date_range(start = start, end = end, freq='MS')]
    
    weeks_per_month = [monthrange(months[i].year, months[i].month)[1]/7 for i in range(0, len(months))]
    
    return weeks_per_month

# convert a time period to aggregated sum of fte per month
def duration_to_fte_pm(df, function):

    months = pd.date_range(start = start_month_future, end = start_month_future + pd.DateOffset(months = months_forecast), freq='MS')
    temp_df = []

    for index, row in df.iterrows():
        
        temp_df_row = []
        
        for month in months:

            # in case of a limited period
            if 'out' in df.columns:
                if month >= pd.to_datetime(row['in']) and month < pd.to_datetime(row['out']):
                    temp_df_row.append(row['fte'])
                else:
                    temp_df_row.append(0)

            # in case of a limitless period
            else:
                if month >= pd.to_datetime(row['in']):
                    temp_df_row.append(row['fte'])
                else:
                    temp_df_row.append(0)
                
        temp_df.append(temp_df_row)

    # combine data and aggregate
    df = pd.DataFrame(temp_df, columns = months)
    df = df.agg({df.columns[i]:'sum' for i in range(0, months_forecast)})
    
    return pd.DataFrame(df, columns=['fte ' + function])

# simulate outflow
def decrease(df, number):

    df[df.columns[0]] = [df[df.columns[0]][i] + (i * number) for i in range(0, len(df[df.columns[0]]))]
    
    return df

# manager employees (new employee and reset)
def employees(submit_button, function, fte, date):
    
    global employees_future

    # in case a new employees is added
    if track_submit(submit_button) == True and function != None and fte != None and date != None:
        date = datetime.strptime(date, '%Y-%m-%d')
        new_employee = pd.DataFrame({'fte':[fte], 'in':[date], 'function':[function], 'new':[True]})
        employees_future = employees_future.append(new_employee, ignore_index = True, sort=False)
    # when reset button is clicked
    elif submit_button == None and function == None and fte == None and date == None:
        employees_future = pd.read_csv(find_data('employees future.csv'))
        employees_future['in'] = pd.to_datetime(employees_future['in'], dayfirst=True)
    
    return employees_future

# add date features and lags for model
def add_features(data, var, lag, n_lag):
    
    # date features
    data['date'] = data.index
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['quarter'] = data['date'].dt.quarter
    data['pos_in_quarter'] = data['month'] % 3
    data['pos_in_quarter'] = data['pos_in_quarter'].replace(0, 3)
    data['tertile'] = data['month'].replace([1, 2, 3, 4], 1).replace([5, 6, 7, 8], 2).replace([9, 10, 11, 12], 3)
    data['pos_in_tertile'] = data['month'] % 4
    data['pos_in_tertile'] = data['pos_in_tertile'].replace(0, 4)
    
    # lags
    for n in range(0, n_lag):
        data[var + '_t' + str(-lag + n)] = data[var].shift(-lag + n)
    
    return data

# check if the submit button is clicked
def track_submit(n_clicks):
    global clicks
    if n_clicks == None:
        return False
    elif n_clicks > clicks:
        clicks = n_clicks
        return True
    else:
        return False

# ------------------------------------------------ update data ------------------------------------------------

# update the dataset
def update_data(data, employees_future, avg_duration_care_day, number_beds, 
                ctpc_nurse, ctpc_student, ctpc_support, 
                outflow_nurses, outflow_support, unexpected_absenteeism):
    
    # update average duration of care day
    data['care days hours'] = data['care days'] * avg_duration_care_day
    data['demand care days hours'] = data['demand care days'] * avg_duration_care_day
    data['maximum capacity'] = data['days per month'] * avg_duration_care_day * number_beds

    # update future supply

    # get all employees and their fte
    student_future = employees_future.loc[employees_future['function'] == 'student']
    student_future = student_future.copy()
    student_future['out'] = student_future['in'] + pd.DateOffset(months = months_training)
    student_future = student_future[['in', 'out', 'fte']]
    new_nurses = student_future[['out', 'fte']]
    new_nurses.columns = ['in', 'fte']
    nurse_future = employees_future.loc[employees_future['function'] == 'nurse']
    nurse_future = nurse_future.append(new_nurses, ignore_index = True, sort = False)
    nurse_future = nurse_future[['in', 'fte']]
    support_future = employees_future.loc[employees_future['function'] == 'support']
    support_future = support_future[['in', 'fte']]

    # convert their fte to fte per month
    nurse_future = duration_to_fte_pm(nurse_future, 'nurses')
    student_future = duration_to_fte_pm(student_future, 'students')
    support_future = duration_to_fte_pm(support_future, 'support')

    # deduct outflow
    nurse_future = decrease(nurse_future, -outflow_nurses)
    support_future = decrease(support_future, -outflow_support)

    # merge three kinds of employees
    supply_future = pd.concat([nurse_future, student_future, support_future], axis=1).sort_index(ascending=True)
    supply_future.index.name = 'month'
    
    # convert fte to gross hours
    supply_future['nurses gross hours'] = supply_future['fte nurses'] * fte_hours_per_week * weeks_per_month(supply_future.index[0], supply_future.index[-1])
    supply_future['students gross hours'] = supply_future['fte students'] * fte_hours_per_week * weeks_per_month(supply_future.index[0], supply_future.index[-1])
    supply_future['support gross hours'] = supply_future['fte support'] * fte_hours_per_week * weeks_per_month(supply_future.index[0], supply_future.index[-1])

    # deduct abstenteeism
    supply_future['unexpected absenteeism'] = unexpected_absenteeism
    supply_future['presence'] = 1 - supply_future['unexpected absenteeism'] - data['expected absenteeism'][0]
    supply_future['nurses net hours'] = supply_future['nurses gross hours'] * supply_future['presence']
    supply_future['students net hours'] = supply_future['students gross hours'] * supply_future['presence']
    supply_future['support net hours'] = supply_future['support gross hours'] * supply_future['presence']

    # update future supply in data
    data.update(supply_future)

    # update all supply data
    
    # update contribution to patient care employee
    data['nurses net hours'] = data['nurses net hours'] * ctpc_nurse
    data['students net hours'] = data['students net hours'] * ctpc_student
    data['support net hours'] = data['support net hours'] * ctpc_support

    return data

# update the forecasts
def update_forecast(dataset):
    
    dataset = dataset.sort_index(ascending = False)
    data = dataset.copy()
    
    # settings
    n_lag = 1
    lag = 22
    n_neighbors = 4

    # variables to use
    x_var = 'care days hours'
    y_var = ['nurses net hours', 'students net hours', 'support net hours']
    not_y_var = set([i for i in dataset.columns]) - set(y_var)

    # prepare data
    data = add_features(data, x_var, lag, n_lag)
    data_future = data[:months_forecast]
    data_past = data[:-months_forecast]
    data_past = data_past.dropna()

    # define x and y variable(s)
    y = data_past[x_var]
    x = data_past.drop(columns = not_y_var).drop(columns='date')
    x = (x-x.min())/(x.max()-x.min())

    # fit model
    knn = KNeighborsRegressor(n_neighbors = n_neighbors)
    knn.fit(x, y)
    
    # define x again for future data
    data_future = data_future.drop(columns = not_y_var).drop(columns='date').dropna()
    x = data_future
    x = (x-x.min())/(x.max()-x.min())
    
    # forecast
    forecast = knn.predict(x)
    
    # insert in dataframe
    part_to_update = dataset[:months_forecast][[x_var]]
    part_to_update[x_var] = forecast
    dataset.update(part_to_update)

    return dataset

# ------------------------------------------------ graph functions ------------------------------------------------

# draw the main graph
def main_graph(data):

    data[data < 0] = 0

    # from net hours to cumulative hours
    data['nurses net hours cum'] = data['nurses net hours']
    data['students net hours cum'] = data['students net hours'] + data['nurses net hours cum']
    data['support net hours cum'] = data['support net hours'] + data['students net hours cum']

    fig = go.Figure()

    # hover texts
    nurse_hover_text = ['{:.1f}'.format(ps) + '% of supply <br> {:.2f}'.format(npr) + ' nurse-patient ratio' for ps, npr in zip(list(data['nurses net hours'] / data['support net hours cum'] * 100), list(data['care days hours'] / data['nurses net hours']))]
    student_hover_text = ['{:.0f}'.format(ah) + ' hours (actual) <br> {:.1f}'.format(ps) + '% of supply' for ah, ps in zip(list(data['students net hours']), list(data['students net hours'] / data['support net hours cum'] * 100))]
    support_hover_text = ['{:.0f}'.format(ah) + ' hours (actual) <br> {:.1f}'.format(ps) + '% of supply' for ah, ps in zip(list(data['support net hours']), list(data['support net hours'] / data['support net hours cum'] * 100))]

    # supply cumulative 
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data['nurses net hours cum'], 
        mode = 'lines', 
        name = 'Nurses (cumulative)', 
        fill = 'tozeroy', 
        marker_color = nurse_color, 
        text = nurse_hover_text, 
        hovertemplate = '<i>%{x}</i><br>%{y:.0f} hours<br>%{text}',
        hoverlabel = dict(bordercolor = 'white')
        ))
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data['students net hours cum'], 
        mode='lines', 
        name='Students (cumulative)', 
        fill='tonexty', 
        marker_color = student_color,
        text = student_hover_text,
        hovertemplate = '<i>%{x}</i><br>%{y:.0f} hours (cumulative)<br>%{text}',
        hoverlabel = dict(bordercolor = 'white')
        ))
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data['support net hours cum'], 
        mode='lines', 
        name='Support (cumulative)', 
        fill='tonexty', 
        marker_color = support_color,
        text = support_hover_text,
        hovertemplate = '<i>%{x}</i><br>%{y:.0f} hours (cumulative)<br>%{text}',
        hoverlabel = dict(bordercolor = 'white')
        ))
    # supply actual
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data['nurses net hours'], 
        mode='lines', 
        name='Nurses (actual)', 
        visible = 'legendonly', 
        marker_color=nurse_color,
        text = nurse_hover_text, 
        hovertemplate = '<i>%{x}</i><br>%{y:.0f} hours<br>%{text}',
        hoverlabel = dict(bordercolor = 'white')
        ))
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data['students net hours'], 
        mode='lines', 
        name='Students (actual)', 
        visible = 'legendonly', 
        marker_color=student_color,
        text = ['{:.1f}'.format(ps) + '% of supply' for ps in list(data['students net hours'] / data['support net hours cum'] * 100)],
        hovertemplate = '<i>%{x}</i><br>%{y:.0f} hours<br>%{text}',
        hoverlabel = dict(bordercolor = 'white')
        ))
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data['support net hours'],
        mode='lines', 
        name='Support (actual)', 
        visible = 'legendonly', 
        marker_color=support_color,
        text = ['{:.1f}'.format(ps) + '% of supply' for ps in list(data['support net hours'] / data['support net hours cum'] * 100)],
        hovertemplate = '<i>%{x}</i><br>%{y:.0f} hours<br>%{text}',
        hoverlabel = dict(bordercolor = 'white')
        ))
    # production, demand and maximum
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data['care days hours'], 
        mode='lines', 
        name='Patient care', 
        marker_color=care_color,
        text = data['care days hours'] / data['demand care days hours'] * 100,
        hovertemplate = '%{x} <br>%{y:.0f} hours<br>%{text:.1f}% of demand',
        hoverlabel = dict(bordercolor = 'white')
        )),
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data['demand care days hours'], 
        mode='lines', 
        name='Demand', 
        marker_color=demand_color,
        hovertemplate = '%{x} <br>%{y:.0f} hours',
        hoverlabel = dict(bordercolor = 'white')
        )),
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data['maximum capacity'], 
        mode='lines', name='Maximum capacity', 
        visible = 'legendonly', 
        marker_color=capacity_color,
        hovertemplate = '%{x} <br>%{y:.0f} hours',
        hoverlabel = dict(bordercolor = 'white')
        )),
    # layout
    fig.update_layout(
        font_size = 11,
        yaxis_title= 'hours',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        shapes=[dict(
            type='line', 
            x0=pd.Timestamp('2020-03-01'), 
            y0=0, x1=pd.Timestamp('2020-03-01'), 
            y1=20000, line=dict(color="Grey",width=2,))],
        font = dict(family='Source Sans Pro', size = 13),
        margin = dict(t = 30, b = 40),
        height = 300,
        )
    # style
    fig.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='#efefef'),
    fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='#efefef', tickformat = 'digit')
    
    return fig

# draw the hired graph
def hired_graph(data):
    fig = go.Figure(    
        data=[
            go.Bar(
                # students
                name = 'Student', 
                x = data.loc[data['function'] == 'Student']['year'], 
                y = data.loc[data['function'] == 'Student']['fte'], 
                marker_color=student_color,
                text = ['{:.2f}'.format(tpf) + ' total fte function <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['function'] == 'Student']['total per function']), list(data.loc[data['function'] == 'Student']['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # support
                name = 'Support', 
                x = data.loc[data['function'] == 'Support']['year'], 
                y = data.loc[data['function'] == 'Support']['fte'], 
                marker_color = support_color, 
                text = ['{:.2f}'.format(tpf) + ' total fte function <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['function'] == 'Support']['total per function']), list(data.loc[data['function'] == 'Support']['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
    ])
    # layout
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text= 'Employed Staff', font = dict(size = 16)),
        title_x=0.5,
        font = dict(family='Source Sans Pro', size = 12),
        yaxis_title= 'fte',
        barmode='stack',
        margin = dict(t = 32, b = 0, l = 0, r = 0),
        height = 180,
        )
    # style
    fig.update_traces(marker_line_width=0.1)
    fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='#d7d7d7')

    return fig

# draw the resignation reasons graph
def resignation_graph(data):
    reasons = sorted(list(dict.fromkeys([i for i in data['reason']])), reverse=True)
    fig = go.Figure(    
        data=[
            go.Bar(
                # unkown, grey
                name = reasons[0], 
                x = data.loc[data['reason'] == reasons[0]]['year'], 
                y = data.loc[data['reason'] == reasons[0]]['fte'], 
                marker_color = '#b4b0af',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[0]]['total per reason']), list(data.loc[data['reason'] == reasons[0]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # other specialism, green
                name = reasons[2], 
                x = data.loc[data['reason'] == reasons[2]]['year'], 
                y=data.loc[data['reason'] == reasons[2]]['fte'], 
                marker_color='#6bbfa1',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[2]]['total per reason']), list(data.loc[data['reason'] == reasons[2]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # other job, green
                name = reasons[3], 
                x = data.loc[data['reason'] == reasons[3]]['year'], 
                y=data.loc[data['reason'] == reasons[3]]['fte'], 
                marker_color='#61b87c',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[3]]['total per reason']), list(data.loc[data['reason'] == reasons[3]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # other career, green
                name = reasons[4],
                x = data.loc[data['reason'] == reasons[4]]['year'], 
                y = data.loc[data['reason'] == reasons[4]]['fte'], 
                marker_color = '#65b331',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[4]]['total per reason']), list(data.loc[data['reason'] == reasons[4]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # long-term illness, green
                name = reasons[5], 
                x = data.loc[data['reason'] == reasons[5]]['year'], 
                y=data.loc[data['reason'] == reasons[5]]['fte'], 
                marker_color='#aaca44',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[5]]['total per reason']), list(data.loc[data['reason'] == reasons[5]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # commute, green
                name = reasons[10], 
                x = data.loc[data['reason'] == reasons[10]]['year'], 
                y=data.loc[data['reason'] == reasons[10]]['fte'], 
                marker_color='#d5db4d',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[10]]['total per reason']), list(data.loc[data['reason'] == reasons[10]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # irregularity, orange
                name = reasons[6], 
                x = data.loc[data['reason'] == reasons[6]]['year'], 
                y=data.loc[data['reason'] == reasons[6]]['fte'], 
                marker_color='#fcca3e',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[6]]['total per reason']), list(data.loc[data['reason'] == reasons[6]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # (pre)retirement, orange
                name = reasons[11], 
                x = data.loc[data['reason'] == reasons[11]]['year'], 
                y=data.loc[data['reason'] == reasons[11]]['fte'], 
                marker_color='#fbb800',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[11]]['total per reason']), list(data.loc[data['reason'] == reasons[11]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # physically demanding, orange
                name = reasons[1], 
                x = data.loc[data['reason'] == reasons[1]]['year'], 
                y=data.loc[data['reason'] == reasons[1]]['fte'], 
                marker_color='#f4982b',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[1]]['total per reason']), list(data.loc[data['reason'] == reasons[1]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # insufficient practice, red
                name = reasons[7], 
                x = data.loc[data['reason'] == reasons[7]]['year'], 
                y=data.loc[data['reason'] == reasons[7]]['fte'], 
                marker_color='#dd6b1e',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[7]]['total per reason']), list(data.loc[data['reason'] == reasons[7]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # discontent with department, red
                name = reasons[9], 
                x = data.loc[data['reason'] == reasons[9]]['year'], 
                y=data.loc[data['reason'] == reasons[9]]['fte'], 
                marker_color='#e74b15',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[9]]['total per reason']), list(data.loc[data['reason'] == reasons[9]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
            go.Bar(
                # discontent with job, red
                name = reasons[8], 
                x = data.loc[data['reason'] == reasons[8]]['year'], 
                y=data.loc[data['reason'] == reasons[8]]['fte'], 
                marker_color='#e83a4e',
                text = ['{:.2f}'.format(tpf) + ' total fte reason <br>'+ '{:.2f}'.format(t) + ' total fte' for tpf, t in zip(list(data.loc[data['reason'] == reasons[8]]['total per reason']), list(data.loc[data['reason'] == reasons[8]]['total']))],
                hovertemplate = '<i>%{x}</i> <br> %{y:.2f} fte <br>%{text} %{x}',
                hoverlabel = dict(bordercolor = 'white')),
    ])
    # layout
    fig.update_layout(
        title=dict(text= 'Resignation Reasons', font = dict(size = 16)),
        title_x=0.5,
        yaxis_title= 'fte',
        barmode='stack',
        height = 180,
        font = dict(family='Source Sans Pro', size = 12),
        margin = dict(t = 32, b = 0, l = 0, r = 0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    # style
    fig.update_traces(marker_line_width=0.1)
    fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='#d7d7d7')

    return fig

# ------------------------------------------------ html elements ------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div(children = [
    # header
    html.Div([
        html.H1('NICU Amsterdam UMC Nursing Dashboard'),
    ],
    className = 'title-div',
    ),
    # upper inputs
    # average patient care day
    html.Div([
        html.H3('Average patient care day'),
        html.Label('Hours per day'),
        dcc.Input(
            id = 'avg_duration_day',
            type= 'number',
            value = 21.8,
            min = 0.0,
            step = 0.1,
            max = 24.0,
        ),
    ], 
    className = 'top-divs',
    style={'margin-left': '5vw'},
    ),
    # maximum capacity
    html.Div([
        html.H3('Maximum capacity'),
        html.Label('Beds'),
        dcc.Input(
            id = 'number_beds',
            type= 'number',
            value = 25,
            min = 0,
            step = 1,
            max = 100,
        ),
    ], 
    className = 'top-divs',
    ),
    # contribution to patient care
    html.Div([
        html.H3('Contribution to patient care'),
        html.Div([
            html.Label('Nurses %'),
            dcc.Input(
                id = 'ctpc_nurse',
                type= 'number',
                value = 100,
                min = 0,
                step = 1,
                max = 100,
            ),
        ],
        className = 'top-divs-sub',
        ),
        html.Div([
            html.Label('Students %'),
            dcc.Input(
                id = 'ctpc_student',
                type= 'number',
                value = 100,
                min = 0,
                step = 1,
                max = 100,
            ),
        ],
        className = 'top-divs-sub',
        ),
        html.Div([
            html.Label('Support %'),
            dcc.Input(
                id = 'ctpc_support',
                type= 'number',
                value = 100,
                min = 0,
                step = 1,
                max = 100,
            ),
        ],
        className = 'top-divs-sub',
        style={'margin-right': 0},
        ),
    ], 
    className = 'top-divs',
    ),
    # reset button
    html.Div([
        html.Button('Reset', id='reset_button'),
    ],
    style={'display': 'inline-block', 'float': 'right', 'margin-right': '5vw', 'margin-top': '3vh'}
    ),
    # main graph
    html.Div([
        dcc.Graph(id = 'main_graph'),
    ],
    className='main_div',
    ),
    html.Div([
        dcc.Tabs(parent_className='custom-tabs', children=[
             # tab 1: inflow
            dcc.Tab(label='Hire', className='custom-tab', selected_className='custom-tab--selected', children=[
                # inflow past
                html.Div([
                    html.H2('Past'),
                    dcc.Graph(id = 'hired_graph')
                ],
                className = 'tab-div-sides',
                ),
                # inflow future
                html.Div([
                    # give input
                    html.Div([
                        html.H2('Future'),
                        html.Div([
                            html.Label('Function'),
                            dcc.Dropdown(
                                id = 'inflow_function',
                                options=[
                                    {'label': 'Nurse', 'value': 'nurse'},
                                    {'label': 'Student', 'value': 'student'},
                                    {'label': 'Support', 'value': 'support'},
                                ],
                            )
                        ],
                        className = 'dropdown',
                        ),
                        html.Div([
                            html.Label('FTE'),
                            dcc.Input(
                                id = 'inflow_fte',
                                type= 'number',
                                value = 1.00,
                                min = 0.00,
                                step = 0.01,
                                max = 100.00,
                            ),
                        ],
                        className = 'bottom-divs-sub',
                        ),
                        html.Div([
                            html.Label('Start date'),
                            dcc.Dropdown(
                                id = 'inflow_start',
                                className = 'dropdown',
                                options=[
                                    {'label': 'April 2020', 'value': '2020-04-01'},
                                    {'label': 'May 2020', 'value': '2020-05-01'},
                                    {'label': 'June 2020', 'value': '2020-06-01'},
                                    {'label': 'July 2020', 'value': '2020-07-01'},
                                    {'label': 'August 2020', 'value': '2020-08-01'},
                                    {'label': 'September 2020', 'value': '2020-09-01'},
                                    {'label': 'October 2020', 'value': '2020-10-01'},
                                    {'label': 'November 2020', 'value': '2020-11-01'},
                                    {'label': 'December 2020', 'value': '2020-12-01'},
                                    {'label': 'January 2021', 'value': '2021-01-01'},
                                    {'label': 'February 2021', 'value': '2021-02-01'},
                                    {'label': 'March 2021', 'value': '2021-03-01'},
                                ],
                            ),
                        ],
                        className = 'dropdown',
                        ),

                        html.Div([
                            html.Label('.'),
                            html.Button('Submit', id='submit_button', type='submit'),
                        ],
                        className = 'bottom-divs-sub',
                        ),
                    ],
                    ),
                    # show input
                    html.Div([
                        dash_table.DataTable(
                            id = 'table',
                            columns = [],
                            style_cell = {'backgroundColor': 'white', 'color': 'rgb(50, 50, 50)', 'font_family': 'sans-serif', 'width': '13vw'},
                            style_as_list_view = True,
                        )
                    ], 
                    style={'margin-top': 90}
                    )
                ],
                className = 'tab-div-sides',
                )
            ]),


            # tab 2: outflow
            dcc.Tab(label='Retain', className='custom-tab', selected_className='custom-tab--selected', children=[
                # outlfow past
                html.Div([
                    html.H2('Past'),
                    dcc.Graph(id = 'resignation_graph')
                ],
                className = 'tab-div-sides',
                ),
                # outflow future
                html.Div([
                    html.H2('Future'),
                    html.Div([
                        html.H3('Absenteeism'),
                        html.Label('Percentage'),
                        dcc.Input(
                            id = 'absenteeism_rate',
                            type= 'number',
                            value = 7,
                            min = 0,
                            step = 1,
                            max = 100,
                        ),
                    ], 
                    className = 'top-divs',
                    ),
                    html.Div([
                        html.H3('Outflow per year'),
                        html.Div([
                            html.Label('Fte nurses'),
                            dcc.Input(
                                id = 'outflow_nurses',
                                type= 'number',
                                value = 8.4,
                                min = 0.0,
                                step = 0.1,
                                max = 50,
                            ),
                        ],
                        className = 'top-divs-sub',
                        ),
                        html.Div([
                            html.Label('Fte support'),
                            dcc.Input(
                                id = 'outflow_support',
                                type= 'number',
                                value = 1.2,
                                min = 0.0,
                                step = 0.1,
                                max = 50,
                            ),
                        ],
                        className = 'top-divs-sub',
                        style={'margin-right': 0},
                        ),
                    ], 
                    className = 'top-divs',
                    ),
                ],
                className = 'tab-div-sides',
                )
            ]),
        ]),
    ],
    style={'margin-left': '5vw', 'margin-right': '5vw'}
    )
])

# ------------------------------------------------ app callback functions ------------------------------------------------

# update main graph
@app.callback(
    Output('main_graph', 'figure'), 
    [ # main
    Input('avg_duration_day', 'value'),
    Input('number_beds', 'value'),
    Input('ctpc_nurse', 'value'),
    Input('ctpc_student', 'value'),
    Input('ctpc_support', 'value'),
    # inflow
    Input('inflow_function', 'value'),
    Input('inflow_fte', 'value'),
    Input('inflow_start', 'value'),
    Input('submit_button', 'n_clicks'),
    # outflow
    Input('absenteeism_rate', 'value'),
    Input('outflow_nurses', 'value'),
    Input('outflow_support', 'value')]
    )
def update_figure(avg_duration_day, number_beds, ctpc_nurse, ctpc_student, ctpc_support, inflow_function, inflow_fte, inflow_start, submit_button, absenteeism_rate, outflow_nurses, outflow_support):
    
    if avg_duration_day == None or number_beds == None or ctpc_nurse == None or ctpc_student == None or ctpc_support == None or absenteeism_rate == None or outflow_nurses == None or outflow_support  == None:
        raise PreventUpdate
    
    # percentage to unit interval
    ctpc_nurse /= 100
    ctpc_student /= 100
    ctpc_support /= 100
    absenteeism_rate /= 100

    # outflow per year to per month
    outflow_nurses /= 12
    outflow_support /= 12
    
    # load data
    dataset = load_data()

    # add employees
    employees_future = employees(submit_button, inflow_function, inflow_fte, inflow_start)
    
    # update data and forecast
    data = update_data(dataset, employees_future, avg_duration_day, number_beds, ctpc_nurse, ctpc_student, ctpc_support, outflow_nurses, outflow_support, absenteeism_rate)
    data = update_forecast(data)
    
    return main_graph(data)

# update table
@app.callback(
    [Output('table', 'data'), 
    Output('table', 'columns')],
    [Input('inflow_function', 'value'),
    Input('inflow_fte', 'value'),
    Input('inflow_start', 'value'), 
    Input('submit_button', 'n_clicks'),
    Input('reset_button', 'n_clicks')]
)
def update_table(inflow_function, inflow_fte, inflow_start, submit_button, reset_button):

    # add new employee
    data = employees(submit_button, inflow_function, inflow_fte, inflow_start)

    # if reset button is clicked
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'reset_button' in changed_id:
         data = employees(None, None, None, None)

    # prepare for table
    data = data.loc[employees_future['new'] == True]
    data = data.copy()
    data['in'] = data['in'].dt.strftime('%B %Y')
    data = data.drop(columns='new')
    columns = [{'name': col, 'id': col} for col in data.columns]
    data = data.to_dict(orient='records')

    return data, columns

# update hired graph
@app.callback(
    Output('hired_graph', 'figure'),
    [Input('submit_button', 'n_clicks')]
)
def update_figure(submit_button):
    data = pd.read_excel(find_data('hired.xlsx'))
    return hired_graph(data)

# update resignation graph
@app.callback(
    Output('resignation_graph', 'figure'),
    [Input('absenteeism_rate', 'value')]
)
def update_figure(absenteeism_rate):
    data = pd.read_excel(find_data('resignation reasons.xlsx'))
    return resignation_graph(data)

# reset button
@app.callback(
    [Output('avg_duration_day', 'value'),
    Output('number_beds', 'value'),
    Output('ctpc_nurse', 'value'),
    Output('ctpc_student', 'value'),
    Output('ctpc_support', 'value'),
    Output('absenteeism_rate', 'value'),
    Output('outflow_nurses', 'value'),
    Output('outflow_support', 'value'), 
    Output('inflow_function', 'value'),
    Output('inflow_fte', 'value'),
    Output('inflow_start', 'value')],
    [Input('reset_button', 'n_clicks')]
)
def update_figure(reset_button):
    
    # all the defaults
    avg_duration_day = 21.8
    number_beds = 25
    ctpc_nurse = 100
    ctpc_student = 100
    ctpc_support = 100
    absenteeism = 7
    outflow_nurses = 8.4
    outflow_support = 1.2
    inflow_function = None
    inflow_fte = 1
    inflow_start = None

    return avg_duration_day, number_beds, ctpc_nurse, ctpc_student, ctpc_support, absenteeism, outflow_nurses, outflow_support, inflow_function, inflow_fte, inflow_start

# run app
if __name__ == '__main__':
    app.run_server(debug = True)