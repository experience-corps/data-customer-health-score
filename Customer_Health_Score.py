# Import modules
import numpy as np
import pandas as pd
import datetime as dt
import math
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# DATA CLEANING & TRANSFORMATION

def data_cleaning(data):
    df_copy = data.drop(labels=['SFDC Account ID','Current Location (primary)', 'Current Location (secondary)', 'Filename (primary)', \
    'Filename (secondary)', 'Expected Renewal ARR * (converted) Currency','Expected Renewal ARR * (converted)', 'Account Status.1', \
    'Account Contract End Date', 'Account Owner','Customer Success Coach', 'Fortune Rank', 'Billing State/Province', 'Billing Country', \
    'Hoovers Industry'], axis=1)
    
    # Remove irrelevant account status from dataframe
    remove_list = ['Active Customer - Reverse Ghost', 'Ghost Active - Content', 'Ghost Churned', 'Lead Partner', 'Prospect', 'Sandbox']
    for i in range(len(remove_list)):
        df_copy = df_copy[df_copy['Account Status'] != remove_list[i]]

    #null_row = df_copy[df_copy['Initial Term Start Date'].isnull()]

    df_copy = df_copy[df_copy['Customer Name'] != 'Technology Association of Oregon'].reset_index(drop=True)
    
    return df_copy


def create_duration(data, variable):
    # Change 'Initial term start date' into 'Duration' then bin the variable
    df_copy['start_date'] = pd.to_datetime(data[variable])

    # Use strftime (string format time) to get string mm/dd/yyyy instead of the datetime object
    end_date = dt.datetime.today().strftime("%m/%d/%Y")
    end_date = pd.to_datetime(end_date)
    end_date = [end_date for i in range (len(df_copy))]
    df_copy['end_date'] = end_date

    # Create new variable called Duration representing 
    # how long the customer has been with Showpad
    diff = (df_copy['end_date'] - df_copy['start_date']).dt.days
    diff = diff.apply(lambda x : int(x))

    return diff


def transform_okay_for_speech_api_rollout(data, variable):
    # 1: OK, 2: Hold for Special Notice, 3: Hold for Consent
    ok_list = ['OK-2', 'OK-3', 'OK-4', 'OK-DPA', 'ok', 'ok-5']
    tmp = np.array(data[variable])

    for i in range (tmp.size):
        if tmp[i] in ok_list:
            tmp[i] = '1'
        elif tmp[i] == 'Hold for Special Notice':
            tmp[i] = '2'
        elif tmp[i] == 'Hold for Consent':
            tmp[i] = '3'
        else:
            raise ValueError('Value not accepted.')
    
    speech_api_rollout = pd.DataFrame(tmp)

    return speech_api_rollout


def transform_initial_term_length(data, variable):
    # Create 4 equal width bins: (0,12], (12,24], (24,36],(36,48]
    tmp = data[variable].apply(lambda x: int(x))
    bins = [0, 12, 24, 36, 48]
    tmp_bin = pd.cut(tmp, bins).astype(str)

    for i in range (tmp.size):
        try:
            if tmp_bin[i] == '(0,12]':
                tmp_bin[i] = '1'
            elif tmp_bin[i] == '(12,24]':
                tmp_bin[i] = '2'
            elif tmp_bin[i] == '(24,36]':
                tmp_bin[i] = '3'
            else: #tmp[i] == '(36,48]'
                tmp_bin[i] = '4'
        except:
            raise ValueError('Value not accepted.')
    
    return tmp_bin


def transform_auto_renewal(data, variable):
    # 1: 'No', 2: 'Yes - 30-day notice', 3:' Yes - less than 30-day notice', 4: 'Yes - greater than 30-day notice'
    yes_30_list = ['Yes', 'Yes ', 'Yes (30-day opt out)', 'Yes (30-day opt-out)', 'yes (30-day opt-out)','Yes (written notice)', \
    'Yes (Custom- written confirmation; 30-day opt-out)', 'Yes (custom opt-out)', 'Yes (custom-30-day opt-out)', 'Yes (written agreement)']
    yes_30_more_list = ['Yes (3-month opt-out)', 'Yes (90-day opt-out)', 'Yes (90-day notification & 30-day opt-out)']
    yes_30_less_list = ['Yes (1-day opt-out)', 'Yes (5-day opt-out)']
    no_list = ['No', 'No (written agreement)', 'No (written notice required)']

    tmp = np.array(data[variable]) 

    for i in range (tmp.size):
        try:
            if tmp[i] in no_list: 
                tmp[i] = '1'
            elif tmp[i] in yes_30_list:
                tmp[i] = '2'
            elif tmp[i] in yes_30_less_list:
                tmp[i] = '3'
            elif tmp[i] in yes_30_more_list:
                tmp[i] = '4'
        except:
            raise ValueError('Value not accepted.')
    
    auto_renewal = pd.DataFrame(tmp)

    return auto_renewal


def transform_termination_rights(data, variable):
    # 1:'No', 2: 'Yes - less than 90-day notice', 3: 'Yes - at least 90-day notice'
    yes_90_less_list = ['Yes', 'Yes (30 days)', 'Yes (5 days)', 'Yes (60-day notice)', 'Yes (Custom)', 'Yes (first 3 months)',\
        'Yes (written notice)', 'Yes - 30-day notice', 'Yes- Two Months Notice', 'Yes-Custom']
    yes_90_or_more_list = ['Yes (90-day notice)', 'Yes - 90-day notice']

    tmp = np.array(data[variable])

    for i in range (tmp.size):
        if tmp[i] == 'No':
            tmp[i] = '1'
        elif tmp[i] in yes_90_less_list:
            tmp[i] = '2'
        elif tmp[i] in yes_90_or_more_list:
            tmp[i] = '3'
        else: 
            raise ValueError('Value not accepted.')   
    
    termination_rights = pd.DataFrame(tmp)

    return termination_rights


def transform_marketing(data, variable):
    # 1: 'Affirmative prohibition', 2: 'Silent', 3: 'Basic permission to reference as a customer', 4:'Enhanced marketing commitment'
    prohibition_list = ['Affirmative Prohibition', 'Affirmative prohibition', "Referral Partner requires Showpad's approval", 'No']
    silent_list = ['Silent', 'Silent  ', 'Silent or Soft Commitment', 'Silent or Soft Commitment ', 'Silent or Soft Support']
    reference_list = ['Agreed to provide marketing support']
    enhanced_list = ['Enhanced marketing commitment']

    tmp = np.array(data[variable])

    for i in range (tmp.size):
        try:
            if tmp[i] in prohibition_list:
                tmp[i] = '1'
            elif tmp[i] in silent_list:
                tmp[i] = '2'
            elif tmp[i] in reference_list:
                tmp[i] = '3'    
            elif tmp[i] in enhanced_list: 
                tmp[i] = '4'
        except:
            raise ValueError('Value not accepted.')  
        
    marketing = pd.DataFrame(tmp)

    return marketing


def transform_non_solicit_obligations(data, variable):
    # 1: Yes, 0: No
    yes_list = ['Yes', 'Yes - 1 year']

    tmp = np.array(df_copy['Non-Solicit Obligations'])

    for i in range (tmp.size):
        if tmp[i] in yes_list:
            tmp[i] = '1'
        elif tmp[i] == 'No':
            tmp[i] = '0'
        else:
            raise ValueError('Value not accepted.')
        
    obligations = pd.DataFrame(tmp)

    return obligations


def transform_legal_terms_n_conditions(data, variable):
    # 1: 'Custom', 2: 'Standard (MSA 2018v2)', 3: 'Standard (MSA 1Apr18)', 4: 'Standard (3Apr17)', 5: 'Standard (15Nov16)', 6: 'Standard (pre-15Nov16)'
    tmp = np.array(data[variable])

    custom_list = ['Custom (amended Standard (15nov2016 - 31mar2017))', 'Custom']

    for i in range (tmp.size):
        if tmp[i] in custom_list :
            tmp[i] = '1'
        elif tmp[i] == 'Standard (MSA 2018v2)':
            tmp[i] = '2'
        elif tmp[i] == 'Standard (MSA 01apr2018)': 
            tmp[i] = '3'
        elif tmp[i] == 'Standard (03apr2017-31mar2018)':
            tmp[i] = '4'
        elif tmp[i] == 'Standard (15nov2016 - 31mar2017)':
            tmp[i] = '5'
        elif tmp[i] == 'Standard (pre-15nov2016)':
            tmp[i] = '6'
        else:
            raise ValueError('Value not accepted.')
        
    legal_terms = pd.DataFrame(tmp)

    return legal_terms
   

def transform_sla(data, variable):
     # 1: 'No', 2: 'Custom', 3: 'Standard (Nov2016)', 4: 'Standard (2015)'
    custom_list = ['Yes', 'Yes ', 'Yes (Custom)', 'Yes (custom)', 'Yes (unknown)', 'Yes (updated time to time)']
    std_2016_list = ['Yes (Standard SLA Nov2016)']
    std_2015_list = ['Yes (Standard SLA 2015)', 'Yes (standard SLA 2015)']

    tmp = np.array(data[variable])

    for i in range (tmp.size):
        if tmp[i] == 'No':
            tmp[i] = '1'
        elif tmp[i] in custom_list:
            tmp[i] = '2'
        elif tmp[i] in std_2016_list:
            tmp[i] = '3'
        elif tmp[i] in std_2015_list: 
            tmp[i] = '4'
        else:
            raise ValueError('Value not accepted.')
    
    sla = pd.DataFrame(tmp)

    return sla


def transform_dpa(data, variable):
    # 1: Yes, 0: No
    tmp = np.array(data[variable])

    for i in range (tmp.size):
        if tmp[i] == 'Yes':
            tmp[i] = '1'
        elif tmp[i] == 'No':
            tmp[i] = '0'
        else:
            raise ValueError('Value not accepted.')
        
    dpa = pd.DataFrame(tmp)

    return dpa


def transform_privacy_change_notice(data, variable):
    # Convert blank to 'No' 
    # 1: 'No', 2: 'Notice and Opportunity to Object', 3: 'Passive Notice'
    tmp = np.array(data[variable].fillna(1))

    for i in range (len(tmp)):
        try:
            if tmp[i] == 'No':
                tmp[i] = '1'
            elif tmp[i] == 'Affirmative Opt-in or Amendment' or tmp[i] == 'Provide Opportunity to Review and Object':
                tmp[i] = '2'
            elif tmp[i] == 'Publication of Privacy Policy': 
                tmp[i] = '3'
        except:
            raise ValueError('Value not accepted.')
        
    prvcy_change_note = pd.DataFrame(tmp)

    return prvcy_change_note


def transform_data_security_breach_notice(data, variable):
    # 1: 'No', 2: 'Yes - within 24 hours of discovery', 3: 'Yes - between 24-48 hours after discovery', 4: 'Yes - greater than 48 hours after discovery'
    w_24_list = ['24 Hours', 'Yes (12 hours)', 'Yes (24 hours)', 'Yes (ASAP)', 'Yes (Immediately)', 'Yes (immediate)',\
        'Yes (immediately)', 'Yes (within 24 hours of discovery)', 'Yes- immediate notice of CI loss']
    w_48_list = ['Yes', 'Yes  ', 'Yes (2 business days)', 'Yes (48 hours)', 'Yes (Custom)', 'Yes (Within 48 hours of discovery)', \
        'Yes (custom)', 'Yes (undefined)', 'Yes (immediate, within 48 hours)']
    m_48_list = ['Yes (72 hours)', 'Yes (Without Undue Delay)', 'Yes (within 72 hours of discovery)']

    tmp = np.array(data[variable])

    for i in range (tmp.size):
        try:
            if tmp[i] == 'No':
                tmp[i] = '1'
            elif tmp[i] in w_24_list:
                tmp[i] = '2'
            elif tmp[i] in w_48_list:
                tmp[i] = '3'
            elif tmp[i] in m_48_list:
                tmp[i] = '4'
        except: 
            raise ValueError('Value not accepted.')
          
    sec_breach_note = pd.DataFrame(tmp)

    return sec_breach_note


def transform_change_of_control_provision(data, variable):
    # 1: Yes, 0: No
    tmp = np.array(data[variable])

    for i in range (tmp.size):
        if tmp[i] == 'Yes':
            tmp[i] = '1'
        elif tmp[i] == 'No':
            tmp[i] = '0'
        else: 
            raise ValueError('Value not accepted.')
        
    cntrl_prvsn = pd.DataFrame(tmp)

    return cntrl_prvsn


def transform_governing_law(data, variable):
    # 1: New York,  2: Delaware, 3: California, 4: Illinois, 5: Texas, 6: Belgium, 7: Ireland, 8: France, 9: Germany, 
    # 10: rest of the states in the USA, 11: rest of the countries in the EU
    # Since there are only 2 records of Germany, I will change it into group 11 unless we have more data 
    tmp = np.array(data[variable])

    CA_list = ['California', 'California ']
    USA_rest_list = ['Florida', 'Georgia',  'Indiana', 'Nevada', 'New Jersey',  'Pennsylvania', 'Wisconsin', 'None']
    Germany_list = ['Germany', 'German']
    EU_rest_list = ['England', 'England & Wales', 'England and Wales', 'England/Wales', 'English', 'UK', 'Dutch', 'Switzerland' ]
 
    for i in range (tmp.size):
        if tmp[i] == 'New York':
            tmp[i] = '1'
        elif tmp[i] == 'Delaware':
            tmp[i] = '2'
        elif tmp[i] in CA_list:
            tmp[i] = '3'
        elif tmp[i] == 'Illinois':
            tmp[i] = '4'
        elif tmp[i] == 'Texas':
            tmp[i] = '5'
        elif tmp[i] == 'Belgium':
            tmp[i] = '6'
        elif tmp[i] == 'Ireland':
            tmp[i] = '7'
        elif tmp[i] == 'France':
            tmp[i] = '8'
        elif tmp[i] in Germany_list:
            tmp[i] = '9'
        elif tmp[i] in USA_rest_list:
            tmp[i] = '10'  
        elif tmp[i] in EU_rest_list or tmp[i] in Germany_list:
            tmp[i] = '11'
        else:
            raise ValueError('Value not accepted.')
                
    govn_law = pd.DataFrame(tmp)

    return govn_law


def transform_terms(data, variable):
    # 'Not an actual customer' treat as blank
    # Change yes or any content to 1 and no or blank to 0
    tmp = np.array(data[variable].fillna(0))

    for i in range (len(tmp)):
        if tmp[i] == 'Not an actual customer' or tmp[i] == 'No' or tmp[i] == 0:
            tmp[i] = '0'
        else:
            tmp[i] = '1'
        
    terms = pd.DataFrame(tmp)

    return terms


def combine_terms(df1, df2):
    # Combine 'Notable Non-Standard Terms' and 'Outrageous Terms'
    non_std_outrageous = np.array(pd.concat([df1, df2], axis=1)).astype(int)

    s = [non_std_outrageous[i][0] for i in range (len(non_std_outrageous))]
    n = 0

    for i in range (len(s)):
        for j in range (2):
            n += non_std_outrageous[i][j]
        s[i] = n
        n = 0    

    combined_terms = s.copy()
    for i in range (len(combined_terms)):
        if combined_terms[i] == 0:
            combined_terms[i] = '0'
        else: combined_terms[i] = '1'
    
    return combined_terms


def transform_account_status(data, variable):
    # Change status from string to numeric representation 
    # 1 = Churned/Locked, 0 = Active
    tmp = np.array(data[variable])

    active_list = ['Active Customer - Coach', 'Active Customer - Content', 'Active Customer - Platform']
    churn_list = ['Active Customer - Churned', 'Active Customer - locked', 'Churned Customer']

    for i in range (len(tmp)):
        if tmp[i] in churn_list:
            tmp[i] = '1'
        elif tmp[i] in active_list: 
            tmp[i] = '0'
        else:
            raise ValueError('Value not accepted.')
           
    status = pd.DataFrame(tmp).astype(int)

    return status


def transform_arr(data, var1, var2):
    # Convert GBP and USD to EUR
    # 6/28 curency rate
    # 1 USD = 0.88 EUR
    # 1 GBP = 1.12 EUR
    tmp = np.array(data[[var1, var2]])

    for i in range (len(tmp)):
        if tmp[i][0] == 'GBP':
            tmp[i][1] = tmp[i][1] * 1.12
        elif tmp[i][0] == 'USD':
            tmp[i][1] = tmp[i][1] * 0.88
           
    renewal_ARR = pd.DataFrame(tmp)
    renewal_ARR = renewal_ARR[1].apply(lambda x: int(x))

    return renewal_ARR


def transform_package(data, variable):
    # 1: Enterprise, 2: Essential, 3: Plus, 4: Professional, 5: Ultimate
    tmp = np.array(data[variable])

    for i in range (len(tmp)):
        if tmp[i] == 'Enterprise':
            tmp[i] = '1'
        elif tmp[i] == 'Essential':
            tmp[i] = '2'
        elif tmp[i] == 'Plus':
            tmp[i] = '3'
        elif tmp[i] == 'Professional':
            tmp[i] = '4'
        else: tmp[i] = '5'

    package = pd.DataFrame(tmp)

    return package


def transform_segment(data, variable):
    # 1: ENT, 0: MM
    tmp = np.array(data[variable])

    for i in range (tmp.size):
        if tmp[i] == 'ENT':
            tmp[i] = '1'
        else:
            tmp[i] = '0'
        
    segment = pd.DataFrame(tmp)

    return segment


def transform_emloyees(data, variable):
    # Fill in missing value with most frequent value 
    tmp = np.array([data[variable]]).T

    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    emp = imp_freq.fit_transform(tmp)

    employees_trans = pd.DataFrame(emp)

    return employees_trans


# Load file into pandas
# Data starts in row #5
df = pd.read_csv("/Users/liyan.wang/Desktop/Customer_Health_Score/Metadata SFDC (customers active at Feb2018) - Main.csv", header=0)
df = df[4:].reset_index(drop=True)

df_copy = data_cleaning(df)
df_copy['duration'] = create_duration(df_copy, 'Initial Term Start Date')
df_copy['speech_api_rollout'] = transform_okay_for_speech_api_rollout(df_copy, 'Okay for Speech API Rollout?')          
df_copy['term_length'] = transform_initial_term_length(df_copy, 'Initial Term Length')
df_copy['auto_renewal'] = transform_auto_renewal(df_copy, 'Auto-renewal')
df_copy['termination_rights'] = transform_termination_rights(df_copy, 'Termination Rights')
df_copy['marketing'] = transform_marketing(df_copy, 'Marketing')
df_copy['obligations'] = transform_non_solicit_obligations(df_copy, 'Non-Solicit Obligations')
df_copy['legal_terms'] = transform_legal_terms_n_conditions(df_copy, 'Legal Terms and Conditions')
df_copy['sla'] = transform_sla(df_copy, 'SLA')
df_copy['dpa'] = transform_dpa(df_copy, 'DPA')
df_copy['prvcy_change_note'] = transform_privacy_change_notice(df_copy, 'Privacy Policy Change Notice Requirement')
df_copy['sec_breach_note'] = transform_data_security_breach_notice(df_copy, 'Data or Security Breach Notice Requirement')
df_copy['cntrl_prvsn'] = transform_change_of_control_provision(df_copy, 'Change of Control provision')
df_copy['govn_law'] = transform_governing_law(df_copy, 'Governing Law')
non_std_terms = transform_terms(df_copy, 'Notable Non-Standard Terms')
outrageous = transform_terms(df_copy, 'Outrageous Terms')
df_copy['non_std_outrageous'] = combine_terms(non_std_terms, outrageous)
df_copy['status'] = transform_account_status(df_copy, 'Account Status')
df_copy['renewal_ARR'] = transform_arr(df_copy, 'Expected Renewal ARR * Currency', 'Expected Renewal ARR *')
df_copy['package'] = transform_package(df_copy, 'Package')
df_copy['assigned_user'] = df_copy['Assigned users'].apply(lambda x: int(x))
df_copy['segment'] = transform_segment(df_copy, 'Segment')
df_copy['employees_trans'] = transform_emloyees(df_copy, 'Employees')


# Create dataframe for binned variables
# Select transformed variables(binned) into a new dataframe
bin_df = df_copy[['speech_api_rollout', 'duration', 'term_length', 'auto_renewal', 'termination_rights', 'marketing','obligations', 'legal_terms', \
    'sla', 'dpa', 'prvcy_change_note', 'sec_breach_note', 'cntrl_prvsn','govn_law', 'non_std_outrageous', 'status', 'renewal_ARR', 'package',\
    'N° of Licenses', 'Used Licenses', 'assigned_user', 'segment', 'employees_trans']]


# Seperate dependent and indepednet variables
x = bin_df.drop('status',axis = 1)
y = bin_df['status']
#print( 'x shape is: ', x.shape, 'y shape is: ', y.shape)


# Create train and test set split at 67/33
# MAKE SURE TRAIN AND TEST SHAPE MATCH
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=11)
#print('x_train shape is: ', x_train.shape, 'y_train shape is: ', y_train.shape)
#print('x_test shape is: ', x_test.shape, 'y_test shape is: ', y_test.shape)


# Normalization on numeric variables
min_max_scaler = preprocessing.MinMaxScaler()
x_train_norm = min_max_scaler.fit_transform(x_train[['duration','renewal_ARR','N° of Licenses', 'Used Licenses', 'assigned_user', 'employees_trans']])
x_test_norm = min_max_scaler.fit_transform(x_test[['duration','renewal_ARR','N° of Licenses', 'Used Licenses','assigned_user', 'employees_trans']])

new_x_train = pd.DataFrame(x_train_norm, columns = ['duration','renewal_ARR','N° of Licenses', 'Used Licenses', 'assigned_user', 'employees_trans'])
new_x_test = pd.DataFrame(x_test_norm, columns = ['duration','renewal_ARR','N° of Licenses', 'Used Licenses', 'assigned_user', 'employees_trans'])


# Create dummy variables
x_train_dummy = pd.get_dummies(x_train[['auto_renewal', 'termination_rights', 'legal_terms', 'sla', 'sec_breach_note', 'non_std_outrageous', \
    'package', 'segment']]).reset_index(drop=True)
x_test_dummy = pd.get_dummies(x_test[['auto_renewal', 'termination_rights', 'legal_terms', 'sla','sec_breach_note', 'non_std_outrageous', \
    'package', 'segment']]).reset_index(drop=True)


# Combine two dataframes into one for modeling
x_train_frame = pd.concat([x_train_dummy, new_x_train], axis=1)
x_test_frame = pd.concat([x_test_dummy, new_x_test], axis=1)


X_train = np.array(x_train_frame)
X_test = np.array(x_test_frame)

Y_train = np.array(y_train)
Y_test = np.array(y_test)

#print('After transformation and feature selection, x_train shape is:', X_train.shape)
#print('After transformation and feature selection, x_test shape is:', X_test.shape)
#print('After transformation and feature selection, y_train shape is:', Y_train.shape)
#print('After transformation and feature selection, y_test shape is:', Y_test.shape)


# MODELING

logReg = LogisticRegression(penalty='l1', solver='saga')
model = logReg.fit(X_train, Y_train)
predictions = logReg.predict(X_test)
score = logReg.score(X_test, Y_test)
cm = metrics.confusion_matrix(Y_test, predictions, labels=[1,0])


## Probalility of getting 1 (churn)
prob_test_1 = logReg.predict_proba(X_test)[:,1]
prob_train_1 = logReg.predict_proba(X_train)[:,1]


# APPLICATION

def compute_quantile(df):
    tmp = pd.DataFrame(df)
    tmp = np.array(tmp[0].sort_values())
    
    lv1_len = math.ceil(1/5 * len(tmp))
    lv2_len = math.ceil(2/5 * len(tmp))
    lv3_len = math.ceil(3/5 * len(tmp))
    lv4_len = math.ceil(4/5 * len(tmp))
    
    level1 = tmp[lv1_len]
    level2 = tmp[lv2_len]
    level3 = tmp[lv3_len]
    level4 = tmp[lv4_len]
    
    return level1, level2, level3, level4


def compute_score(df):
    p_bin = np.array(df)
    p_bin = [p_bin[i][0] for i in range (len(p_bin))]

    for i in range (len(p_bin)):
        if p_bin[i] <= p_level1:
            p_bin[i] = '1'
        elif p_level1 < p_bin[i] <= p_level2:
            p_bin[i] = '2'
        elif p_level2 < p_bin[i] <= p_level3:
            p_bin[i] = '3'
        elif p_level3 < p_bin[i] <= p_level4:
            p_bin[i] = '4'
        else:
            p_bin[i] = '5'
        
    return p_bin 


x_name = df_copy['Customer Name']
x_train_name, x_test_name = train_test_split(x_name, test_size=0.33, random_state=11)

p_train_1 = pd.DataFrame(prob_train_1)
p_test_1 = pd.DataFrame(prob_test_1)
p_1 = np.array(pd.concat([p_test_1, p_train_1]).reset_index(drop=True))
p = np.array([p_1[i][0] for i in range (len(p_1))])

p_level1, p_level2, p_level3, p_level4 = compute_quantile(p)

# Apply score back to train set
train_score = compute_score(p_train_1)

x_train_new = pd.DataFrame(x_train_name).reset_index(drop=True)
x_train_new['prob'] = p_train_1
x_train_new['score'] = train_score

# Apply score back to test set
test_score = compute_score(p_test_1)

x_test_new = pd.DataFrame(x_test_name).reset_index(drop=True)
x_test_new['prob'] = p_test_1
x_test_new['score'] = test_score


# Combine train and test then export
combined = pd.concat([x_train_new, x_test_new], ignore_index=True)
combined = combined.sort_values(by=['Customer Name']).reset_index(drop=True)
combined.to_csv('Customer Health Score.csv', index=False)