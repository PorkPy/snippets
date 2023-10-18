
def flatten(bucket,
            key,
            save_file_name=None,
            inference=False,
            testing=False,
            ):
    """
    master_case_numbers contain a number of SRs containing useful information. 
    For machine learning, each master case along with all its SRs need to be 
    flattened into single rows/records. Each categorical column of interest can 
    be pivoted to turn their values into count columns. Not all values from a 
    categorical column need be used. Values with few counts can be summed into 
    their own columns which helps reduce the total number of spares columns.  

    """

    ## imports
   
    import pandas as pd
    import numpy as np
    import glob
    import os
    import re
    import json
    import sagemaker, boto3
    from sagemaker import get_execution_role
    from sagemaker.inputs import TrainingInput
    from sagemaker.serializers import CSVSerializer
    import io
    import pickle


    from Get_Data import get_data

    ## flag to alter code for inference data 
    inference = str(inference)

    ## get local user
    #user = os.getlogin()
    ## os.getlogin() doesnt work in AWS because of how the processes are sporned. 
    import getpass 
    # invoking getuser() to extract
    # current parent process name
    user = getpass.getuser()
    print("\n\n ********* . User = ",user)
#   
#     ## get data location PATH from path.txt file
#     with open("../path.txt", "r") as fileObject:
#         PATH = fileObject.readlines()
#         PATH = '/Users/'+user+PATH[0]
#     ## remove pesky '\n' from list of PATHs.
#     PATH = PATH.strip('\n')

    ## get local pkl file path for saving sr_sub_area values.
    pklpath = '../model_files/'

    ## get data
    data = get_data(bucket, key)
    print('\n **** Getting Data ****\n ')
    #data = pd.read_csv(PATH + 'results_data/cleaned_data/concat_'+save_file_name+'.csv')
    data = data.drop_duplicates()
    # AWS converts number-like values to numbers, so need to convert back to str
    data['master_case_number'] = data['master_case_number'].astype(str)
    
    print("***********",data.shape)
    display(data)
    
    ## extra cleaning to removing '1-' infront of the master case number and convert to type int.
    data['master_case_number'] = data['master_case_number'].str.replace("1-", "").astype(int)
    ## convert dtypes 
    data = data.convert_dtypes()
    ## order data by master case number and sr open date.
    data['sr_open_date'] = pd.to_datetime(data['sr_open_date'])
    ## sort rows so all MCases are together and ordered by SR open date.
    data = data.sort_values(by=['master_case_number','sr_open_date']).reset_index(drop=True)
    data['arrears_review_mopf1'] = data['arrears_review_mopf1'].fillna("none")
    data['arrears_review_mopf1'] = data['arrears_review_mopf1'].apply(lambda x: x.strip().replace(" ", "").lower())
    
    #----------------------------------------------------------------------------------
    """
    ## Breakout sr_sub_area
    Get the value counts for the sr_sub_are. 
    Create new count features using sr-sub_area values with counts grater than 200.
    The remainder of values with less than 200 counts will be summed and used to build a 'None' column which
    is still useful for learning because it contains information about the number of SRs 
    raised against each master case number. It is hypothesised that master cases with a 
    greater number of SRs are more likely to default. 
    If using data for inference, sr_sub_area values not found in the training data columns are set to 'None'.
    columns in training data not present in the data sr_sub_are are corrected for letter in the 
    code after concatenating the flattened dfs. 

    """
    ## sr sub area

    if inference == 'True' or testing == 'True':
        ## The features between training testing and inference need to be the same.
        ## As there wont be as much inference data as training data, 
        ## there won't be as many sub_area values.
        ## All values within sr_sub_area therefore need to included.
        ## Any values that are missing during inference relative to training will be 
        ## added with a value of zero, to keep the data to the same shape.
        ## Retrieve sr_sub_area_values pickle file from training process. 
        filename = pklpath + f'inference_columns.pkl'
        with open(filename, 'rb') as file:  
            inference_columns = pickle.load(file)
        ## filter inference data based on trained model columns. 
        ## if value in sr_sub_area is not in inference_columns list, set value to None. 
        data.loc[~data["sr_sub_area"].isin(inference_columns), "sr_sub_area"] = "None"
        ## there may still be values in inference_columns that are not in sr_sub_area, 
        ## but these are filtered out at the end of the code.

    else:
        ## the value here has been set to 200, but can be adjusted. Sub_area values 
        ## not included due to having too few occurrences are added to the None column.
        ## This None column is one of the most important features as it shows the 
        ## number of interactions on a master case which has been shown to be indicative
        ## of missed payments. 
        sr_sub_area_values = data['sr_sub_area'].value_counts()
        print("\n SR value counts1 ",sr_sub_area_values)
        sr_sub_area_values = pd.DataFrame(sr_sub_area_values)
        sr_sub_area_values = sr_sub_area_values[sr_sub_area_values['sr_sub_area'] >10_000 ]        
        sr_sub_area_values = sr_sub_area_values.reset_index()
        sr_sub_area_values.columns = ['variable', 'count']
        print("\n SR value counts2 ",sr_sub_area_values)

        sr_sub_area_list = list(sr_sub_area_values['variable'])
        ## convert values in SR sub area to None if not in Sr sub area list above.
        data.loc[~data["sr_sub_area"].isin(sr_sub_area_list), "sr_sub_area"] = "None"
        ## There are the default/always present columns that need to be maintained. 
        default_columns = [
                        'None',
                        'master_case_number','ar_cc_outstanding_balance', 
                        #'arrears_rev_unpaid_ogm',
                        #'arrears_rev_ogm_per_period', 
                        'none_p', 'benefits_p','default_so_p', 'deo_p', 'der_p', 'directdebit_p','voluntary_so_p', 
            #'ar_missed_payment_count',
                        'income', 
            'age', 
                        'sr_open_date','open_sr_count' , 'Closed', 
            #'mopf_collection_day', 
                        'mcase_life_cycle', 
            'target',
                        ]
        inference_columns = sr_sub_area_list + default_columns
        filename = pklpath + f'inference_columns.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(inference_columns, file)
    #---------------------------------------------------------------------

    ## create new columns using most common SR sub area values
    ## group sr_sub_area
    sr_sub_area_group = data[['master_case_number', 'sr_sub_area']]
    sr_sub_area_group = sr_sub_area_group.groupby(['master_case_number', 'sr_sub_area']).agg({'sr_sub_area': ['count']})
    sr_sub_area_group.columns = sr_sub_area_group.columns.droplevel(0)
    sr_sub_area_group.columns.name = None              
    sr_sub_area_group = sr_sub_area_group.reset_index()  

    ## build pivot table
    sr_sub_area_group = pd.pivot_table(sr_sub_area_group, index=['master_case_number'], values=['count'], columns=['sr_sub_area'])
    sr_sub_area_group.columns = sr_sub_area_group.columns.droplevel(0)
    sr_sub_area_group.columns.name = None              
    sr_sub_area_group = sr_sub_area_group.reset_index()  
    sr_sub_area_group = sr_sub_area_group.fillna(0)
    sr_sub_area_group = sr_sub_area_group.astype(int)
    print("\\nn SR sub area group \n")
    display(sr_sub_area_group)

    
    #-----------------------------------------------------

    ## create new columns using most common arrears_review_mopf1 values
    ## group arrears_review_mopf1
    arrears_mop_group = data[['master_case_number', 'arrears_review_mopf1']]
    print("\n mop ")
    display(arrears_mop_group)

    arrears_mop_group = arrears_mop_group.fillna('none')
    arrears_mop_group = arrears_mop_group.groupby(['master_case_number', 'arrears_review_mopf1']).agg({'arrears_review_mopf1': ['count']}).unstack(fill_value=0).stack() #
    arrears_mop_group.columns = arrears_mop_group.columns.droplevel(0)
    arrears_mop_group.columns.name = None 
    arrears_mop_group = arrears_mop_group.reset_index() 
    
    ## build pivot table
    arrears_mop_group = pd.pivot_table(arrears_mop_group, index=['master_case_number'], values=['count'], columns=['arrears_review_mopf1'])
    arrears_mop_group.columns = arrears_mop_group.columns.droplevel(0)
    arrears_mop_group.columns.name = None              
    arrears_mop_group = arrears_mop_group.reset_index()  
    arrears_mop_group = arrears_mop_group.fillna(0)
    arrears_mop_group = arrears_mop_group.astype(int)
    arrears_mop_group.columns = arrears_mop_group.columns.str.lower()
    print('\n\n Arrears mop group1 \n ')
    display(arrears_mop_group)
    arrears_mop_group.rename(columns={'standingorder': 'default_so',
                                       'voluntarystandingorder': 'voluntary_so',
                                       'benefitagency': 'benefits',
                                       '<na>': 'none'}, inplace=True)

    ## New inference data or testing data might not have the same values 
    ## in arrear_mop so they need to be added manually to maintain data shape.
    mop_list = ['master_case_number', 
                'none',
                'benefits',
                'default_so', 
                'deo', 
                'der', 
                'directdebit',
                'voluntary_so']

    #arrears_mop_group.columns = [x + 'NEW' if any(k in x for k in keys) else x for x in df]

    
    empty_mop_values = [x for x in mop_list if x not in arrears_mop_group.columns]
    for i in empty_mop_values:
        if inference == 'True' or testing == 'True':
            ## create new columns with value 0.
            arrears_mop_group[i]=0

            
    print('\n\n Arrears mop group2 \n ')
    display(arrears_mop_group)
            
    ## reselect columns so they are always in the same order prior to changing names.
    arrears_mop_group = arrears_mop_group[['master_case_number', 
                                            'none',
                                            'benefits',
                                            'default_so', 
                                            'deo', 
                                            'der', 
                                            'directdebit',
                                            'voluntary_so']]

    ## alter some columns name so not to conflict with SR sub area values with similar names.
    arrears_mop_group.columns = ['master_case_number', 
                                'none_p',
                                'benefits_p',
                                'default_so_p', 
                                'deo_p', 
                                'der_p', 
                                'directdebit_p',
                                'voluntary_so_p']
    ## get number of unique master cases

    #-------------------------------------------------------

    # Outstanding Balance 
    ## convert zeros to nans so .last() method can find latest positive value
    data['ar_cc_outstanding_balance'].replace(0.0, np.nan, inplace=True)
    #data['arrears_rev_unpaid_ogm'].replace(0.0, np.nan, inplace=True)
    #data['arrears_rev_ogm_per_period'].replace(0.0, np.nan, inplace=True)

    ## get the last (non-zero/nan) outstanding balence value for each MCase.
    balance = data[['master_case_number', 
                    'ar_cc_outstanding_balance', 
                    #'arrears_rev_unpaid_ogm', 
                    #'arrears_rev_ogm_per_period',
                ]]
    balance = balance.groupby(['master_case_number']).last()
    ## fill in blank spaces with zeros.
    #balance['arrears_rev_unpaid_ogm'].replace(np.nan, 0, inplace=True)
    #balance['arrears_rev_ogm_per_period'].replace(np.nan, 0, inplace=True)
    balance = balance.reset_index()

    #-------------------------------------------------

    ## Missed Payment Count
    """
    The missed payment count is not fully understood. It seems that while an SR is open, 
    missed payments can add up until 
    the SR is closed. Most ARL SRs have an associated missed payment count of 1 which doesn't 
    provide any information. Missed payments from previous SRs that have had 
    time to accrue are suspected to be more indicative of a potential to miss future payments. 
    This code cell orders the data by master_case_number and missed payment counts. 
    It then deletes duplicate master_case_numbers keeping only the last one which 
    will have the largest missed payment count. 
    """
    ## get the highest missed payment count.
#     missed_payment = data[['master_case_number', 'ar_missed_payment_count']].sort_values(
#                         ['master_case_number','ar_missed_payment_count']).drop_duplicates(
#                         ['master_case_number'], keep='last')

    #missed_payment#[missed_payment['ar_missed_payment_count'] > 0]

    #--------------------------------------------------

    # Target Variable
    target = data[['master_case_number', 'target']]
    target = target.groupby(['master_case_number', 'target']).last()
    target.columns.name = None              
    target = target.reset_index()
    target = target[['master_case_number','target']]

    #------------------------------------------------

    ## Last SR Date
    last_sr_date = data[['master_case_number', 'sr_open_date']]
    last_sr_date = last_sr_date.groupby(['master_case_number']).last()
    last_sr_date.columns.name = None              
    last_sr_date = last_sr_date.reset_index()
    last_sr_date = last_sr_date[['master_case_number','sr_open_date']]
    last_sr_date = last_sr_date.drop_duplicates('master_case_number', keep='last')

    #---------------------------------------------------

    # Next Payment Due Date
#     next_payment = data[['master_case_number', 'mopf_collection_day']]
#     next_payment = next_payment.groupby(['master_case_number']).last()
#     next_payment.columns.name = None              
#     next_payment = next_payment.reset_index()
#     next_payment = next_payment[['master_case_number','mopf_collection_day']]
#     next_payment = next_payment.drop_duplicates('master_case_number', keep='last')
    #--------------------------------------------------------

    # Open SR Count

    # group sr_open
    sr_open = data[['master_case_number', 'sr_status']]
    sr_open = sr_open.groupby(['master_case_number', 'sr_status']).agg({'sr_status': ['count']})
    sr_open.columns = sr_open.columns.droplevel(0)
    sr_open.columns.name = None              
    sr_open = sr_open.reset_index() 

    ## build sr_open pivot table 
    sr_open = pd.pivot_table(sr_open, index=['master_case_number'], values=['count'], columns=['sr_status'])
    sr_open.columns = sr_open.columns.droplevel(0)
    sr_open.columns.name = None              
    sr_open = sr_open.reset_index()  
    sr_open = sr_open.fillna(0)
    sr_open = sr_open.astype(int)
    ## using list comp to find columns names with extra whitespace.
    sr_closed = sr_open[['master_case_number', [c for c in sr_open if c.startswith('Closed')][0]]]
    sr_closed = pd.DataFrame(sr_closed)
    sr_open = sr_open.drop([c for c in sr_open if c.startswith('Closed')][0],axis=1)
    sr_open['sum'] = sr_open.iloc[:,1:].sum(axis=1)
    sr_open = pd.DataFrame(sr_open[['master_case_number', 'sum']])
    sr_open.columns = ['master_case_number', 'open_sr_count']

    #--------------------------------------------------------
    ## last_assessed_annual_income
    income = data[['master_case_number', 'last_assessed_annual_income']]
    income = income.groupby(['master_case_number']).last()
    income.columns.name = None              
    income = income.reset_index() 
    income.columns = ['master_case_number', 'income']
    income = income.drop_duplicates()

    #----------------------------------------------------------

    ## Merge new columns to build dataset
    """
    Merge data.
    If using data for inference, check which columns from the training data are not present in the inference data.
    For each missed column, append column to data with a value of 0.
    This is to maintain the shape of the input data. 
    """
    merged_data = pd.merge(sr_sub_area_group, balance, on="master_case_number")
    merged_data = pd.merge(merged_data, arrears_mop_group, on="master_case_number")
#     merged_data = pd.merge(merged_data, missed_payment, on="master_case_number")
    merged_data = pd.merge(merged_data, last_sr_date, on="master_case_number")
#     merged_data = pd.merge(merged_data, next_payment, on="master_case_number")
    merged_data = pd.merge(merged_data, sr_open, on="master_case_number")
    merged_data = pd.merge(merged_data, sr_closed, on="master_case_number")
    merged_data = pd.merge(merged_data, income, on="master_case_number")
    merged_data = pd.merge(merged_data, target, on="master_case_number")
    merged_data['ar_cc_outstanding_balance'].replace(np.nan, 0, inplace=True)
    #merged_data = merged_data.astype(int)

    print("sjsjkndjknjsdn")
    with pd.option_context( 'display.max_columns', None,
                      #'display.max_rows', None,
                      ):
        display(merged_data)
    
    
    
    ## find missing columns
    empty_values = [x for x in inference_columns if x not in merged_data.columns]
    for i in empty_values:
        if inference == 'True' or testing == 'True':
            ## create new columns with value 0 to maintain data shape.
            merged_data[i]=0
        else:
            ## if training data, add missing columns data from 'data' df.
            merged_data[i] = data[i]

    if inference == 'True' or testing == 'True':
        ## add life_cycle columns for stand alone testing.
        merged_data['mcase_life_cycle'] = data.mcase_life_cycle
        merged_data['age'] = data.age
    ## reorder columns based on training data.
    merged_data = merged_data[inference_columns]
    
    ## drop duplicates
    merged_data = merged_data.drop_duplicates(subset='master_case_number')
    #--------------------------------------------------
    
    ## data shape
    #print("Flattened data shape",merged_data.shape)
    # Save Dataset
    print("\n Saving flattened data")
    #PATH = '/home/ec2-user/SageMaker/AWS_files/compliance-cases/results_data/'
    
#     merged_data.to_csv('flatten.csv', index=False)

#     import sagemaker
#     BUCKET_NAME='cmg-sagemaker-compliance-cases-data'
#     ## define new folder name to save cleaned data back in S3.
#     save_path = "phase3_training_data"

    
#     sagemaker.Session().upload_data(bucket=BUCKET_NAME, 
#                                       path='flatten.csv', 
#                                       key_prefix = save_path)
    
#     file_path = '/home/ec2-user/SageMaker/flatten.csv'
#     # Save the DataFrame as a CSV file
#     merged_data.to_csv(file_path)
#     # Set your bucket name and S3 file name
#     bucket_name = bucket
#     s3_file_name = save_path +'/'+'flatten.csv'
#     # Create an S3 resource
#     s3 = boto3.resource('s3')
#     # Upload the file to S3
#     s3.meta.client.upload_file(file_path, bucket_name, s3_file_name)
    
    
    
    ## upload cleaned data csv files to S3

    ## S3 bucket name
    bucket='cmg-sagemaker-compliance-cases-data'
    ## subfolder in bucket
    file_path = "phase3_flatten_data"
    def upload(df, file_name):
        ## save df as StringIO object so as not to save csv to notebook instance
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        boto3.client('s3').put_object(Body=csv_buffer.getvalue(), Bucket=bucket, Key= file_path +'/'+file_name)    

    upload(merged_data, "phase3_flatten.csv")

    print()
    with pd.option_context( 'display.max_columns', None,
                      #'display.max_rows', None,
                      ):
        display(merged_data)
        
        
    return merged_data
    #-----------------------------------------------------------------------------

if __name__=="__main__":
    print("main")

    save_file_name = 'test_data'

    flatten(save_file_name,
            )
