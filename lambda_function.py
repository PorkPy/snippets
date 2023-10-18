
def lambda_handler(event, context):

    ## import libraries
    import pandas as pd
    import sagemaker, boto3
    from sagemaker import get_execution_role
    from sagemaker.inputs import TrainingInput
    import io


    """Script to train XGBoost model.
        Code fetches data from specified S3 location, cleans data, saves to S3 training data location,
        initiates XGBoost container, trains model and saves model back into S3 for 
        future deployment"""

    ## create a unique model name for xgboost model in local file system.
    def get_name():

        # Get the current date and time
        now = datetime.now()

        # Format the date and time as a string
        timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')

        # Create a unique model name using the timestamp
        model_name = f'xgboost-model-{timestamp}'

        print('Model Name is - ', model_name )
        return model_name
    
    def get_data(bucket, key):
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket= bucket, Key= key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))

        return df


    def train_model(bucket, key):

        ## get data from S3
        #------------------

        df = get_data(bucket, key)

        display(df)

        # create 'target' object to move target column to start of the data.
        target = df.target
        ## remove unnecessary features.
        df = df.drop(columns=['master_case_number',
                              'target'
                              ])
        # Make 'Target' first column.
        df = pd.concat([target, df], axis=1)

        ## upload cleaned data csv files to S3
        #--------------------------------------

        ## S3 bucket name for training data
        bucket='cmg-sagemaker-compliance-cases-data'
        ## subfolder in bucket
        file_path = "training_data"

        ## function to save data directly to S3 and not notebook instance.
        def upload(df, file_name):
            ## save df as StringIO object
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, header=False, index=False)
            boto3.client('s3').put_object(Body=csv_buffer.getvalue(), Bucket=bucket, Key=file_path+'/'+file_name)    

        upload(df, 'j_train.csv') 

        ## Create pointer objects that direct the estimator to the data in S3
        #--------------------------------------------------------------------

        ## S3 bucket name for saving training data
        bucket='cmg-sagemaker-compliance-cases-data'

        ## subfolder in S3 bucket
        file_path = "training_data"

        s3_input_train = TrainingInput(
            s3_data="s3://{}/{}/j_train".format(bucket, file_path), content_type="csv"
        )

        ## XGB Container
        #---------------
        ## this just needs defining for some reason.
        role = get_execution_role()

        # Sagemaker session
        sess = sagemaker.Session()

        ## get xgboost container
        container = sagemaker.image_uris.retrieve("xgboost", sess.boto_region_name, "1.5-1")

        ## define estimator parameters.
        xgb = sagemaker.estimator.Estimator(
            container,
            role,
            instance_count=1,
            instance_type="ml.m4.xlarge",
            output_path="s3://{}/{}/output".format(bucket, file_path),
            sagemaker_session=sess,
        )

        ## set basic estimator hyper-parameters.
        xgb.set_hyperparameters(
                                eval_metric="auc",
                                objective="binary:logistic",
                                num_round=10,
                                rate_drop=0.3,
                                tweedie_variance_power=1.4,
                                seed=42,
        )
        ## How should the algorithm evaluate the model?
        objective_metric_name = "validation:auc"

        ## fit model
        #-----------
        xgb.fit({"train": s3_input_train}, job_name=get_name()) # , "validation": s3_input_test


    if __name__=='__main__':

        bucket = 'cmg-sagemaker-compliance-cases-data'
        training_data_key = 'arrears-compliance-processed-data/30-04-2023-train-test/trainingCoCDataV6-30-04-2023.csv'

        train_model(bucket, training_data_key)