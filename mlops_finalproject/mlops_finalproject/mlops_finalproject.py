"""Main module."""
from load.load_data import DataRetriever
from train.train_data import HotelReservationsDataPipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score

DATASETS_DIR = './data/'
URL = '/Users/norma.perez/Documents/GitHub/MLOps_FinalProject/mlops_finalproject/mlops_finalproject/data/Hotel_Reservations.csv'
DROP_COLS = ['Booking_ID']
RETRIEVED_DATA = 'retrieved_data.csv'


SEED_SPLIT = 404
TRAIN_DATA_FILE = DATASETS_DIR + 'train.csv'
TEST_DATA_FILE  = DATASETS_DIR + 'test.csv'


TARGET = 'booking_status'
FEATURES = ['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights','type_of_meal_plan','required_car_parking_space','room_type_reserved','lead_time','arrival_year','arrival_month','arrival_date','market_segment_type','repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests','booking_status']
NUMERICAL_VARS = ['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights','required_car_parking_space','lead_time','arrival_year','arrival_month','arrival_date','repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests']
CATEGORICAL_VARS = ['type_of_meal_plan','room_type_reserved','market_segment_type']


NUMERICAL_VARS_WITH_NA = []
CATEGORICAL_VARS_WITH_NA = []
NUMERICAL_NA_NOT_ALLOWED = [var for var in NUMERICAL_VARS if var not in NUMERICAL_VARS_WITH_NA]
CATEGORICAL_NA_NOT_ALLOWED = [var for var in CATEGORICAL_VARS if var not in CATEGORICAL_VARS_WITH_NA]


SEED_MODEL = 404
SELECTED_FEATURES = ['lead_time', 'avg_price_per_room', 'no_of_special_requests', 'arrival_date', 'arrival_month', 'no_of_week_nights']

TRAINED_MODEL_DIR = './mlops_finalproject/models/'
MODEL_NAME = 'extra_trees_classifier_model'
PIPELINE_NAME = 'extra_trees_classifier_pipeline'
MODEL_SAVE_FILE = f'{MODEL_NAME}_output.pkl'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'


if __name__ == "__main__":
    
    print(os.getcwd())
    os.chdir('/Users/norma.perez/Documents/GitHub/MLOps_FinalProject/mlops_finalproject/mlops_finalproject')
    # Retrieve data
    data_retriever = DataRetriever(URL, DATASETS_DIR)
    result = data_retriever.retrieve_data()
    print(result)
    
    # Instantiate the TitanicDataPipeline class
    hotelreservations_data_pipeline = HotelReservationsDataPipeline(seed_model=SEED_MODEL,
                                                numerical_vars=NUMERICAL_VARS, 
                                                categorical_vars_with_na=CATEGORICAL_VARS_WITH_NA,
                                                numerical_vars_with_na=NUMERICAL_VARS_WITH_NA,
                                                categorical_vars=CATEGORICAL_VARS,
                                                selected_features=SELECTED_FEATURES)
    
    # Read data
    df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
                                                        df.drop(TARGET, axis=1),
                                                        df[TARGET],
                                                        test_size=0.2,
                                                        random_state=404
                                                   )
    
    
    extra_trees_classifier_model = hotelreservations_data_pipeline.fit_extra_trees_classifier(X_train, y_train)
    
    X_test = hotelreservations_data_pipeline.PIPELINE.fit_transform(X_test)
    y_pred = extra_trees_classifier_model.predict(X_test)
    
    class_pred = extra_trees_classifier_model.predict(X_test)
    proba_pred = extra_trees_classifier_model.predict_proba(X_test)[:,1]
    print(f'test roc-auc : {roc_auc_score(y_test, proba_pred)}')
    print(f'test accuracy: {accuracy_score(y_test, class_pred)}')
    
    # # Save the model using joblib
    save_path = TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE
    joblib.dump(extra_trees_classifier_model, save_path)
    print(f"Model saved in {save_path}")
    