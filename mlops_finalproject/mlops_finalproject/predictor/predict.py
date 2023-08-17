import argparse
import joblib

class ModelPredictor:
    """
    A class to load a trained machine learning model and make predictions on new data.

    Parameters:
        model_path (str): Path to the trained model file (joblib format).

    Methods:
        predict(new_data):
            Makes predictions on the provided new_data using the loaded model.

    Usage:
        $ python model_predictor.py trained_models/logistic_regression_output.pkl path_to_new_data
    """

    def __init__(self, model_path):
        """
        Initializes the ModelPredictor instance.

        Parameters:
            model_path (str): Path to the trained model file (joblib format).
        """
        self.model = joblib.load(model_path)

    def predict(self, new_data):
        """
        Makes predictions on the provided new_data using the loaded model.

        Parameters:
            new_data: The data on which to make predictions.

        Returns:
            Predicted outputs from the model.
        """
        return self.model.predict(new_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Predictor')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    parser.add_argument('new_data', type=str, help='Path to the file containing new data for prediction')
    args = parser.parse_args()

    predictor = ModelPredictor(args.model_path)

    new_data = args.new_data

    predictions = predictor.predict(new_data)
    print(predictions)