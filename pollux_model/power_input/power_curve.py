from pollux_model.model_abstract import Model
import os
import pandas as pd
import csv


class DataProcessor(Model):

    """
    This is a class where you can add the power as an input file and read it in python
             as next steps we can add new attributes

    """

    def __init__(self):
        """ Model initialization
        """
        super().__init__()

    def update_parameters(self, parameters):
        """ To update model parameters

        Parameters
        ----------
        parameters : dict
            parameters dict as defined by the model
        """
        for key, value in parameters.items():
            self.parameters[key] = value

    def initialize_state(self, x):
        """ generate an initial state based on user parameters """
        pass

    def calculate_output(self, u):
        """calculate output based on input u"""

        self.run(u)

    # def __init__(self, file_path, data_type):
    #     self.file_path = file_path
    #     self.data_type = data_type.lower()
    #     self.data = None

    def read_user_file(self):
        # Get the file extension

        if self.parameters['file_path'] is not None:
            _, file_extension = os.path.splitext(self.parameters['file_path'])

            # Read the file based on its extension
            if file_extension == '.csv':
                self.data = pd.read_csv(self.parameters['file_path'])
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(self.parameters['file_path'])
            elif file_extension == '.txt':
                with open(self.parameters['file_path'], 'r') as file:
                    self.data = file.readlines()
            else:
                raise ValueError("Unsupported file type")
        else:
            self.data_type = 'custom_solar'  # an auxiliary vaiable for the
            # simulation,you can use also self.output['data_type']
            self.output['data_type'] = 'custom_solar'
            self.output['power'] = 100

    def process_data(self):
        results = []

        # Example nested loops (adjust as needed)
        for i in range(5):  # Outer loop
            for j in range(3):  # Middle loop
                for k in range(2):  # Inner loop
                    # Perform some operations based on the data type
                    if self.data_type == 'solar':
                        value = i * j * k * 1.5  # Example operation for solar
                    elif self.data_type == 'wind':
                        value = i * j * k * 2.0  # Example operation for wind
                    elif self.data_type == 'both':
                        value = i * j * k * 2.5  # Example operation for both
                    elif self.data_type == 'custom_solar':
                        value = 123
                    else:
                        raise ValueError("Invalid data type. Please enter 'solar', "
                                         "'wind', or 'both'.")
                    # Save the iteration values
                    results.append([i, j, k, value])

        return results

    def save_results(self, results, output_file):
        # Save results to CSV
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['i', 'j', 'k', 'value'])
            # Write the data rows
            writer.writerows(results)

    def run(self, u=0):
        # Read the user file
        try:
            self.read_user_file()
            print(f"Data from {self.parameters['file_path']} has been read successfully.")
        except ValueError as e:
            print(e)
            return

        # Process the data based on the type
        results = self.process_data()

        # Define the output CSV file name
        output_csv_file = 'results.csv'

        # Save the iteration results to the CSV file
        self.save_results(results, output_csv_file)

        # Print the full path where the CSV file is saved
        full_path = os.path.abspath(output_csv_file)
        print(f"Results saved to {full_path}")
