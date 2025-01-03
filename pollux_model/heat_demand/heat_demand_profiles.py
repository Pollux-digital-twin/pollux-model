from pollux_model.model_abstract import Model


class HeatDemand(Model):
    def __init__(self):
        super().__init__()

    def initialize_state(self, x):
        """ generate an initial state based on user parameters
            """
        pass

    def calculate_output(self):
        self.output['heat_demand'] = self.time_function.evaluate(self.current_time)
