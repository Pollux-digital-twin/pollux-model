from pollux_model.model_abstract import Model
from pollux_model.solver.step_function import StepFunction


class Splitter(Model):
    def __init__(self, time_function):
        super().__init__()
        self.time_function = time_function
        self.current_time = 0

    def initialize_state(self, x):
        """ generate an initial state based on user parameters """

    pass

    def calculate_output(self):
        if len(self.input) == 1:
            self.output['output_0'] = self.input['input'] * \
                self.time_function.evaluate(self.current_time)
            self.output['output_1'] = self.input['input'] * \
                (1 - self.time_function.evaluate(self.current_time))
        else:
            raise ValueError("splitter requires exactly 1 input.")

    def update_time(self, time_step):
        self.current_time += time_step

    def set_time(self, time):
        self.current_time = time

    def update_time_function(self, control):
        step_size = self.time_function.get_step_size()
        step_function = StepFunction(control, step_size)
        self.time_function = step_function
