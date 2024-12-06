from pollux_model.model_abstract import Model


class Adder(Model):
    def __init__(self):
        super().__init__()
        self.current_time = 0

    def initialize_state(self, x):
        pass

    def calculate_output(self):
        if len(self.input) == 2:
            self.output['output'] = self.input['input_0'] + self.input['input_1']
        else:
            raise ValueError("adder requires exactly 2 inputs.")
