from pollux_model.model_abstract import Model


class HydrogenTankModel(Model):
    """Compressed gas isothermal model for hydrogen"""

    def __init__(self):
        super().__init__()

        self.parameters['timestep'] = 1  # seconds
        self.parameters['maximum_capacity'] = 6  # maximum mass [kg]
        self.parameters['maximum_pressure'] = 7e7  # maximum pressure [Pa]
        self.parameters['maximum_volume'] = \
            HydrogenTankModel._hydrogen_mass_to_volume(
                self.parameters['maximum_capacity'],
                self.parameters['maximum_pressure'])  # maximum volume [m3]

        x = dict()
        x['current_mass'] = 0
        self.initialize_state(x)

    def initialize_state(self, x):
        """ generate an initial state based on user parameters """

        self.state['current_mass'] = x['current_mass']
        self._calculate_fill_level()

    def calculate_output(self, u):
        """calculate output based on input u"""

        mass_flow = u['mass_flow']
        timestep = self.parameters['timestep']

        delta_mass = mass_flow * timestep
        self.state['current_mass'] = self.state['current_mass'] + delta_mass

        self._calculate_fill_level()

        # Assign output to state
        self.output = self.state

    def _calculate_fill_level(self):
        """function to calculate fill level"""
        self.state['fill_level'] = self.state['current_mass'] / self.parameters['maximum_capacity']

    @staticmethod
    def _hydrogen_compressibility(pressure):
        """function to calculate hydrogen compressibility at 15 C.
        ref: https://arxiv.org/pdf/1702.06015 page 5"""

        z = 1 + 0.39 * pressure * 1e-5 / 600

        return z

    @staticmethod
    def _hydrogen_density(pressure):
        """function to calculate hydrogen density at 15 C"""
        T = 15 + 273.15  # Temperature in Kelvin [K]
        Rgas = 8314.51  # [J/kmol K]
        Mw = 2  # Molecular Weight

        Z = HydrogenTankModel._hydrogen_compressibility(pressure)

        rho_gas = pressure * (Mw / (Z * Rgas * T))

        return rho_gas

    @staticmethod
    def _hydrogen_mass_to_volume(mass, pressure):
        volume = mass / HydrogenTankModel._hydrogen_density(pressure)

        return volume