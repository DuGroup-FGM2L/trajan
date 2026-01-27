from .base_handler import BASE
from trajan import constants
import numpy as np
import sys

class DENSITY(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose, args.steps, args.buffer)

        self.masses = list()
        for el in args.elements:
            try:
                self.masses.append(float(el))
            except ValueError:
                self.masses.append(constants.ATOMIC_MASSES[el])


        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "density_" + self.outfile

        if args.units in constants.AVAILABLE_UNITS:
            self.units = args.units
        else:
            unitstr = '\n'.join(constants.AVAILABLE_UNITS)
            self.verbose_print(f"ERROR: Unit set \"{args.units}\" is not supported. Please choose out of available unit sets: \n\n{unitstr}\n")
            sys.exit(1)

        self.parse_file()
        self.check_required_columns("type")

        num_masses = len(self.masses)
        types = self.get_types()
        ntypes = len(types)
        if num_masses < ntypes:
            self.verbose_print(f"ERROR: Not enough masses given ({num_masses}) for all types ({' '.join(types.astype(str))}) present.")
            sys.exit(1)
        elif num_masses > ntypes:
            self.verbose_print(f"WARNING: Too many masses given ({num_masses}) for all types ({' '.join(types.astype(str))}) present.")

        self.masses = np.concatenate(([0], self.masses))

        self.densities = list()

    def analyze(self):
        columns = self.get_columns()
        for frame_idx in self.trajectory_reader():
            atomic_data = self.get_atomic_data()
            all_types = atomic_data[:, columns["type"]].astype(int)
            all_masses = self.masses[all_types]
            density = np.sum(all_masses) / (np.prod(self.get_lengths())) #amu/A^3
            density *= constants.MASS_CONVERSIONS[self.units] / np.power(constants.DISTANCE_CONVERSIONS[self.units], 3)
            self.densities.append(density)

            self.verbose_print(f"{frame_idx} analysis of TS {self.get_timestep()}", verbosity = 2)

        self.densities = np.array(self.densities)
        self.verbose_print("Analysis complete")

    def write(self):

        super().write(data = np.column_stack((np.arange(1, self.get_user_frame()), self.get_timesteps(), self.densities)),
                      header = "frame, time step, density",
                      outfile = self.outfile,
                      )

    def statistics(self):
        #Name : (value, verbosity)
        stats_dict = {"Mean density" : (np.mean(self.densities), 1),
                      "Standard deviation" : (np.std(self.densities), 1),
        }

        super().statistics(stats_dict = stats_dict)
