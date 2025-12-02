from .base_handler import BASE
from trajan import constants
import numpy as np
import sys

class DENSITY(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose)

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
            print(f"ERROR: Unit set \"{args.units}\" is not supported. Please choose out of available unit sets: \n\n{'\n'.join(constants.AVAILABLE_UNITS)}\n")
            sys.exit(1)

        self.parse_file()
        self.check_required_columns("type")

        num_masses = len(self.masses)
        types = self.get_types()
        ntypes = len(types)
        if num_masses < ntypes:
            print(f"ERROR: Not enough masses given ({num_masses}) for all types ({' '.join(types.astype(str))}) present.")
            sys.exit(1)
        elif num_masses > ntypes:
            print(f"WARNING: Too many masses given ({num_masses}) for all types ({' '.join(types.astype(str))}) present.")

        self.masses = np.concatenate(([0], self.masses))

        self.wrap_positions()

        self.densities = list()

    def analyze(self):
        atomic_data = self.get_atomic_data()
        columns = self.get_columns()
        for i in range (self.get_Nframes()):
            all_types = atomic_data[i][:, columns["type"]].astype(int)
            all_masses = self.masses[all_types]
            density = np.sum(all_masses) / (np.prod(self.lengths[i])) #amu/A^3
            density *= constants.MASS_CONVERSIONS[self.units] / np.power(constants.DISTANCE_CONVERSIONS[self.units], 3)
            self.densities.append(density)

            self.verbose_print(f"{i + 1} analysis of TS {self.get_timesteps()[i]}", verbosity = 2)

        self.densities = np.array(self.densities)
        print("Analysis complete")

    def write(self):

        super().write(data = np.column_stack((np.arange(1, self.get_timesteps().shape[0] + 1), self.get_timesteps(), self.densities)),
                      header = "frame, time step, density",
                      outfile = self.outfile,
                      )

    def statistics(self):
        #Name : (value, verbosity)
        stats_dict = {"Mean density" : (np.mean(self.densities), 1),
                      "Standard deviation" : (np.std(self.densities), 1),
        }

        super().statistics(stats_dict = stats_dict)
