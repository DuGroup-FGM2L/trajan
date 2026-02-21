from .base_handler import BASE
from trajan import constants

import numpy as np
import sys
from collections import deque

c_cm_s = constants.c * 100

class VDOS(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose, args.steps, args.buffer, args.filter_type)
        if args.units in constants.AVAILABLE_UNITS:
            self.units = args.units
        else:
            unitstr = '\n'.join(constants.AVAILABLE_UNITS)
            self.verbose_print(f"ERROR: Unit set \"{args.units}\" is not supported. Please choose out of available unit sets: \n\n{unitstr}\n")
            sys.exit(1)


        frame_step = self.get_frame_step()
        self.timestep = args.timestep * frame_step
        self.max_timelag = int(args.max_timelag / self.timestep)
        if args.lag_step is None:
            self.lag_step = constants.DEFAULT_LAG_STEP
        else:
            self.lag_step = int(args.lag_step * frame_step / self.timestep)


        if self.lag_step < 1:
            print(f"ERROR: The velocity autocorrelation function resolution ({args.lag_step}) cannot be smaller than the time between frames ({args.timestep}).")
            sys.exit(1)
        self.units = args.units
        self.timestep *= constants.TIMESTEPS[self.units]

        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "vDOS_" + self.outfile

        self.taper_fraction = args.taper
        if args.resolution is None:
            self.padding_total = 0
        else:
            self.padding_total = 1 / (c_cm_s * args.resolution * self.timestep * self.lag_step)

        self.parse_file()

        self.check_required_columns("mass", "vx", "vy", "vz")

        self.autocorrel = list()
        self.autocount = list()

    def analyze(self):
        velocities = list()
        finite_cap = self.max_timelag != np.inf
        if finite_cap:
            max_storage = int(self.max_timelag // self.lag_step) + 1
            self.autocorrel = np.zeros(shape = (max_storage, ))
            self.autocount = np.zeros(shape = (max_storage, ))
        else:
            self.max_timelag = None

        self.history = deque(maxlen = self.max_timelag)

        for frame_idx in self.trajectory_reader():
            if not finite_cap:
                self.autocorrel.append(0)
                self.autocount.append(0)

            current_velocities = self.extract_columns("vx", "vy", "vz")
            current_masses = self.extract_column("mass")
            self.history.append(current_velocities)

            for tau, past_velocities in enumerate(reversed(self.history)):
                if tau % self.lag_step:
                    continue

                storage_idx = tau // self.lag_step

                self.autocount[storage_idx] += 1
                self.autocorrel[storage_idx] += np.einsum("i,ij,ij->", current_masses, current_velocities, past_velocities) / current_velocities.shape[0]


            self.verbose_print(f"{frame_idx} analysis of TS {self.get_timestep()}", verbosity = 2)


        self.num_autocorrel_points = len(self.autocorrel)

        non_zeros = np.flatnonzero(self.autocount)
        if non_zeros.size == 0:
            trim = self.num_autocorrel_points
        else:
            trim = non_zeros[-1] + 1

        self.autocorrel = self.autocorrel[:trim]
        self.autocount = self.autocount[:trim]

        self.num_autocorrel_points = trim

        self.autocorrel = np.array(self.autocorrel) /  np.array(self.autocount)
        self.autocorrel /= self.autocorrel[0]


        self.padding_total = int(max((self.padding_total, self.num_autocorrel_points)))
        self.padding_total = 1 << (self.padding_total - 1).bit_length()

        window = np.ones(self.num_autocorrel_points)
        num_taper = int(self.num_autocorrel_points * self.taper_fraction)
        if num_taper > 0:
            x = np.linspace(0, np.pi, num_taper)
            decay = 0.5 * (1 + np.cos(x))
            window[-num_taper:] = decay

        self.tapered_autocorrel = self.autocorrel * window

        dt_seconds = self.timestep * self.lag_step
        vdos_complex = np.fft.rfft(self.tapered_autocorrel, n = self.padding_total)
        self.vDOS = vdos_complex.real * dt_seconds

        #Conversion from Hz to cm^-1
        self.frequencies = np.fft.rfftfreq(self.padding_total, d = self.timestep * self.lag_step) / c_cm_s

    def write(self):

        super().write(data = np.column_stack((self.frequencies, self.vDOS)),
                      header = "Frequency (cm^-1), vDOS",
                      outfile = self.outfile,
                      )

        if self.get_verbosity() >= 3:
            super().write(data = np.column_stack((np.arange(0, self.num_autocorrel_points, 1) * self.timestep * self.lag_step / constants.TIMESTEPS[self.units], self.autocorrel)),
                          header = "Time, ACF",
                          outfile = "ACF_" + self.outfile,
                          )

    def statistics(self):
        #Name : (value, verbosity)
        stats_dict = {"Simulation dictated resolution (cm^-1)" : (1 / (self.num_autocorrel_points * self.timestep * self.lag_step * c_cm_s), 1),
                      "Numerical resolution (cm^-1)" : (1 / (self.padding_total * self.timestep * self.lag_step * c_cm_s), 1),
                      "Effective time step" : (self.timestep * self.lag_step / constants.TIMESTEPS[self.units], 1),
        }

        super().statistics(stats_dict = stats_dict)
