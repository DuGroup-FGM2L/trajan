from .base_handler import BASE
from trajan import constants

import numpy as np
import sys
from collections import deque

c_cm_s = constants.c * 100

class FTIR(BASE):
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
        if args.lag_step is None:
            self.lag_step = constants.DEFAULT_LAG_STEP
        else:
            self.lag_step = int(args.lag_step * frame_step / self.timestep)


        if self.lag_step < 1:
            print(f"ERROR: The charge flux autocorrelation function resolution ({args.lag_step}) cannot be smaller than the time between frames ({args.timestep}).")
            sys.exit(1)
        self.units = args.units
        self.timestep *= constants.TIMESTEPS[self.units]

        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "IR_" + self.outfile

        self.taper_fraction = args.taper
        if args.resolution is None:
            self.padding_total = 0
        else:
            self.padding_total = 1 / (c_cm_s * args.resolution * self.timestep * self.lag_step)

        self.parse_file()

        self.correct_QM = not (args.quantum_correction is None)
        if self.correct_QM:
            self.sim_temp = args.quantum_correction

        self.check_required_columns("q", "vx", "vy", "vz")

        self.autocorrel = list()
        self.autocount = list()

    def analyze(self):
        velocities = list()
        finite_cap = self.max_timelag != np.inf
        if finite_cap:
            self.max_timelag = int(args.max_timelag / self.timestep)
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
            current_charges = self.extract_columns("q")
            current_flux = np.sum(current_charges * current_velocities, axis = 0)
            self.history.append(current_flux)

            for tau, past_velocities in enumerate(reversed(self.history)):
                if tau % self.lag_step:
                    continue

                storage_idx = tau // self.lag_step

                self.autocount[storage_idx] += 1
                self.autocorrel[storage_idx] += np.einsum("i,i->", current_flux, past_velocities)


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
        ftir_complex = np.fft.rfft(self.tapered_autocorrel, n = self.padding_total)
        self.ftir = ftir_complex.real
        self.ftir *= (dt_seconds * (constants.electron * constants.DISTANCE_CONVERSIONS[self.units] * 0.01 / constants.TIMESTEPS[self.units])**2)

        #Conversion from Hz to cm^-1
        self.frequencies = np.fft.rfftfreq(self.padding_total, d = dt_seconds) / c_cm_s

        #Frequencty-dependent correction
        if self.correct_QM:
            h_nu = constants.h * self.frequencies * c_cm_s
            kBT = constants.kB * self.sim_temp
            h_nu_over_kBT = h_nu/kBT
            correction = np.ones_like(h_nu_over_kBT)
            mask = h_nu_over_kBT > 0
            correction[mask] = h_nu_over_kBT[mask] / (1 - np.exp(-h_nu_over_kBT[mask]))
            self.ftir *= correction

        sim_volume = np.prod(self.get_lengths()) * (constants.DISTANCE_CONVERSIONS[self.units] * 0.01)**3
        self.ftir /= (sim_volume * 3 * kBT )

    def write(self):

        super().write(data = np.column_stack((self.frequencies, self.ftir)),
                      header = "Frequency (cm^-1), ftir",
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
