from .base_handler import BASE
from trajan import constants
import numpy as np
import sys
import scipy as sp
import scipy.signal
import scipy.integrate
import scipy.spatial

class RDF(BASE):
    def __init__(self, args):
        super().__init__(args)

        self.cutoff = args.cutoff
        self.cutsq = self.cutoff**2
        self.nbins = args.bincount

        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "rdfs_" + self.outfile

        self.parse_file()
        self.check_required_columns("type", "x", "y", "z")

        self.wrap_positions()

        self.calc_coord = args.coordination_number

        scatter_params = args.scatter_partial or args.scatter or []
        self.calc_scatter_partial = not args.scatter_partial is None
        self.calc_scatter = (not (args.scatter is None)) and (not self.calc_scatter_partial)

        if self.calc_scatter_partial:
            scatter_arg = "-sp / --scatter-partial"
        elif self.calc_scatter:
            scatter_arg = "-s / --scatter"
        else:
            scatter_arg = "-s / --scatter or -sp / --scatter-partial"

        mappings = args.total_partial or args.total or []
        self.calc_total_partial = bool(args.total_partial)
        self.calc_total = bool(args.total) and not self.calc_total_partial

        if self.calc_total_partial:
            total_arg = "-tp / --total-partial"
        elif self.calc_total:
            total_arg = "-t / --total"
        else:
            total_arg = "-t / --total or -tp / --total-partial"


        if scatter_params and not mappings:
            self.verbose_print(f"WARNING: Scattering factors' calculation was requested without the scattering lengths provided. Please invoke the {total_arg} option. No scattering will be calculated.")

        if mappings:
            self.mappings = np.array(mappings)
            present_types = self.get_types()
            if self.mappings.size != present_types.size:
                self.verbose_print(f"ERROR: Number of provided mappings for {total_arg} ({self.mappings.size}) is not equal to the number of present types ({present_types.size}).")
                sys.exit(1)
            else:
                self.scat_lengths = np.zeros(shape = (np.max(present_types) + 1, ))
                for i in range(present_types.size):
                    try:
                        self.scat_lengths[present_types[i]] = float(self.mappings[i])
                    except:
                        if self.mappings[i] in constants.NEUTRON_SCATTERING_LENGTHS.keys():
                            self.scat_lengths[present_types[i]] = constants.NEUTRON_SCATTERING_LENGTHS[self.mappings[i]]
                        else:
                            self.verbose_print(f"Element ({self.mappings[i]}) is not currently supported)")
                            input_accepted = False
                            while not input_accepted:
                                scat_input = input("Please proivide custom neutron scattering length: ")
                                try:
                                    self.scat_lengths[present_types[i]] = float(scat_input)
                                    input_accepted = True
                                except:
                                    pass

        if args.pair and not self.calc_total:
            self.pairs = np.array(args.pair)
        else:
            if self.calc_total:
                self.verbose_print("WARNING: You have invoked the total correlation function calculations. For this each pairwise radial distribution function has to be computed. -p/--pairs argument is disregarded.\n")
            types = self.get_types()
            self.pairs = types[np.column_stack(np.triu_indices(types.size))]

        self.batch_size = args.batch_size

        self.broaden = False
        self.Qmax = None
        if args.broaden:
            if not (self.calc_total or self.calc_total_partial):
                self.verbose_print("WARNING: Broadening of the total correlation function requested, without the request for T(r) itself. This will be ignored.\n")
            else:
                self.broaden = True
                self.Qmax = args.broaden

        if self.calc_scatter_partial or self.calc_scatter:
            if len(scatter_params) > 2:
                self.verbose_print(f"WARNING: Too many arguments supplied to {scatter_arg}. Only first two will be interpreted as momentum resolution and maximum momentum.")
                scatter_params = scatter_params[:2]
            if len(scatter_params) == 2:
                self.dq, self.sQmax = scatter_params
            else:
                if self.Qmax is None:
                    self.verbose_print(f"ERROR: No maximum momentum transfer value Q_max was defined. Either provide it by invoking -br / --broaden or as a second argument of {scatter_arg}")
                    sys.exit(1)
                else:
                    self.sQmax = self.Qmax

                if len(scatter_params) == 1:
                    self.dq = args.scatter[0]
                else:
                    self.dq = np.pi / self.cutoff
                    self.verbose_print(f"WARNING: No momentum grid resolution provided. The MD limiting resolution of pi / cutoff will be applied ({round(self.dq, 3)}).\n")




    def analyze(self):
        self.bins = np.linspace(0, self.cutoff, self.nbins)
        self.hist_counts = np.zeros(shape = (self.pairs.shape[0], self.bins.size - 1), dtype = np.int64)
        r_inner = self.bins[:-1]
        r_outer = self.bins[1:]
        self.edges = (r_outer + r_inner)/2
        shell_volumes = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        self.g_r = np.zeros(shape = (self.pairs.shape[0], self.bins.size - 1), dtype = np.float64)
        n1_rho2 = np.zeros(shape = (self.pairs.shape[0], ), dtype = np.float64)

        bin_width = self.cutoff / self.nbins
        inv_bin_width = 1.0 / bin_width

        #For T(r) calculations
        type_bound = self.get_types().size + 1
        type_fractions = np.zeros(shape = (type_bound, ))
        number_density = 0

        for frame_idx in self.trajectory_reader():
            current_type_counts = self.get_num_each_type()
            lengths = self.get_lengths()
            inv_lengths = 1.0 / lengths
            volume = np.prod(lengths)
            natoms = self.get_natoms()
            number_density += natoms / volume
            type_fractions += current_type_counts / natoms
            self.total_n1 = np.zeros(shape = (self.pairs.shape[0], ), dtype = np.int64)

            frame_coords = dict()
            for t in np.unique(self.pairs):
                frame_coords[t] = self.extract_positions(self.select_type(t))

            for pair_idx, pair in enumerate(self.pairs):
                self.verbose_print(f"Calculating distribution for pair: {pair[0]} {pair[1]}", verbosity = 3)
                atoms1 = frame_coords[pair[0]]
                atoms2 = frame_coords[pair[1]]

                if atoms1 is None or atoms2 is None:
                    continue

                n1 = atoms1.shape[0]
                n2 = atoms2.shape[0]
                self.total_n1[pair_idx] += n1



                for i in range(0, n1, self.batch_size):
                    batch_a1 = atoms1[i : i + self.batch_size]

                    delta = batch_a1[:, 0, None] - atoms2[None, :, 0]
                    delta -= lengths[0] * np.rint(delta * inv_lengths[0])
                    d2 = delta**2

                    delta = batch_a1[:, 1, None] - atoms2[None, :, 1]
                    delta -= lengths[1] * np.rint(delta * inv_lengths[1])
                    d2 += delta**2

                    delta = batch_a1[:, 2, None] - atoms2[None, :, 2]
                    delta -= lengths[2] * np.rint(delta * inv_lengths[2])
                    d2 += delta**2

                    d2 = d2.ravel()

                    mask = d2 <= self.cutsq

                    if pair[0] == pair[1]:
                        mask &= (d2 > 1e-8)

                    valid_d2 = d2[mask]

                    if valid_d2.size > 0:
                        dists = np.sqrt(valid_d2)

                        bin_indices = (dists * inv_bin_width).astype(np.int32)

                        bin_indices = bin_indices[bin_indices < self.nbins - 1]

                        counts = np.bincount(bin_indices, minlength = self.nbins - 1)
                        self.hist_counts[pair_idx] += counts


                if pair[0] == pair[1]:
                    n1_rho2[pair_idx] += n1 * (n2 - 1) / volume
                else:
                    n1_rho2[pair_idx] += n1 * n2 / volume


            self.verbose_print(f"{frame_idx} analysis of TS {self.get_timestep()}", verbosity = 2)

        self.g_r += self.hist_counts / (shell_volumes[np.newaxis, :] * n1_rho2[:, np.newaxis])
        nframes = self.get_user_frame()
        if self.calc_coord:
            self.coordination = np.cumsum(self.hist_counts, axis = 1) / (self.total_n1[:, np.newaxis] * nframes)

        if self.calc_total or self.calc_total_partial:
            type_fractions /= nframes
            number_density /= nframes
            average_scat = np.sum(type_fractions * self.scat_lengths)
            self.T_partials = np.zeros_like(self.g_r)
            T0_partials = np.zeros_like(self.g_r)
            pair_weights = np.zeros(shape = (self.pairs.shape[0], ))

            global_factor = 4 * np.pi * self.edges * number_density * 0.01

            for pid, pair in enumerate(self.pairs):
                pair_weights[pid] = np.prod(type_fractions[pair]) * np.prod(self.scat_lengths[pair])
                if pair[0] != pair[1]:
                    pair_weights[pid] *= 2
                self.T_partials[pid] = global_factor * pair_weights[pid] * self.g_r[pid]
                T0_partials[pid] = global_factor * pair_weights[pid]

            self.T_r = np.sum(self.T_partials, axis = 0)

            T_0 = global_factor * (average_scat**2)
            self.oscillations = self.T_r - T_0
            self.oscillations_partial = self.T_partials - T0_partials

            if self.calc_scatter or self.calc_scatter_partial:
                self.q_grid = np.arange(2 * np.pi / self.cutoff, self.sQmax + self.dq, self.dq)

                Q_2d = self.q_grid[:, np.newaxis]
                r_2d = self.edges[np.newaxis, :]

                integrand = self.oscillations[np.newaxis, :] * np.sin(Q_2d * r_2d)
                integral = scipy.integrate.simpson(integrand, x=self.edges, axis=-1)

                self.i_Q = integral / self.q_grid
                self.S_Q = 1.0 + (self.i_Q / (0.01 * (average_scat**2)))

                if self.calc_scatter_partial:
                    self.i_Q_partials = np.zeros((self.pairs.shape[0], self.q_grid.size))
                    self.S_Q_partials = np.zeros_like(self.i_Q_partials)

                    for pid in range(self.pairs.shape[0]):
                        integrand_p = self.oscillations_partial[pid][np.newaxis, :] * np.sin(Q_2d * r_2d)
                        integral_p = scipy.integrate.simpson(integrand_p, x = self.edges, axis=-1)
                        self.i_Q_partials[pid] = integral_p / self.q_grid
                        self.S_Q_partials[pid] = 1.0 + (self.i_Q_partials[pid] / (0.01 * pair_weights[pid]))

            #Lorch Broadening
            if self.broaden:
                dr = self.edges[1] - self.edges[0]
                npoints = int(np.ceil(100 * 2 * np.pi / self.Qmax / dr))
                kernel_r = np.arange(-npoints, npoints + 1) * dr
                kernel = np.zeros_like(kernel_r)

                q_grid = np.linspace(0, self.Qmax, constants.FT_GRID_DENSITY)

                lorch_window = np.sinc(q_grid / self.Qmax)

                for i, r_val in enumerate(kernel_r):
                    integrand = np.cos(q_grid * r_val) * lorch_window

                    kernel[i] = (1 / np.pi) * sp.integrate.simpson(integrand, x=q_grid)

                self.broad_T_osc = sp.signal.convolve(self.oscillations, kernel, mode='same') * dr
                self.broad_T_r = T_0 + self.broad_T_osc

                if self.calc_total_partial:
                    self.broad_T_partials = np.zeros_like(self.T_partials)
                    self.broad_T_partials = sp.signal.convolve(self.oscillations_partial, kernel[np.newaxis, :], mode='same') * dr


        self.verbose_print("Analysis complete")

    def write(self):
        data = np.column_stack((self.edges, self.g_r.T))
        header = "r," + ",".join([f"{pair[0]}-{pair[1]}" for pair in self.pairs])
        if self.calc_total:
            data = np.column_stack((data, self.T_r))
            header += ",T(r)"
        if self.calc_total_partial:
            data = np.column_stack((data, self.T_r, self.oscillations, self.oscillations_partial.T))
            header += ",T(r),T(r) - T0," + ",".join([f"{pair[0]}-{pair[1]} (T(r))" for pair in self.pairs])
        if self.broaden:
            data = np.column_stack((data, self.broad_T_r))
            header += ",broad T(r)"
            if self.calc_total_partial:
                data = np.column_stack((data, self.broad_T_osc, self.broad_T_partials.T))
                header += ",broad T(r) - T0," + ",".join([f"{pair[0]}-{pair[1]} (broad T(r))" for pair in self.pairs])

        if self.calc_coord:
            data = np.column_stack((data, self.coordination.T))
            header += ", " + "(CN),".join([f"{pair[0]}-{pair[1]}" for pair in self.pairs]) + "(CN)"
        super().write(data = data,
                      header = header,
                      outfile = self.outfile,
                      )

        if self.calc_scatter:
            data = np.column_stack((self.q_grid, self.i_Q, self.S_Q))
            header = "Q, i_Q, S_Q"
            super().write(data = data,
                          header = header,
                          outfile = "scatter_" + self.outfile,
                          )
        if self.calc_scatter_partial:
            data = np.column_stack((self.q_grid, self.i_Q, self.i_Q_partials.T, self.S_Q, self.S_Q_partials.T))
            header = "Q, i_Q," + ",".join([f"{pair[0]}-{pair[1]} (i_Q)" for pair in self.pairs])
            header += ",S_Q," +  ",".join([f"{pair[0]}-{pair[1]} (S_Q)" for pair in self.pairs])
            super().write(data = data,
                          header = header,
                          outfile = "scatter_" + self.outfile,
                          )

    def statistics(self):
        verbosity = self.get_verbosity()
        #Name : (value, verbosity)
        stats_dict = dict()
        for pid, pair in enumerate(self.pairs):
            if verbosity > 2:
                peaks, _ = sp.signal.find_peaks(self.g_r[pid], height = 1.0, prominence = 1)
                stats_dict[f"Peaks ({pair[0]}-{pair[1]})"] = (" ".join(self.edges[peaks].astype(str)), 3)
            else:
                stats_dict[f"Max Peak ({pair[0]}-{pair[1]})"] = (self.edges[np.argmax(self.g_r[pid])], 2)


        super().statistics(stats_dict = stats_dict)

