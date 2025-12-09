from .base_handler import BASE
from trajan import constants
import numpy as np
import sys
import scipy as sp
import scipy.signal

class RDFS(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose)

        self.cutoff = args.cutoff
        self.nbins = args.bincount

        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "rdfs_" + self.outfile

        self.parse_file()
        self.check_required_columns("type", "x", "y", "z")

        if args.pair:
            self.pairs = np.array(args.pair)
        else:
            types = self.get_types()
            self.pairs = types[np.column_stack(np.triu_indices(types.size))]

        self.batch_size = args.batch_size





    def analyze(self):
        timesteps = self.get_timesteps()
        boxes = self.get_boxes()
        sides = np.diff(boxes, axis = -1).reshape((-1, 3))
        self.bins = np.linspace(0, self.cutoff, self.nbins)
        self.hist_counts = np.zeros(shape = (self.pairs.shape[0], self.bins.size - 1), dtype = np.int64)
        r_inner = self.bins[:-1]
        r_outer = self.bins[1:]
        self.edges = (r_outer + r_inner)/2
        shell_volumes = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        self.g_r = np.zeros(shape = (self.pairs.shape[0], self.bins.size - 1), dtype = np.float64)

        for pair_idx, pair in enumerate(self.pairs):
            n1_rho2 = 0
            self.verbose_print(f"Calculating distribution for pair: {pair[0]} {pair[1]}", verbosity = 2)
            for i in range (self.get_Nframes()):
                atoms1 = self.extract_positions(self.select_type(type = pair[0], frame = i))
                atoms2 = self.extract_positions(self.select_type(type = pair[1], frame = i))
                n1 = atoms1.shape[0]
                n2 = atoms2.shape[0]

                for btch_idx in range(0, n1, self.batch_size):
                    batch_atoms1 = atoms1[btch_idx : btch_idx + self.batch_size]
                    delta = batch_atoms1[:, np.newaxis, :] - atoms2[np.newaxis, :, :]
                    delta -= sides[i] * np.around(delta / sides[i])
                    square_dist = np.sum(delta**2, axis = 2)
                    square_dist = square_dist[square_dist <= self.cutoff**2]
                    if pair[0] == pair[1]:
                        square_dist = square_dist[square_dist > 0]
                    dists = np.sqrt(square_dist)
                    counts, _ = np.histogram(dists, bins = self.bins)
                    self.hist_counts[pair_idx] += counts

                n1_rho2 += n1 * n2/np.prod(sides[i])

                self.verbose_print(f"{i + 1} analysis of TS {timesteps[i]}", verbosity = 3)

            self.g_r[pair_idx] += self.hist_counts[pair_idx] / (shell_volumes * n1_rho2)


        print("Analysis complete")

    def write(self):
        data = np.column_stack((self.edges, self.g_r.T))
        #import matplotlib.pyplot as plt
        #plt.scatter(self.edges, self.g_r[0])
        #plt.show()
        header = "r," + ",".join([f"{pair[0]}-{pair[1]}" for pair in self.pairs])
        super().write(data = data,
                      header = header,
                      outfile = self.outfile,
                      )

    def statistics(self):
        verbosity = self.get_verbosity()
        #Name : (value, verbosity)
        stats_dict = dict()
        for pid, pair in enumerate(self.pairs):
            stats_dict[f"Max Peak ({pair[0]}-{pair[1]})"] = (self.edges[np.argmax(self.g_r[pid])], 1)
            if verbosity > 2:
                peaks, _ = sp.signal.find_peaks(self.g_r[pid], height = 1.0, prominence = 0.5)
                print(self.edges[peaks])

        super().statistics(stats_dict = stats_dict)
