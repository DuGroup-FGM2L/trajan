from .base_handler import BASE
from trajan import constants
import pandas as pd
import numpy as np
import scipy as sp
import scipy.integrate

import sys

class ANGLE(BASE):
    def __init__(self, args):
        super().__init__(args)

        self.types = args.types
        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "angle_" + self.outfile
        self.bincount = args.bincount

        numcuts = len(args.cutoffs)
        self.cutoffs = [np.inf, np.inf]
        if numcuts:
            if numcuts >= 2:
                if numcuts == 2:
                    print(f"WARNING: Too many cutoffs specified ({numcuts}). Recorded: C({self.types[0]}, {self.types[1]}) = {args.cutoffs[0]}, C({self.types[1]}, {self.types[2]}) = {args.cutoffs[1]}")
                self.cutoffs = args.cutoffs[:2]
            elif numcuts == 1:
                if self.types[0] == self.types[2]:
                    print("COMMENT: Only one cutoff specified but non-central bonded types are the same. Will use one cutoff for both.")

                elif self.types[0] != self.types[2]:
                    print(f"WARNING: Only one cutoff specified and non-central bonded types are NOT the same. Will only set one cutoff: C({self.types[0]}, {self.types[1]}) = {args.cutoffs[0]}.")
                self.cutoffs = [args.cutoffs[0], args.cutoffs[0]]

        self.parse_file()

        self.check_required_columns("type", "x", "y", "z")

        self.wrap_positions()

        self.bond_angles = list()

        self.angle_hist = None

    def analyze(self):
        bond_angles = list()

        for frame_idx in self.trajectory_reader():
            central_atoms = self.extract_positions(
                    target_array = self.select_type(
                        type = self.types[1],
                        ),
                    )


            neighs1 = self.extract_positions(
                    target_array = self.select_type(
                        type = self.types[0],
                        ),
                    )


            #If types are the same first and second neighbors are needed
            if self.types[0] == self.types[2]:
                neighs2 = neighs1
                idx = self.get_within(
                        central = central_atoms,
                        neighs = neighs1,
                        cut = max(self.cutoffs[0], self.cutoffs[1]),
                        )

                neighbor_frame = pd.DataFrame({
                    "neigh_idx" : idx
                })
                neighbor_frame = neighbor_frame.explode("neigh_idx")
                try:
                    pair_frame = neighbor_frame.merge(neighbor_frame, left_index = True, right_index = True, suffixes = ("_1", "_2")).query("neigh_idx_1 < neigh_idx_2")
                except MemoryError:
                    self.verbose_print(f"ERROR: Not enough memory for pairwise neighborlist generation. The cutoff ({self.cutoffs[0]}) is too large.")
                    sys.exit(1)

            #If types are different first neighbor of each type is needed
            else:
                neighs2 = self.extract_positions(
                    target_array = self.select_type(
                        type = self.types[2],
                        ),
                    )

                idx1 = self.get_within(
                        central = central_atoms,
                        neighs = neighs1,
                        cut = self.cutoffs[0],
                        )

                idx2 = self.get_within(
                        central = central_atoms,
                        neighs = neighs2,
                        cut = self.cutoffs[1],
                        )


                neigh_frame1 = pd.DataFrame({
                    "neigh_idx" : idx1
                })
                neigh_frame1 = neigh_frame1.explode("neigh_idx")

                neigh_frame2 = pd.DataFrame({
                    "neigh_idx" : idx2
                })
                neigh_frame2 = neigh_frame2.explode("neigh_idx")

                try:
                    pair_frame = neigh_frame1.merge(neigh_frame2, left_index = True, right_index = True, suffixes = ("_1", "_2"))
                except MemoryError:
                    self.verbose_print(f"ERROR: Not enough memory for pairwise neighborlist generation. One of the cutoffs ({self.cutoffs[0]}, {self.cutoffs[1]}) is too large.")
                    sys.exit(1)

            central_idx = pair_frame.index.values.astype(int)
            idx1 = pair_frame["neigh_idx_1"].values.astype(int)
            idx2 = pair_frame["neigh_idx_2"].values.astype(int)

            #Get displacements between central atoms and each of their nearest neighbors
            displ1 = neighs1[idx1] - central_atoms[central_idx]
            displ2 = neighs2[idx2] - central_atoms[central_idx]

            #Account for periodic boundaries
            lengths = self.get_lengths()
            displ1 -= lengths * np.round(displ1 / lengths)
            displ2 -= lengths * np.round(displ2 / lengths)

            #Calculate vector norms
            norms1 = np.linalg.norm(displ1, axis = 1)
            norms2 = np.linalg.norm(displ2, axis = 1)


            dotproduct = np.sum(displ1 * displ2, axis=1)
            costheta = np.sum(displ1 * displ2, axis=1) / (norms1 * norms2)
            theta = 180 * np.arccos(costheta) / np.pi

            bond_angles.append(theta)

            self.verbose_print(f"{frame_idx} analysis of TS {self.get_timestep()}", verbosity = 2)

        print("Analysis complete")
        self.bond_angles = np.concatenate(bond_angles)

    def write(self):

        super().write(data = np.column_stack((self.angle_hist, self.angle_norm_hist[:, 1])),
                      header = "angle, count, area normalized count",
                      outfile = self.outfile,
                      )

    def statistics(self):
        counts, edges = np.histogram(self.bond_angles, bins = self.bincount)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = counts / self.get_user_frame()
        self.angle_hist = np.column_stack([centers, counts])

        hist_min = np.min(centers)
        hist_max = np.max(centers)
        area_under_curve = np.sum(counts) * (hist_max - hist_min) / self.bincount
        self.angle_norm_hist = np.column_stack([centers, counts / area_under_curve])


        #Name : (value, verbosity)
        stats_dict = {"Mean bond angle" : (np.mean(self.bond_angles), 1),
                      "Standard deviation" : (np.std(self.bond_angles), 1),
        }

        if self.get_verbosity() >= 3:
            kde = sp.stats.gaussian_kde(self.bond_angles)
            x_grid = np.linspace(hist_min, hist_max, self.bincount)
            smooth_y = kde(x_grid)

            local_maxima_indices, _ = sp.signal.find_peaks(smooth_y)
            local_minima_indices, _ = sp.signal.find_peaks(-smooth_y)
            boundaries = sorted(np.concatenate(([0], local_minima_indices, [len(x_grid)-1])))

            num_peaks = len(local_maxima_indices)

            peaks = np.empty(shape = (num_peaks, 2))

            for i in range(num_peaks):
                idx_start = int(boundaries[i])
                idx_end = int(boundaries[i+1])

                y_slice = smooth_y[idx_start:idx_end]
                x_slice = x_grid[idx_start:idx_end]

                area = sp.integrate.simpson(y=y_slice, x=x_slice)

                peak_loc = x_grid[local_maxima_indices[i]]
                peaks[i] = np.array([peak_loc, area])

            sorted_peaks = peaks[peaks[:, 1].argsort()[::-1]]


            for peak, fraction in sorted_peaks:
                stats_dict[f"Peak at {peak:.1f}° bond fraction"] = (fraction, 3)


            stats_dict["Integrations bounds"] = (x_grid[boundaries], 3)


        super().statistics(stats_dict = stats_dict)
