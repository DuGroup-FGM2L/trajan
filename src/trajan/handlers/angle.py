from .base_handler import BASE
from trajan import constants
import numpy as np
import scipy as sp
import scipy.integrate

class ANGLE(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose)

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

        for i in range (self.get_Nframes()):
            central_atoms = self.extract_positions(
                    target_array = self.select_type(
                        type = self.types[1],
                        frame = i,
                        ),
                    )


            neighs1 = self.extract_positions(
                    target_array = self.select_type(
                        type = self.types[0],
                        frame = i,
                        ),
                    )


            #If types are the same first and second neighbors are needed
            if self.types[0] == self.types[2]:
                neighs2 = neighs1
                norms, idx = self.get_nclosest(
                        central = central_atoms,
                        neighs = neighs1,
                        N = 2,
                        box = self.lengths[i],
                        )
                norms1, norms2 = norms.T
                idx1, idx2 = idx.T

            #If types are different first neighbor of each type is needed
            else:
                neighs2 = self.extract_positions(
                    target_array = self.select_type(
                        type = self.types[2],
                        frame = i,
                        ),
                    )

                norms1, idx1 = self.get_nclosest(
                        central = central_atoms,
                        neighs = neighs1,
                        N = 1,
                        box = self.lengths[i],
                        )

                norms2, idx2 = self.get_nclosest(
                        central = central_atoms,
                        neighs = neighs2,
                        N = 1,
                        box = self.lengths[i],
                        )

            #Check against cutoffs
            below = (norms1 < self.cutoffs[0]) * (norms2 < self.cutoffs[1])

            norms1 = norms1[below]
            norms2 = norms2[below]
            idx1 = idx1[below]
            idx2 = idx2[below]

            #Get displacements between central atoms and each of their nearest neighbors
            displ1 = neighs1[idx1] - central_atoms[below]
            displ2 = neighs2[idx2] - central_atoms[below]

            #Account for periodic boundaries
            displ1 -= self.lengths[i] * np.round(displ1 / self.lengths[i])
            displ2 -= self.lengths[i] * np.round(displ2 / self.lengths[i])


            dotproduct = np.sum(displ1 * displ2, axis=1)
            costheta = np.sum(displ1 * displ2, axis=1) / (norms1 * norms2)
            theta = 180 * np.arccos(costheta) / np.pi

            bond_angles.append(theta)

            self.verbose_print(f"{i + 1} analysis of TS {self.get_timesteps()[i]}", verbosity = 2)

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
