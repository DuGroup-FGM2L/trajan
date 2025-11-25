from .base_handler import BASE
import numpy as np

class ANGLE(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose)

        self.types = args.types
        self.outfile = args.outfile
        self.bincount = args.bincount

        self.parse_file()

        self.wrap_positions()

        self.bond_angles = list()

    def analyze(self):
        bond_angles = list()

        for i in range (self.Nframes):
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

            #Get displacements between central atoms and each of their nearest neighbors
            displ1 = neighs1[idx1] - central_atoms
            displ2 = neighs2[idx2] - central_atoms

            #Account for periodic boundaries
            displ1 -= self.lengths[i] * np.round(displ1 / self.lengths[i])
            displ2 -= self.lengths[i] * np.round(displ2 / self.lengths[i])


            dotproduct = np.sum(displ1 * displ2, axis=1)
            costheta = np.sum(displ1 * displ2, axis=1) / (norms1 * norms2)
            theta = 180 * np.arccos(costheta) / np.pi

            bond_angles.append(theta)

            self.verbose_print(f"{i + 1} analysis of TS {self.timesteps[i]}")

        self.bond_angles = np.concatenate(bond_angles)

    def write(self):
        counts, edges = np.histogram(self.bond_angles, bins = self.bincount)
        centers = 0.5 * (edges[:-1] + edges[1:])
        super().write(data = np.column_stack([centers, counts]),
                      header = "angle, count",
                      outfile = self.outfile,
                      )
