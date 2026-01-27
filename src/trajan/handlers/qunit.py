from .base_handler import BASE
from trajan import constants
import numpy as np
import sys

class QUNIT(BASE):
    def __init__(self, args):
        super().__init__(args.file, args.verbose, args.steps, args.buffer)

        self.qunits = None
        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "qunit_" + self.outfile

        if 0 in args.types:
            zero_index = args.types.index(0)
            self.formers = args.types[:zero_index]
            self.connectors = args.types[zero_index + 1:]
        else:
            print("ERROR: There must be a separating 0 in qunit types. Please refer to the help message for the qunit analyzer.")
            sys.exit(1)


        nformers = len(self.formers)
        nconnectors = len(self.connectors)
        ncuts = len(args.cutoffs)

        if args.cutoffs and ncuts != nformers * nconnectors:
            print(f"ERROR: number of cutoffs ({ncuts}) is not equal to the number of formers ({nformers}) times the number of connectors ({nconnectors}).")
            sys.exit(1)

        self.parse_file()

        self.check_required_columns("type", "x", "y", "z")

        types = self.get_types()
        max_type = np.max(types)
        self.interaction_matrix = np.full((max_type + 1, max_type + 1), np.inf)

        if args.cutoffs:
            cutoff_iter = iter(args.cutoffs)
            for f in self.formers:
                if not f in types:
                    print(f"ERROR: Incorrect former type ({f}). Types present in trajectories: {' '.join(types.astype(str))}.")
                    sys.exit(1)
                for c in self.connectors:
                    if not c in types:
                        print(f"ERROR: Incorrect connector type ({c}). Types present in trajectories: {' '.join(types.astype(str))}.")
                        sys.exit(1)
                    cut = next(cutoff_iter)
                    self.interaction_matrix[f, c] = cut
                    self.interaction_matrix[c, f] = cut



        self.wrap_positions()


    def analyze(self):
        total_formers = 0
        all_coordinations = list()
        columns = self.get_columns()
        for frame_idx in self.trajectory_reader():
            former_atoms = self.select_types(
                    types = self.formers,
                )

            total_formers += former_atoms.shape[0]

            non_connectors = self.filter_types(
                types = self.connectors,
            )


            non_connector_positions = self.extract_positions(target_array = non_connectors)

            network_connectors = self.select_types(
                    types = self.connectors,
            )

            network_connector_types = network_connectors[:, columns["type"]].astype(int)

            network_connector_positions = self.extract_positions(target_array = network_connectors)

            norms, idx = self.get_nclosest(
                central = network_connector_positions,
                neighs = non_connector_positions,
                N = 2,
            )

            neigh_types = non_connectors[idx][..., columns["type"]].astype(int)


            cutoff_grid = self.interaction_matrix[network_connector_types[:, None], neigh_types]

            distance_mask = norms <= cutoff_grid
            idx = idx[distance_mask.all(axis = 1)]
            neigh_data = non_connectors[idx]
            neigh_types = neigh_data[..., columns["type"]].astype(int)

            neigh_is_former = np.isin(neigh_types, self.formers)
            both_neighs_former = neigh_is_former.all(axis = 1)

            bridging_neighs = neigh_data[..., columns["id"]][both_neighs_former]
            unique_ids, counts = np.unique(bridging_neighs, return_counts = True)
            all_coordinations.append(counts)

            self.verbose_print(f"{frame_idx + 1} analysis of TS {self.get_timestep()}", verbosity = 2)

        self.all_coordinations = np.concatenate(all_coordinations)
        unique_coords, counts = np.unique(self.all_coordinations, return_counts = True)
        self.qvalues = unique_coords
        self.qfractions = counts/total_formers
        self.qunits = dict(zip(unique_coords, self.qfractions))

        print("Analysis complete")

    def write(self):

        super().write(data = np.column_stack((self.qvalues, self.qfractions)),
                      header = "Q, f",
                      outfile = self.outfile,
                      )

    def statistics(self):
        #Name : (value, verbosity)
        stats_dict = {"Average Q-unit" : (np.sum(self.qvalues * self.qfractions), 1),
        }


        super().statistics(stats_dict = stats_dict)
