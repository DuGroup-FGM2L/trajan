from .base_handler import BASE
from trajan import constants
import numpy as np
import sys
import collections

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

class RINGS(BASE):
    def __init__(self, args):
        paral_frame = not args.paral_mode.lower().startswith("a")
        super().__init__(args.file, args.verbose, args.steps, paral_frame = paral_frame)


        self.comm = self.get_comm()
        self.rank = self.get_rank()
        self.size = self.get_size()

        if paral_frame and self.size > 1:
            self.verbose_print("WARNING: Frame parallelization not yet implemented. Will run in atom parallelization mode.")

        self.outfile = args.outfile
        if self.outfile == constants.DEFAULT_OUTFILE:
            self.outfile = "rings_" + self.outfile

        if 0 in args.types:
            zero_index = args.types.index(0)
            self.base = np.array(args.types[:zero_index])
            self.connectors = np.array(args.types[zero_index + 1:])
        else:
            self.verbose_print("ERROR: There must be a separating 0 in atom types. Please refer to the help message for the ring size analyzer.")
            sys.exit(1)

        algo_letter = args.algorithm[0].lower()
        if algo_letter == "s":
            self.algo = self.smallest_rings
        elif algo_letter == "p":
            self.algo = self.primitive_rings


        nbase = len(self.base)
        nconnectors = len(self.connectors)
        ncuts = len(args.cutoffs)

        if args.cutoffs and ncuts != nbase * nconnectors:
            self.verbose_print(f"ERROR: number of cutoffs ({ncuts}) is not equal to the number of base ({nbase}) atoms times the number of connector atoms ({nconnectors}).")
            sys.exit(1)

        self.parse_file()

        self.check_required_columns("type", "x", "y", "z")

        types = self.get_types()
        max_type = np.max(types)
        self.interaction_matrix = np.full((max_type + 1, max_type + 1), np.inf)

        if args.cutoffs:
            cutoff_iter = iter(args.cutoffs)
            for b in self.base:
                if not b in types:
                    self.verbose_print(f"ERROR: Incorrect base atom type ({b}). Types present in trajectories: {' '.join(types.astype(str))}.")
                    sys.exit(1)
                for c in self.connectors:
                    if not c in types:
                        self.verbose_print(f"ERROR: Incorrect connector type ({c}). Types present in trajectories: {' '.join(types.astype(str))}.")
                        sys.exit(1)
                    cut = next(cutoff_iter)
                    self.interaction_matrix[b, c] = cut
                    self.interaction_matrix[c, b] = cut

        self.conbonds = args.connector_bonds
        nconbonds = len(self.conbonds)
        if nconbonds < nconnectors:
            if self.conbonds:
                self.verbose_print(f"WARNING: Number of connectors supplied ({nconnectors}) is not the same as the number of connector bonds arguments ({nconbonds}). Only the first {nconbonds} onnectors will have non default values.")
            for i in range(nconnectors - nconbonds):
                self.conbonds.append(constants.DEFAULT_CONNECTOR_BONDS)
        elif nconbonds > nconnectors:
            self.verbose_print(f"WARNING: Too many connector bond arguments provided. Only the first ({nconnectors}) arguments will be used.")
            self.conbonds = self.conbonds[:nconnectors]

        self.conbonds = np.array(self.conbonds)

        self.max_depth = args.max_size

        self.all_rings = list()
        self.ring_values = np.arange(self.max_depth + 1)


        self.wrap_positions()

    def build_supercell(self, positions, types, max_dist):
        box = self.get_lengths()
        
        n_repeats = np.ceil(max_dist / box).astype(int)
        
        x_range = range(-n_repeats[0], n_repeats[0] + 1)
        y_range = range(-n_repeats[1], n_repeats[1] + 1)
        z_range = range(-n_repeats[2], n_repeats[2] + 1)

        
        translations = []
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    if x == 0 and y == 0 and z == 0: continue
                    translations.append(np.array([x, y, z]) * box)
        
        translations = np.array(translations)
        
        # 1. Start with original atoms (Indices 0 to N-1)
        super_pos = [positions]
        super_types = [types]
        
        # 2. Append all images
        for trans in translations:
            super_pos.append(positions + trans)
            super_types.append(types)
            
        return np.concatenate(super_pos), np.concatenate(super_types)


    def analyze(self):
        max_search_dist = (self.max_depth + 1) * np.max(self.interaction_matrix[np.isfinite(self.interaction_matrix)]) * 2
        if self.rank == 0:
            reader = self.trajectory_reader()
            columns = self.get_columns()

        done = False
        while not done:
            frame_data_bundle = None
            if self.rank == 0:
                try:
                    frame_idx = next(reader)

                    num_each_type = self.get_num_each_type()
                    num_neighs = int(2 * np.sum(num_each_type[self.connectors] * self.conbonds) / np.sum(num_each_type[self.base]))

                    orig_base_atoms = self.select_types(types = self.base)
                    orig_base_pos = self.extract_positions(orig_base_atoms)
                    orig_base_types = orig_base_atoms[..., columns["type"]].astype(int)
                 
                    num_original_base = orig_base_atoms.shape[0]

                    sc_base_pos, sc_base_types = self.build_supercell(orig_base_pos, orig_base_types, max_search_dist)

                    num_sc_base = sc_base_pos.shape[0]

                    frame_graph = np.full(shape = (num_sc_base, num_neighs), fill_value = -1)
                    neighs_recorder = np.zeros(shape = (num_sc_base, ), dtype = np.int32)

                    for cid, connector in enumerate(self.connectors):
                        orig_conn_atoms = self.select_type(connector)
                        orig_conn_pos = self.extract_positions(orig_conn_atoms)
                        orig_conn_types = orig_conn_atoms[..., columns["type"]].astype(int)
                        sc_conn_pos, sc_conn_types = self.build_supercell(orig_conn_pos, orig_conn_types, max_search_dist)

                        numbonds = self.conbonds[cid]
                        norms, idx = self.get_nclosest(
                            central = sc_conn_pos,
                            neighs = sc_base_pos,
                            N = numbonds,
                            use_pbc = False,
                        )

                        cutoff_grid = self.interaction_matrix[sc_conn_types[:, None], sc_base_types[idx]]

                        distance_mask = norms <= cutoff_grid


                        sources = list()
                        targets = list()

                        #Create flat lists of base atoms connected through a connector
                        for i in range(numbonds):
                            for j in range(numbonds):
                                if i == j: continue

                                valid_pairs_mask = distance_mask[:, i] & distance_mask[:, j]
                                sources.append(idx[valid_pairs_mask, i])
                                targets.append(idx[valid_pairs_mask, j])

                        all_sources = np.concatenate(sources)
                        all_targets = np.concatenate(targets)

                        if all_sources.size == 0 or all_targets.size == 0:
                            self.verbose_print("ERROR: No atoms are connected. Faild to construct a proper graph.")
                            sys.exit(1)



                        sort_order = np.argsort(all_sources)
                        all_sources = all_sources[sort_order]
                        all_targets = all_targets[sort_order]

                        #Find at which index repetition starts
                        unique_atoms, start_indices = np.unique(all_sources, return_index = True)
                        #Calculate how many source atoms repeated (num of neighs for each base)
                        neigh2add = np.diff(np.append(start_indices, len(all_sources)))

                        #Get relative position of each target atom in original graph
                        group_offsets = np.repeat(start_indices, neigh2add)
                        relcol = np.arange(len(all_sources)) - group_offsets

                        #Fetch how many neighbors each base atom has already
                        shift = neighs_recorder[all_sources]
                        #Shift relative column positions for new neighbors
                        relcol = relcol + shift

                        #Allocate more memory for new neighbors
                        max_new_numneigh = np.max(relcol)
                        if max_new_numneigh >= num_neighs:
                            extend_to = max_new_numneigh * 2
                            new_frame_graph = np.full((num_sc_base, extend_to), fill_value = -1, dtype = np.int32)
                            new_frame_graph[:, :num_neighs] = frame_graph
                            num_neighs = extend_to
                            frame_graph = new_frame_graph

                        frame_graph[all_sources, relcol] = all_targets

                        neighs_recorder[unique_atoms] += neigh2add


                    timestep = self.get_timestep()
                    frame_data_bundle = {
                        'graph': frame_graph,
                        'n_atoms': num_original_base,
                        'frame_idx': frame_idx,
                        'timestep': timestep,
                    }

                except StopIteration:
                    done = True

            if HAS_MPI: done = self.comm.bcast(done, root = 0)
            if not done:
                if HAS_MPI:
                    data = self.comm.bcast(frame_data_bundle, root = 0)

                    frame_graph = data["graph"]
                    num_original_base = data["n_atoms"]
                    frame_idx = data["frame_idx"]
                    timestep = data["timestep"]

                    if self.rank == 0:
                        atom_indices = np.arange(num_original_base, dtype = np.int32)
                        chunks = np.array_split(atom_indices, self.size)
                    else:
                        chunks = None

                    my_atoms = self.comm.scatter(chunks, root = 0)
                else:
                    my_atoms = range(num_original_base)

                my_counts = self.algo(frame_graph, frame = frame_idx, atom_inds = my_atoms, original_atom_count = num_original_base)

                if HAS_MPI:
                    ring_participation_counts = self.comm.reduce(my_counts, op=MPI.SUM, root=0)
                else:
                    ring_participation_counts = my_counts



                if self.rank == 0:
                    self.all_rings.append(np.mean(ring_participation_counts, axis = 0))

                self.verbose_print(f"{frame_idx} analysis of TS {timestep}", verbosity = 2)

        if self.rank == 0:
            self.all_rings = np.array(self.all_rings)

        self.verbose_print("Analysis complete")

    def smallest_rings(self, frame_graph, frame, atom_inds, original_atom_count):
        ring_participation_counts = np.zeros(shape = (original_atom_count, self.max_depth + 1))
        visited_token = np.full(num_base_atoms, fill_value = -1, dtype = np.int32)
        dist = np.zeros(num_base_atoms, dtype = np.int32)
        parent = np.full(num_base_atoms, fill_value = -1, dtype = np.int32)
        branch_id = np.full(num_base_atoms, -1, dtype = np.int32)

        queue = collections.deque()

        chunk_length = len(atom_inds)
        for atom_ord, start_node in enumerate(atom_inds):
            self.verbose_print(f"{100*atom_ord/chunk_length:.2f}% of timestep {timestep} (frame {frame})", verbosity = 3)

            if frame_graph[start_node, 0] == -1:
                continue

            current_token = start_node
            queue.clear()

            queue.append(start_node)
            visited_token[start_node] = current_token
            dist[start_node] = 0
            parent[start_node] = -1
            branch_id[start_node] = -1

            found_ring = False

            while queue:
                node = queue.popleft()
                if dist[node] >= int(self.max_depth / 2 ) + 1:
                    continue

                for neigh in frame_graph[node]:
                    if neigh == -1: break

                    if neigh == parent[node]: continue

                    if visited_token[neigh] == current_token:
                        if branch_id[node] != branch_id[neigh]:
                            current_size = dist[node] + dist[neigh] + 1
                            if current_size > self.max_depth:
                                continue

                            bt_atom = node
                            while parent[bt_atom] != -1:
                                ring_participation_counts[bt_atom, current_size] += 1
                                bt_atom = parent[bt_atom]

                            bt_atom = neigh
                            while bt_atom != -1:
                                ring_participation_counts[bt_atom, current_size] += 1
                                bt_atom = parent[bt_atom]



                            found_ring = True
                            break

                    else:
                        visited_token[neigh] = current_token
                        dist[neigh] = dist[node] + 1
                        parent[neigh] = node
                        if node == start_node:
                            branch_id[neigh] = neigh
                        else:
                            branch_id[neigh] = branch_id[node]
                        queue.append(neigh)

                if found_ring: break

        return ring_participation_counts

    def primitive_rings(self, frame_graph, frame, atom_inds, original_atom_count):
        num_base_atoms = frame_graph.shape[0]

        ring_participation_counts = np.zeros(shape=(original_atom_count, self.max_depth + 1))

        visited_token = np.full(num_base_atoms, fill_value=-1, dtype=np.int32)
        dist = np.zeros(num_base_atoms, dtype=np.int32)
        parent = np.full(num_base_atoms, fill_value=-1, dtype=np.int32)
        branch_id = np.full(num_base_atoms, -1, dtype=np.int32)

        queue = collections.deque()

        search_limit = (self.max_depth // 2) + 1
        timestep = self.get_timestep()

        chunk_length = len(atom_inds)
        for atom_ord, start_node in enumerate(atom_inds):
            self.verbose_print(f"{100*atom_ord/chunk_length:.2f}% of timestep {timestep} (frame {frame})", verbosity = 3)
            if frame_graph[start_node, 0] == -1:
                continue

            current_token = start_node
            queue.clear()
            queue.append(start_node)

            visited_token[start_node] = current_token
            dist[start_node] = 0
            parent[start_node] = -1
            branch_id[start_node] = -1

            while queue:
                node = queue.popleft()

                if dist[node] >= search_limit:
                    continue

                for neigh in frame_graph[node]:
                    if neigh == -1: break
                    if neigh == parent[node]: continue

                    if visited_token[neigh] == current_token:

                        if branch_id[node] != branch_id[neigh]:

                            if node > neigh: continue

                            current_size = dist[node] + dist[neigh] + 1

                            if current_size <= self.max_depth:

                                ring_atoms = list()
                                bt_atom = neigh
                                while bt_atom != -1:
                                    ring_atoms.append(bt_atom)
                                    bt_atom = parent[bt_atom]
                                ring_atoms = ring_atoms[::-1]

                                bt_atom = node
                                while parent[bt_atom] != -1:
                                    ring_atoms.append(bt_atom)
                                    bt_atom = parent[bt_atom]

                                is_primitive = True
                                half_ring = current_size // 2
                                for i in range(half_ring):
                                    shortest_path = self.shortest_path(frame_graph, ring_atoms[i], ring_atoms[half_ring + i], half_ring)
                                    is_primitive *= (shortest_path >= half_ring)


                                if is_primitive:
                                    ring_participation_counts[start_node, current_size] += 1

                    else:
                        visited_token[neigh] = current_token
                        dist[neigh] = dist[node] + 1
                        parent[neigh] = node

                        if node == start_node:
                            branch_id[neigh] = neigh
                        else:
                            branch_id[neigh] = branch_id[node]

                        queue.append(neigh)


        return ring_participation_counts

    def shortest_path(self, frame_graph, start, target, limit=None):
        if start == target:
            return 0

        visited_fwd = {start: 0}
        visited_bwd = {target: 0}

        q_fwd = collections.deque([start])
        q_bwd = collections.deque([target])

        shortest_dist = float('inf')

        while q_fwd and q_bwd:
            if len(q_fwd) <= len(q_bwd):
                active_q = q_fwd
                active_visited = visited_fwd
                other_visited = visited_bwd
            else:
                active_q = q_bwd
                active_visited = visited_bwd
                other_visited = visited_fwd

            curr = active_q.popleft()
            curr_dist = active_visited[curr]

            if limit is not None and (curr_dist >= limit):
                continue

            if curr_dist + 1 >= shortest_dist:
                continue

            for neigh in frame_graph[curr]:
                if neigh == -1: break

                if neigh in other_visited:
                    found_dist = curr_dist + 1 + other_visited[neigh]

                    if found_dist < shortest_dist:
                        shortest_dist = found_dist

                        if limit is not None and shortest_dist < limit:
                            return shortest_dist

                if neigh not in active_visited:
                    if limit is None or (curr_dist + 1 < limit):
                        active_visited[neigh] = curr_dist + 1
                        active_q.append(neigh)

        return shortest_dist

    def write(self):

        super().write(data = np.column_stack((self.ring_values, self.mean_rings)),
                      header = "Ring size, ",
                      outfile = self.outfile,
                      )

    def statistics(self):
        if self.rank == 0:
            self.mean_rings = np.mean(self.all_rings, axis = 0)
            self.deviations = np.std(self.all_rings, axis = 0)
            #Name : (value, verbosity)
            stats_dict = {"Average ring size" : (np.sum(self.ring_values * self.mean_rings / np.sum(self.mean_rings)), 1),
            }


            super().statistics(stats_dict = stats_dict)
        else:
            self.mean_rings = np.zeros_like(self.ring_values)
