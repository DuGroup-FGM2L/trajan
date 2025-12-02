import numpy as np
import scipy as sp
import sys

from trajan import constants

class BASE():
    def __init__(self, filename, verbosity):
        self.__trajectory = filename
        self.__verbosity = verbosity

        #Per-snapshot arrays
        self.__boxes = None
        self.__box_periods = None
        self.__atom_counts = None
        self.__timesteps = None
        self.__types = None

        #Per-snapshot 2D arrays
        self.__atomic_data = None

        #General data
        self.__columns = dict()
        self.__Nframes = 0

    def get_trajectory(self):
        return self.__trajectory

    def get_verbosity(self):
        return self.__verbosity

    def get_boxes(self):
        return self.__boxes

    def get_box_periods(self):
        return self.__box_periods

    def get_atom_counts(self):
        return self.__atom_counts

    def get_timesteps(self):
        return self.__timesteps

    def get_types(self):
        return self.__types

    def get_atomic_data(self):
        return self.__atomic_data

    def get_columns(self):
        return self.__columns

    def get_Nframes(self):
        return self.__Nframes

    def parse_file(self):
        self.verbose_print(f"Scanning trajectory file: {self.__trajectory}")
        read_natoms = False
        read_timestep = False
        all_data = list()
        all_boxes = list()
        all_timesteps = list()
        atom_counts = list()
        atomic_counts = list()
        box_period = list()
        column_headers = list()
        box_ctr = float("inf")
        atom_ctr = float("inf")
        natoms = 0

        #Max timestep set for testing
        max_ts =  float("inf")
        with open(self.__trajectory, "r") as f:
            for line in f:
                if self.__Nframes > max_ts:
                    continue
                if "ITEM: TIMESTEP" in line:
                    read_timestep = True
                elif read_timestep:
                    timestep = int(line.strip())
                    self.verbose_print(f"{self.__Nframes} scan of TS {timestep}", verbosity = 2)
                    all_timesteps.append(timestep)
                    read_timestep = False

                if "ITEM: NUMBER OF ATOMS" in line:
                    read_natoms = True
                elif read_natoms:
                    natoms = int(line.strip())
                    read_natoms = False
                    atomic_counts.append(natoms)

                if "ITEM: BOX BOUNDS" in line:
                    box = np.empty((3, 2))
                    box_ctr = 0
                    box_period.append([line.split()[-3:]])
                elif box_ctr < 3:
                    box[box_ctr] = np.fromstring(line, sep = " ", count = 2)
                    box_ctr += 1


                if "ITEM: ATOMS" in line:
                    atom_ctr = 0
                    max_type = -np.inf
                    atom_data = list()
                    column_headers = line.split()[2:]
                elif atom_ctr < natoms:
                    atom_data.append(np.fromstring(line, sep = " "))
                    atom_ctr += 1
                elif atom_ctr == natoms:
                    all_data.append(atom_data)
                    all_boxes.append(box)
                    atom_ctr = float("inf")
                    self.__Nframes += 1

        all_data.append(atom_data)
        all_boxes.append(box)
        self.__Nframes += 1

        for i, column_heading in enumerate(column_headers):
            self.__columns[column_heading] = i

        self.__boxes = np.array(all_boxes)
        self.__atomic_data = np.array(all_data)
        self.__atom_counts = np.array(atomic_counts)
        self.__box_periods = np.array(box_period)
        self.__timesteps = np.array(all_timesteps)

        self.lengths = self.__boxes[:, :, 1] - self.__boxes[:, :, 0]

        if "type" in self.__columns:
            self.__types = np.sort(np.unique(self.__atomic_data[..., self.__columns["type"]]).astype(int))

        self.verbose_print(f"\nTrajectory file ({self.__trajectory}) scan complete.\n")

    def verbose_print(self, *args, verbosity = None):
        if verbosity is None:
            verbosity = constants.DEFAULT_VERBOSITY

        if verbosity <= self.__verbosity:
            print(*args)

    def extract_positions(self, target_array):
        if target_array.shape[-1] != self.__atomic_data.shape[-1]:
            raise RuntimeError(f"Cannot extract positions our of array of shape {target_array.shape} when initial data has shape {self.__atomic_data.shape}")

        return target_array[..., [self.__columns["x"], self.__columns["y"], self.__columns["z"]]]

    def wrap_positions(self):
        position_cols = [self.__columns["x"], self.__columns["y"], self.__columns["z"]]
        box_lo = self.__boxes[:, None, :, 0]
        lengths = self.lengths[:, None, :]
        self.__atomic_data[:, :, position_cols] = (self.__atomic_data[:, :, position_cols] - box_lo) % lengths

    def select_type(self, type, frame):
        if frame < 0 or frame >= self.__Nframes:
            raise RuntimeError(f"INTERNAL ERROR (BASE.select_type): Frame out of bounds. Number of frames: {self.__Nframes}. Requested frame: {frame}")

        return self.__atomic_data[frame][self.__atomic_data[frame][:, self.__columns["type"]] == type]

    def filter_type(self, type, frame):
        if frame < 0 or frame >= self.__Nframes:
            raise RuntimeError(f"INTERNAL ERROR (BASE.filter_type): Frame out of bounds. Number of frames: {self.__Nframes}. Requested frame: {frame}")

        return self.__atomic_data[frame][self.__atomic_data[frame][:, self.__columns["type"]] != type]

    def select_types(self, types, frame):
        if frame < 0 or frame >= self.__Nframes:
            raise RuntimeError(f"INTERNAL ERROR (BASE.select_types): Frame out of bounds. Number of frames: {self.__Nframes}. Requested frame: {frame}")

        type_column = self.__atomic_data[frame][:, self.__columns["type"]]

        return self.__atomic_data[frame][np.isin(type_column, types)]

    def filter_types(self, types, frame):
        if frame < 0 or frame >= self.__Nframes:
            raise RuntimeError(f"INTERNAL ERROR (BASE.filter_types): Frame out of bounds. Number of frames: {self.__Nframes}. Requested frame: {frame}")

        type_column = self.__atomic_data[frame][:, self.__columns["type"]]

        return self.__atomic_data[frame][~np.isin(type_column, types)]


    def get_nclosest(self, central, neighs, N, box):
        kdtree = sp.spatial.cKDTree(neighs, boxsize = box)
        norms, idx = kdtree.query(central, k = N)

        return norms, idx

    def write(self, data = None, header = None, outfile = None):
        if self.__class__.write == BASE.write:
            raise RuntimeError("Write function not defined by derived class. No output file will be generated. This is a technical issue with a particular analyzer being used.")

        if data is None:
            raise RuntimeError(f"No data provided to BASE.write by {self.__class__} analyzer. No output file will be generated. This is a technical issue with a particular analyzer being used.")
        if header is None:
            print(f"WARNING: the analyzer {self.__class__} did not provide the colum labels.")
            header = ""

        if outfile is None:
            print(f"WARNING: the analyzer {self.__class__} did not provide the output file name. The defualt name ({constants.DEFAULT_OUTFILE}) will be used.")
            outfile = constants.DEFAULT_OUTFILE

        np.savetxt(fname = outfile,
                   X = data,
                   delimiter = ",",
                   header = header,
                   comments = "",
                   )

    def analyze(self):
        print("WARNING: Handler does not perform any analyses.")
        sys.exit(0)


    def statistics(self, stats_dict = None):
        if self.__class__.statistics == BASE.statistics:
            print(f"WARNING: Statistics function not defined by derived class {self.__class__}. Nothing to display.")

        if not stats_dict is None:
            print("\nAnalyzer statistics:\n")

            for name, args in stats_dict.items():
                value, verbosity = args
                self.verbose_print(f"{name}: {value}", verbosity = verbosity)

    def check_required_columns(self, *args):
        to_check = iter(args)
        found = True
        stop = False
        while found and not stop:
            arg = next(to_check, None)
            if arg is None:
                stop = True
            elif not arg in self.__columns:
                found = False

        if not found:
            print(f"ERROR: Per-atom field \"{arg}\" is missing in provided trajectory file ({self.__trajectory}). Analyzer {self.__class__} requires {' '.join(args)} fields.")
            sys.exit(1)
