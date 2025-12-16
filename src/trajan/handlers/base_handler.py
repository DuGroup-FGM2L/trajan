import numpy as np
import scipy as sp
import sys

from trajan import constants

class BASE():
    def __init__(self, filename, verbosity):
        self.__trajectory = filename
        self.__verbosity = verbosity

        self.__box = None
        self.__lengths = None
        self.__box_periods = None
        self.__natoms = None
        self.__timestep = None
        self.__types = None

        self.__atomic_data = None

        #General data
        self.__columns = dict()
        self.__frame = 0

        self.__wrap_positions = False

        self.__timesteps = list()

    def get_trajectory(self):
        return self.__trajectory

    def get_verbosity(self):
        return self.__verbosity

    def get_box(self):
        return self.__box

    def get_lengths(self):
        return self.__lengths

    def get_box_periods(self):
        return self.__box_periods

    def get_natoms(self):
        return self.__natoms

    def get_timestep(self):
        return self.__timestep

    def get_timesteps(self):
        return self.__timesteps

    def get_types(self):
        return self.__types

    def get_atomic_data(self):
        return self.__atomic_data

    def get_columns(self):
        return self.__columns

    def get_frame(self):
        return self.__frame

    def parse_file(self):
        self.verbose_print(f"Peeking at trajectory file: {self.__trajectory}", verbosity = 2)

        iterator = self.trajectory_reader(run_once = True)
        try:
            next(iterator)
        except StopIteration:
            raise RuntimeError("File seems empty or invalid.")

        self.verbose_print(f"Metadata scan complete. Columns found: {list(self.__columns.keys())}", verbosity = 2)


    def trajectory_reader(self, run_once = False):
        if not run_once:
            self.verbose_print(f"Scanning trajectory file: {self.__trajectory}")

        read_timestep = False
        read_natoms = False
        read_box = False
        read_atoms = False

        box_lines = list()
        atom_lines = list()

        self.__frame = 0
        self.__timesteps = list()

        with open(self.__trajectory, "r") as f:
            for line in f:
                if "ITEM: TIMESTEP" in line:
                    if read_atoms and len(atom_lines) > 0:
                        self._postprocess(atom_lines)
                        yield self.__frame

                        self.__frame += 1
                        if run_once:
                            return

                    read_timestep = True
                    read_atoms = False
                    atom_lines = list()
                    box_lines = list()

                elif read_timestep:
                    self.__timestep = int(line.strip())
                    self.verbose_print(f"{self.__frame} scan of TS {self.__timestep}", verbosity = 2)
                    read_timestep = False

                elif "ITEM: NUMBER OF ATOMS" in line:
                    read_natoms = True
                elif read_natoms:
                    natoms = int(line.strip())
                    self.__natoms = natoms
                    read_natoms = False

                elif "ITEM: BOX BOUNDS" in line:
                    read_box = True
                    self.__box_periods = line.split()[-3:]
                elif read_box:
                    box_lines.append(line)
                    if len(box_lines) == 3:
                        self.__box = np.array([np.fromstring(l, sep=" ", count=2) for l in box_lines])
                        read_box = False

                elif "ITEM: ATOMS" in line:
                    read_atoms = True
                    if not self.__columns:
                        for i, h in enumerate(line.split()[2:]):
                            self.__columns[h] = i
                elif read_atoms:
                    atom_lines.append(line)

        if len(atom_lines) > 0:
            self._postprocess(atom_lines)
            yield self.__frame

            self.__frame += 1



        self.verbose_print(f"\nTrajectory file ({self.__trajectory}) scan complete.\n")

    def _postprocess(self, atom_lines):
        self.__atomic_data = np.loadtxt(atom_lines)

        self.__lengths = self.__box[:, 1] - self.__box[:, 0]

        if self.__wrap_positions:
            self._position_wrapper()


        if "type" in self.__columns.keys():
            self.__types = np.sort(np.unique(self.__atomic_data[:, self.__columns["type"]]).astype(int))

        self.__timesteps.append(self.__timestep)


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
        self.__wrap_positions = True

    def _position_wrapper(self):
        position_cols = [self.__columns["x"], self.__columns["y"], self.__columns["z"]]
        box_lo = self.__box[:, 0]
        self.__atomic_data[:, position_cols] = (self.__atomic_data[:, position_cols] - box_lo) % self.__lengths

    def select_type(self, type):
        return self.__atomic_data[self.__atomic_data[:, self.__columns["type"]] == type]

    def filter_type(self, type):
        return self.__atomic_data[self.__atomic_data[:, self.__columns["type"]] != type]

    def select_types(self, types):
        type_column = self.__atomic_data[:, self.__columns["type"]]

        return self.__atomic_data[np.isin(type_column, types)]

    def filter_types(self, types):
        type_column = self.__atomic_data[:, self.__columns["type"]]

        return self.__atomic_data[~np.isin(type_column, types)]


    def get_nclosest(self, central, neighs, N):
        kdtree = sp.spatial.cKDTree(neighs, boxsize = self.__lengths)
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

        print(f"Analyzer output has been saved in the \"{outfile}\" file.")

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
