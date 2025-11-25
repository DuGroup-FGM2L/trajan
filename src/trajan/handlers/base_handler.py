import numpy as np
import sys
from scipy import spatial as scisp

from trajan import constants

class BASE():
    def __init__(self, filename, verbosity):
        self.trajectory = filename
        self.verbosity = verbosity

        #Per-snapshot arrays
        self.boxes = None
        self.box_periods = None
        self.atom_counts = None
        self.timesteps = None

        #Per-snapshot 2D arrays
        self.atomic_data = None

        #General data
        self.columns = dict()
        self.Nframes = 0

    def parse_file(self):
        self.verbose_print(f"Scanning trajectory file: {self.trajectory}")
        read_natoms = False
        read_timestep = False
        all_data = list()
        all_boxes = list()
        all_timesteps = list()
        atom_counts = list()
        atomic_counts = list()
        atomic_counts = list()
        box_period = list()
        column_headers = list()
        box_ctr = float("inf")
        atom_ctr = float("inf")
        natoms = 0

        #Max timestep set for testing
        max_ts =  float("inf")
        with open(self.trajectory, "r") as f:
            for line in f:
                if self.Nframes > max_ts:
                    continue
                if "ITEM: TIMESTEP" in line:
                    read_timestep = True
                elif read_timestep:
                    timestep = int(line.strip())
                    self.verbose_print(f"{self.Nframes} scan of TS {timestep}", verbosity = 2)
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
                    atom_data = list()
                    column_headers = line.split()[2:]
                elif atom_ctr < natoms:
                    atom_data.append(np.fromstring(line, sep = " "))
                    atom_ctr += 1
                elif atom_ctr == natoms:
                    all_data.append(atom_data)
                    all_boxes.append(box)
                    atom_ctr = float("inf")
                    self.Nframes += 1

        for i, column_heading in enumerate(column_headers):
            self.columns[column_heading] = i

        self.boxes = np.array(all_boxes)
        self.atomic_data = np.array(all_data)
        self.atom_counts = np.array(atomic_counts)
        self.box_periods = np.array(box_period)
        self.timesteps = np.array(all_timesteps)

        self.lengths = self.boxes[:, :, 1] - self.boxes[:, :, 0]

        self.verbose_print(f"Trajectory file ({self.trajectory}) scan complete.")

    def verbose_print(self, *args, verbosity = None):
        if verbosity is None:
            verbosity = constants.DEFAULT_VERBOSITY

        if verbosity <= self.verbosity:
            print(*args)

    def extract_positions(self, target_array):
        if target_array.shape[-1] != self.atomic_data.shape[-1]:
            raise RuntimeError(f"Cannot extract positions our of array of shape {target_array.shape} when initial data has shape {self.atomic_data.shape}")

        return target_array[..., [self.columns["x"], self.columns["y"], self.columns["z"]]]

    def wrap_positions(self):
        position_cols = [self.columns["x"], self.columns["y"], self.columns["z"]]
        box_lo = self.boxes[:, None, :, 0]
        lengths = self.lengths[:, None, :]
        self.atomic_data[:, :, position_cols] = (self.atomic_data[:, :, position_cols] - box_lo) % lengths

    def select_type(self, type, frame):
        if frame < 0 or frame >= self.Nframes:
            raise RuntimeError(f"INTERNAL ERROR (BASE.select_type): Frame out of bounds. Number of frames: {self.Nframes}. Requested frame: {frame}")

        return self.atomic_data[frame][self.atomic_data[frame][:, self.columns["type"]] == type]

    def get_nclosest(self, central, neighs, N, box):
        kdtree = scisp.cKDTree(neighs, boxsize = box)
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
        print("WARNING: Handler does not perform any analysis.")
        sys.exit(0)
