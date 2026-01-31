import sys
import numpy as np
from scipy.spatial import cKDTree
import argparse

from .handlers import ANGLE, QUNIT, DENSITY, RDFS, RINGS, VDOS

from . import constants
from . import utils



def parse_args():
    KNOWN_COMMANDS = {"angle", "qunit", "density", "rdf", "rings", "vdos"}
    DUMMY_FILE = "__MISSING_FILE__"

    if len(sys.argv) > 1 and sys.argv[1] in KNOWN_COMMANDS:
        sys.argv.insert(1, DUMMY_FILE)

    parser = utils.ErrorHandlingParser(prog = "trajan", formatter_class = utils.NoMetavarHelpFormatter)

    parser.add_argument("file", help = "LAMMPS trajectory file to be analized.", type = str)

    parser.add_argument("-v", "--verbose", help = f"Screen log verbosity (1 - {constants.MAX_VERBOSITY}). Default: {constants.DEFAULT_VERBOSITY}", type = utils.verbosity_type(), default = constants.DEFAULT_VERBOSITY)

    parser.add_argument("-o", "--outfile", help = f"Name of the ouput data file", type = str, default = constants.DEFAULT_OUTFILE)

    parser.add_argument("-s", "--steps", help = "Frame selection pattern (e.g., '10:' or '0:1000:10'). Default: ::1", type = str, default = "::1")

    parser.add_argument("-b", "--buffer", help = f"Maximum allowed per-core RAM in Mb. No buffer given corresponds to line by line file reading. Default: {constants.DEFAULT_BUFFER_MB}", type = int, default = constants.DEFAULT_BUFFER_MB)

    subparsers = parser.add_subparsers(dest="command", required = True, metavar = "analyzer", action = utils.StrictSubParsersAction)

    bond_angle = subparsers.add_parser("angle", help = "Argument parser for extracting bond angle distributions from LAMMPS-generated trajectory files.", epilog = "Verbosity Controls:\n   1 : File scan and analysis messages\n       Mean bond angle and standard deviation\n   2 : Frame scan and analysis messages\n   3 : Peak position and fraction of species analysis", formatter_class = utils.NoMetavarHelpFormatter)

    bond_angle.add_argument("types", nargs = 3, type = int, default = [1, 1, 1], help = "Integer values representing atomic types of a bonded triplet with the atom of second type in the middle.")

    bond_angle.add_argument("-c", "--cutoffs", nargs ="+", type = float, default = [], help = "Maximum distances allowed for nearest species of a triplet to be considered bonded.")

    bond_angle.add_argument("-b", "--bincount", type = int, default = constants.DEFAULT_HIST_BINCOUNT, help = f"Number of bins for the bond angle histogram. Default: {constants.DEFAULT_HIST_BINCOUNT}.")

    bond_angle.set_defaults(handler_class = ANGLE)


    qunit = subparsers.add_parser("qunit", help = "Argument parser for calculating Q-unit distributions from LAMMPS-generated trajectory files.", epilog = "Verbosity Controls:\n   1 : File scan and analysis messages\n       Average Q unit\n   2 : Frame scan and analysis messages", formatter_class = utils.NoMetavarHelpFormatter)
    qunit.add_argument("types", nargs = "+", type = int, default = [], help = "Integer values representing atomic types of network formers and connectors separated by a zero. Any connector bonding two network formers will contribute to the Qunit count of both of the network formers. Exmaple of input: 1 2 3 0 4 5; here atoms of types 1, 2, and 3 are considered network formers and atoms of types 4 and 5 are considered network connectors.")
    qunit.add_argument("-c", "--cutoffs", nargs = "+", type = float, default = [], help = "Maximum distances between each former and connector (in Angstroms) for them to be considered bonded. Default: inf. Example: 1.1 1.2 1.3 1.4. If types supplied are (1 2 0 3 4) then C(1, 3) = 1.1, C(1, 4) = 1.2, C(2, 3) = 1.3, and C(2, 4) = 1.4.")

    qunit.set_defaults(handler_class = QUNIT)


    density = subparsers.add_parser("density", help = "Argument parser for density calculation from LAMMPS-generated trajectory files.", epilog = "Verbosity Controls:\n   1 : File scan and analysis messages\n       Mean density and standard deviation\n   2 : Frame scan and analysis messages", formatter_class = utils.NoMetavarHelpFormatter)
    density.add_argument("elements", nargs = "+", type = str, default = [], help = "Space separated list of element names for each type present in the trajectory file. In case a custom mass has been used a numerical value can be specified. Example: Si 15.799 Na 0.2")
    density.add_argument("-u", "--units", type = str, default = constants.DEFAULT_UNITS, help = f"LAMMPS unit set for conversion. Default: {constants.DEFAULT_UNITS}.")
    density.set_defaults(handler_class = DENSITY)


    pair_distribution = subparsers.add_parser("rdf", help = "Argument parser for raidal pair distribution function calculations from LAMMPS-generated trajectory files.", epilog = "Verbosity Controls:\n   1 : File scan and analysis messages\n   2 : Frame scan and analysis messages\n       Maximum peak position\n   3 : Per-pair analysis messages\n       Peak positions", formatter_class = utils.NoMetavarHelpFormatter)
    pair_distribution.add_argument("-c", "--cutoff", type = float, help = f"Maximum atomic separation distance considered for the generation of the radial distribution functions. Default: {constants.DEFAULT_RDF_CUTOFF}", default = constants.DEFAULT_RDF_CUTOFF)
    pair_distribution.add_argument("-p", "--pair", nargs = 2, type = int, action = "append", help = "Atomic type pairs for selecting specific partial distribution functions.\n Repeat for multiple pairs: -p 1 2 -p 3 4")
    pair_distribution.add_argument("-b", "--bincount", type = int, default = constants.DEFAULT_HIST_BINCOUNT, help = f"Number of bins for the radial distribution functions. Default: {constants.DEFAULT_HIST_BINCOUNT}.")
    pair_distribution.add_argument("-bs", "--batch-size", type = int, default = constants.DEFAULT_ATOM_BATCH, help = f"Number of atoms whose neighbors are analyzed at a time. Default: {constants.DEFAULT_ATOM_BATCH}.\n Helpful to avoid memory overload when analyzing large systems.")
    pair_distribution.add_argument("-t", "--total", nargs = "+", default = [], help = "Calculate the total correlation function based on the atom type mappings or custom neutron scattering lengths.\n This flag should be followd by a space separated list of atomic names or scattering lengths for each type in ascending order. Note: This value is calculated per atom while some experimental papers normalize the result per formula unit of the material. This will require scaling the output.")
    pair_distribution.add_argument("-br", "--broaden", type = float, help = "When this flag is the idealized total correlation function is broadened to match experimental results. The value of the maximum momentum transfer Q_max should be provided in inverse angstroms for broadening.")
    pair_distribution.set_defaults(handler_class = RDFS)

    ring_size = subparsers.add_parser("rings", help = "Argument parser for ring size distribution calculation from LAMMPS-generated trajectory files.", epilog = "Verbosity Controls:\n   1 : File scan and analysis messages\n       Average ring size\n   2 : Frame scan and analysis messages\n   3 : In-frame ring analysis percentage", formatter_class = utils.NoMetavarHelpFormatter)
    ring_size.add_argument("types", nargs = "+", type = int, default = [], help = "Integer values representing atomic types of base and connector atoms separated by a zero. Exmaple of input: 1 2 3 0 4 5; here atoms of types 1, 2, and 3 are considered base and atoms of types 4 and 5 are considered connectors.")
    ring_size.add_argument("-c", "--cutoffs", nargs = "+", type = float, default = [], help = "Maximum distances between each base and connector atoms (in Angstroms) for them to be considered bonded. Default: inf. Example: 1.1 1.2 1.3 1.4. If types supplied are (1 2 0 3 4) then C(1, 3) = 1.1, C(1, 4) = 1.2, C(2, 3) = 1.3, and C(2, 4) = 1.4.")
    ring_size.add_argument("-cb", "--connector-bonds", type = int, nargs = "+", default = [], help = f"Space separated list of integers that specifies the number of nearest neighbors that are considered bonded for each connector. Default: {constants.DEFAULT_CONNECTOR_BONDS}")
    ring_size.add_argument("-m", "--max-size", type = int, default = constants.DEFAULT_MAX_RING_SIZE, help = f"Maximum detectable ring size. Default: {constants.DEFAULT_MAX_RING_SIZE}")
    ring_size.add_argument("-a", "--algorithm", type = str, default = constants.DEFAULT_RING_ALGORITHM, help = f"Switch between p (primitive) and s (smallest) ring detecting algorithm. Default: {constants.DEFAULT_RING_ALGORITHM}.")
    ring_size.add_argument("-p", "--paral-mode", type = str, default = constants.DEFAULT_PARAL_MODE, help = f"MPI Parallelization strategy. 'atom': Rank 0 reads/builds graph, all Ranks search rings. 'frame': Ranks process different frames independently. Default: {constants.DEFAULT_PARAL_MODE}. Note: Ignored for single processor runs.")
    ring_size.set_defaults(handler_class = RINGS)


    vdos = subparsers.add_parser("vdos", help = "Argument parser for the vibrational density of states (vDOS) calculation from LAMMPS-generated trajectory files.", formatter_class = utils.NoMetavarHelpFormatter)
    vdos.add_argument("timestep", type = float, help = f"Difference in simulation time between recorded timesteps. Comment: This is rarely simulation timestep. Default: {constants.DEFAULT_TIMESTEP_NUM}", default = constants.DEFAULT_TIMESTEP_NUM)
    vdos.add_argument("-m", "--max-timelag", type = float, help = "Maximum velocity autocorrelation function time difference in simulation units. Default: MaxTime", default = np.inf)
    vdos.add_argument("-u", "--units", type = str, help = f"LAMMPS unit set for conversion. Default: {constants.DEFAULT_UNITS}.", default = constants.DEFAULT_UNITS)
    vdos.add_argument("-l", "--lag-step", type = int, help = f"Velocity autocorrelation function resolution in simulation time units. Default: timestep", default = None)
    vdos.add_argument("-t", "--taper", type = int, help = f"Fraction of velocity autocorrelation function to be tapered down to 0 for a clean FFT. Default: {constants.DEFAULT_VDOS_TAPER}", default = constants.DEFAULT_VDOS_TAPER)
    vdos.add_argument("-p", "--padding", type = int, help = f"End zero padding size as a fraction of velocity autocorrelation length. Default: {constants.DEFAULT_VDOS_TAPER}", default = constants.DEFAULT_VDOS_TAPER)

    vdos.set_defaults(handler_class = VDOS)

    args = parser.parse_args()

    if args.file == DUMMY_FILE:
        parser.error("the following arguments are required: file")

    return args


def main():
    args = parse_args()

    handler = args.handler_class(args)

    handler.analyze()

    handler.statistics()

    handler.write()


if __name__ == "__main__":
    main()
