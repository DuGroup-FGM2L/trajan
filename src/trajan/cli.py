import sys
import numpy as np
from scipy.spatial import cKDTree
import argparse

from .handlers import ANGLE, QUNIT, DENSITY, RDFS

from . import constants
from . import utils



def parse_args():
    parser = utils.ErrorHandlingParser(prog = "trajan", formatter_class = utils.NoMetavarHelpFormatter)

    parser.add_argument("file", help = "LAMMPS trajectory file to be analized.", type = str)

    parser.add_argument("-v", "--verbose", help = f"Screen log verbosity (1 - {constants.MAX_VERBOSITY}). Default: {constants.DEFAULT_VERBOSITY}", type = utils.verbosity_type(), default = constants.DEFAULT_VERBOSITY)

    parser.add_argument("-o", "--outfile", help = f"Name of the ouput data file", type = str, default = constants.DEFAULT_OUTFILE)

    #parser.add_argument("-s", "--steps", help = "Pattern for specifying which steps to use", type = str)

    subparsers = parser.add_subparsers(dest="command", required = True, metavar = "analyzer", action = utils.StrictSubParsersAction)

    bond_angle = subparsers.add_parser("angle", help = "Argument parser for extracting bond angle distributions from LAMMPS-generated trajectory files.", formatter_class = utils.NoMetavarHelpFormatter)

    bond_angle.add_argument("types", nargs = 3, type = int, default = [1, 1, 1], help = "Integer values representing atomic types of a bonded triplet with the atom of second type in the middle.")

    bond_angle.add_argument("-c", "--cutoffs", nargs ="+", type = float, default = [], help = "Maximum distances allowed for nearest species of a triplet to be considered bonded.")

    bond_angle.add_argument("-b", "--bincount", type = int, default = constants.DEFAULT_HIST_BINCOUNT, help = f"Number of bins for the bond angle histogram. Default: {constants.DEFAULT_HIST_BINCOUNT}.")

    bond_angle.set_defaults(handler_class = ANGLE)


    qunit = subparsers.add_parser("qunit", help = "Argument parser for calculating Q-unit distributions from LAMMPS-generated trajectory files.", formatter_class = utils.NoMetavarHelpFormatter)
    qunit.add_argument("types", nargs = "+", type = int, default = [], help = "Integer values representing atomic types of network formers and connectors separated by a zero. Any connector bonding two network formers will contribute to the Qunit count of both of the network formers. Exmaple of input: 1 2 3 0 4 5; here atoms of types 1, 2, and 3 anre considered network formers and atoms of types 4 and 5 are considered network connectors.")
    qunit.add_argument("-c", "--cutoffs", nargs = "+", type = float, default = [], help = "Maximum distances between each former and connector (in Angstroms) for them to be considered bonded. Default: inf. Example: 1.1 1.2 1.3 1.4. If types supplied are (1 2 0 3 4) then C(1, 3) = 1.1, C(1, 4) = 1.2, C(2, 3) = 1.3, and C(2, 4) = 1.4.")

    qunit.set_defaults(handler_class = QUNIT)


    density = subparsers.add_parser("density", help = "Argument parser for density calculation from LAMMPS-generated trajectory files.", formatter_class = utils.NoMetavarHelpFormatter)
    density.add_argument("elements", nargs = "+", type = str, default = [], help = "Space separated list of element names for each type present in the trajectory file. In case a custom mass has been used a numerical value can be specified. Example: Si 15.799 Na 0.2")
    density.add_argument("-u", "--units", type = str, default = constants.DEFAULT_UNITS, help = f"LAMMPS unit set for conversion. Default: {constants.DEFAULT_UNITS}.")
    density.set_defaults(handler_class = DENSITY)


    pair_distribution = subparsers.add_parser("rdf", help = "Argument parser for raidal pair distribution function calculations from LAMMPS-generated trajectory files.", formatter_class = utils.NoMetavarHelpFormatter)
    pair_distribution.add_argument("-c", "--cutoff", type = float, help = f"Maximum atomic separation distance considered for the generation of the radial distribution functions. Default: {constants.DEFAULT_RDF_CUTOFF}", default = constants.DEFAULT_RDF_CUTOFF)
    pair_distribution.add_argument("-p", "--pair", nargs = 2, type = int, action = "append", help = "Atomic type pairs for selecting specific partial distribution functions.\n Repeat for multiple pairs: -p 1 2 -p 3 4")
    pair_distribution.add_argument("-b", "--bincount", type = int, default = constants.DEFAULT_HIST_BINCOUNT, help = f"Number of bins for the radial distribution functions. Default: {constants.DEFAULT_HIST_BINCOUNT}.")
    pair_distribution.add_argument("-bs", "--batch-size", type = int, default = constants.DEFAULT_ATOM_BATCH, help = f"Number of atoms whose neighbors are analyzed at a time. Default: {constants.DEFAULT_ATOM_BATCH}.\n Helpful to avoid memory overload when analyzing large systems.")
    pair_distribution.set_defaults(handler_class = RDFS)


    return parser.parse_args()


def main():
    args = parse_args()

    handler = args.handler_class(args)

    handler.analyze()

    handler.statistics()

    handler.write()


if __name__ == "__main__":
    main()
