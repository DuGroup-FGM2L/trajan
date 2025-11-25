import sys
import numpy as np
from scipy.spatial import cKDTree
import argparse

from .handlers import ANGLE

from . import constants
from . import utils



def parse_args():
    parser = utils.ErrorHandlingParser(prog = "trajan", formatter_class = utils.NoMetavarHelpFormatter)

    parser.add_argument("file", help = "LAMMPS trajectory file to be analized.", type = str)

    parser.add_argument("-v", "--verbose", help = f"Screen log verbosity (1 - {constants.MAX_VERBOSITY}). Default: {constants.DEFAULT_VERBOSITY}", type = utils.verbosity_type(), default = constants.DEFAULT_VERBOSITY)

    parser.add_argument("-o", "--outfile", help = f"Name of the ouput data file", type = str, default = constants.DEFAULT_OUTFILE)

    #parser.add_argument("-s", "--steps", help = "Pattern for specifying which steps to use", type = str)

    subparsers = parser.add_subparsers(dest="command", required=True, metavar = "analyzer", action = utils.StrictSubParsersAction)

    bond_angle = subparsers.add_parser("angle", help = "Argument parser for extracting bond angle distributions from LAMMPS-generated trajectory files.", formatter_class = utils.NoMetavarHelpFormatter)

    bond_angle.add_argument("types", nargs = 3, type = int, default = [1, 1, 1], help = "Integer values representing atomic types of a bonded triplet with the atom of second type in the middle.")

    #bond_angle.add_argument("-m", "--maxcut", nargs ="+", type = float, default = [], help = "Maximum distance allowed for nearest species of a triplet to be considered bonded.")

    bond_angle.add_argument("-b", "--bincount", type = int, default = constants.DEFAULT_HIST_BINCOUNT, help = f"Number of bins for the bond angle histogram. Default: {constants.DEFAULT_HIST_BINCOUNT}.")

    bond_angle.set_defaults(handler_class = ANGLE)

    return parser.parse_args()


def main():
    args = parse_args()

    handler = args.handler_class(args)

    handler.analyze()

    handler.write()


if __name__ == "__main__":
    main()
