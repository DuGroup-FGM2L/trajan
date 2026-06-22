import argparse
import sys

from . import constants

_ = lambda s: s


class StrictSubParsersAction(argparse._SubParsersAction):

    def __call__(self, parser, namespace, values, option_string=None):
        parser_name = values[0]
        arg_strings = values[1:]

        # set the parser name if requested
        if self.dest is not argparse.SUPPRESS:
            setattr(namespace, self.dest, parser_name)

        # select the parser
        try:
            subparser = self._name_parser_map[parser_name]
        except KeyError:
            args = {'parser_name': parser_name,
                    'choices': ', '.join(self._name_parser_map)}
            msg = _('unknown parser %(parser_name)r (choices: %(choices)s)') % args
            raise ArgumentError(self, msg)


        #Parse argse with error calls from subparser instead of delegating
        #unrecognized argument error to the main parser
        subnamespace = subparser.parse_args(arg_strings)
        for key, value in vars(subnamespace).items():
            setattr(namespace, key, value)



class NoMetavarHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def _get_default_metavar_for_optional(self, action):
        return ""

    def _format_args(self, action, default_metavar):
        get_metavar = self._metavar_formatter(action, default_metavar)
        return "%s" % get_metavar(1)


class ErrorHandlingParser(argparse.ArgumentParser):
    DUMMY_FILE = "__MISSING_FILE__"
    def __init__(self, known_subparsers = None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.known_subparsers = known_subparsers or set()

    def error(self, message):
        sys.stderr.write("\n\nERROR: %s\n\n" % message)
        self.print_help()
        sys.exit(2)

    def parse_args(self, args = None, namespace = None):
        if args is None:
            args = sys.argv[1:]
        if not self.known_subparsers or not args:
            return super().parse_args(args, namespace)

        cmd_idx = 0
        num_args_total = len(args)
        found = False
        while cmd_idx < num_args_total and not found:
            if args[cmd_idx] in self.known_subparsers or args[cmd_idx][0] == "-":
                found = True
            else:
                cmd_idx += 1

        if cmd_idx == 0:
            args.insert(0, self.DUMMY_FILE)
            cmd_idx = 1
        elif cmd_idx == -1:
            cmd_idx = len(args)

        for action in self._actions:
            if action.dest == "files":
                action.nargs = cmd_idx
                break

        parsed = super().parse_args(args, namespace)

        if len(parsed.files) == 1 and parsed.files[0] == self.DUMMY_FILE:
            self.error("the following arguments are required: file")

        return parsed


def verbosity_type():
    def checker(value):
        ivalue = int(value)
        if ivalue < 1 or ivalue > constants.MAX_VERBOSITY:
            raise argparse.ArgumentTypeError(f"Verbosity must be between 1 and {constants.MAX_VERBOSITY}")
        return ivalue
    return checker

def parse_frame_pattern(pattern_str):
    """
    Parses a string in the format start:stop:step into integer values.

    Examples:
      "100"       -> start=100, stop=101, step=1 (Single frame)
      "100:"      -> start=100, stop=Inf, step=1
      ":200"      -> start=0,   stop=200, step=1
      "::10"    -> start=0,   stop=Inf, step=10
      "100:200:5" -> start=100, stop=200, step=5
    """

    # Defaults
    start = 0
    stop = float('inf')
    step = 1

    if not pattern_str:
        return start, stop, step

    parts = pattern_str.split(':')


    def parse_token(token, default):
        token = token.strip()
        if token == "":
            return default
        return int(token)

    if len(parts) == 1:
        val = parse_token(parts[0], start)
        return val, val + 1, 1

    if len(parts) >= 2:
        start = parse_token(parts[0], start)
        stop = parse_token(parts[1], stop)

    if len(parts) == 3:
        step = parse_token(parts[2], step)

    return start, stop, step
