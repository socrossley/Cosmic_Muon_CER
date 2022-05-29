from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--save", default=False, action='store_true', help="Save to file in \'./data\'")
args = vars(parser.parse_args())

print(args['save'])
