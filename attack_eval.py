import argparse
from attacksLID import generate_attack, attack_parser
from verify import eval

if __name__ == '__main__':
    parser = attack_parser()
    # eval
    parser.add_argument('--output_file', default='log.csv')
    args = parser.parse_args()
    generate_attack(args)
    eval(args.output_dir,args.output_file)