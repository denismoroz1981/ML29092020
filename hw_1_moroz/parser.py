import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Input for a filename with params for cubic equation')

    parser.add_argument('--path', type=str, required=True,
                        help='Filename with params for cubic equition')


    return parser.parse_args()


#if __name__ == "__main__":
 #   print('This program is being run by itself')
  #  args = parse_args()
   # print(args.b)

#else:
 #   print('I am being imported from another module')