from cleanfid import fid
import os  
import argparse


def calc_kid(dir1, dir2): 
    score = fid.compute_kid(dir1, dir2)
    print('KID Score: ', score) 


if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir1', type=str, default=None, help='path to first directory of images')
    parser.add_argument('--dir2', type=str, default=None, help='path to second directory of images') 
    args = parser.parse_args()

    main(args.dir1, args.dir2)

