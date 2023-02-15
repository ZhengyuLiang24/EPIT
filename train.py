from common import main
from option import args

args.task = 'SSR'
args.max_angRes = 9
args.patch_dor_train = 32
args.patch_size_for_test = 32
args.stride_for_test = 16
args.minibatch_for_test = 1


if __name__ == '__main__':
    args.device = 'cuda:0'
    args.data_list = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']

    args.scale_factor = 4
    args.angRes_in = 5
    args.angRes_out = 5

    args.model_name = 'EPIT'
    args.start_epoch = 0
    main(args)
