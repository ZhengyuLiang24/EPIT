from common import main_test as main
from option import args

args.task = 'SSR'
args.max_angRes = 9
args.batch_size = 4
args.patch_size_for_test = 32
args.stride_for_test = 16
args.minibatch_for_test = 1

if __name__ == '__main__':
    args.device = 'cuda:0'
    args.data_list = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']

    args.scale_factor = 4
    args.model_name = 'EPIT'
    args.path_pre_pth = './pth/EPIT_5x5_4x_model.pth'
    for index in range(1, args.max_angRes + 1):
        args.angRes_in = index
        args.angRes_out = index
        main(args)

    args.scale_factor = 2
    args.model_name = 'EPIT'
    args.path_pre_pth = './pth/EPIT_5x5_2x_model.pth'
    for index in range(1, args.max_angRes + 1):
        args.angRes_in = index
        args.angRes_out = index
        main(args)
