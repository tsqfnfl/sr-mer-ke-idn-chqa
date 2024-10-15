from argparse import ArgumentParser

def print_args(args):
    """Prints the values of all command-line arguments."""
    print('=' * 80)
    print('Args'.center(80))
    print('-' * 80)
    for key in args.keys():
        if args[key]:
            print('{:>30}: {:<50}'.format(key, args[key]).center(80))
    print('=' * 80)
    
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/workspace", help="base path")
    parser.add_argument("--tasks", type=str, default="sr,mer,ke")
    parser.add_argument("--structure", type=str, default="parallel", help="parallel/hierarchical")
    parser.add_argument("--pretrained_model", type=str, default="indobenchmark/indobert-base-p2", help="name of the pre-trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--word_representation", type=str, default="first", help="choose between avg or first")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--me_soft_emb_size_ke", type=int, default=50, help="soft embedding size of medical entity for keyphrase extraction")
    parser.add_argument("--me_soft_emb_size_sr", type=int, default=50, help="soft embedding size of medical entity for sentence recognition")
    parser.add_argument("--ke_soft_emb_size", type=int, default=50, help="soft embedding size of keyphrase")
    parser.add_argument("--hidden_layer_dim", type=int, default=400, help="output size of hidden layer")
    parser.add_argument("--sr_lw", type=float, default=1.0, help="loss weight for sentence recognition")
    parser.add_argument("--mer_lw", type=float, default=1.0, help="loss weight for medical entity recognition")
    parser.add_argument("--ke_lw", type=float, default=1.0, help="loss weight for keyphrase extraction")

    args = vars(parser.parse_args())
    print_args(args)
    return args
