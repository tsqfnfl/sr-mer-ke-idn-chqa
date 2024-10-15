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
    parser.add_argument("--task", type=str, default="sentence_recognition", help="choose between sentence_recognition, medical_entity_recognition, or keyphrase_extraction")
    parser.add_argument("--pretrained_model", type=str, default="indobenchmark/indobert-base-p2", help="name of the pre-trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--word_representation", type=str, default="first", help="choose between avg or first")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = vars(parser.parse_args())
    print_args(args)
    return args
