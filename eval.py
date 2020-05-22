import json
import argparse
import datasets
import models
from core import BaseEval
from utils.parse import pretty_table


def get_instance(module, base_name, args_name, config, *args):
    return getattr(module, config[base_name]['name'])(*args, **config[base_name][args_name])


def parse_arguments():
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('-c', '--config', default='config/VOC', type=str,
                        help='The config used to train the model')
    parser.add_argument('-w', '--weight', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    val_loader = get_instance(datasets, 'dataset', 'val_args', config)
    print(val_loader.dataset)

    model = get_instance(models, 'model', 'args', config)

    print(model)

    base_eval = BaseEval(model=model, resume_path=args.weight, config=config, val_loader=val_loader)

    results = base_eval.evaluate()

    print(pretty_table(results))
    # for k, v in results.items():
    #     if k != "Class_IoU":
    #         print(f"{str(k)} : {v}")
    #     else:
    #         print(pretty_table(v))


if __name__ == '__main__':
    main()

