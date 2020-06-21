from datasets import VOCDataset
from datasets import CocoStuff10K
from datasets import CocoStandard
from datasets import CityScapesDataset
from datasets import LaneSequenceDataset
from datasets import LaneDataset
from datasets import CULaneDataset
from datasets import TuSimpleDataset
from utils.utils_datasets import show_random_image
import json


if __name__ == '__main__':
    MEAN = [0.45734706, 0.43338275, 0.40058118]
    STD = [0.23965294, 0.23532275, 0.2398498]

    config = json.load(open('./config/lanenet.json'))

    args = config['dataset']['train_args']
    kwargs = {
        'data_dir': args['data_dir'],
        'crop_size': args['crop_size'],
        'base_size': args['base_size'],
        'augment': args['augment'],
        'flip': args['flip'],
        'rotate': args['rotate'],
        'scale': args['scale'],
        'crop': args['crop'],
        'blur': args['blur'],
        'split': 'train',
        'mean': MEAN,
        'std': STD,
    }

    # dataset = CityScapesDataset(mode='fine', **kwargs)
    dataset = LaneDataset(**kwargs)
    # dataset = CityScapesDataset(**kwargs)
    # dataset = VOCDataset(**kwargs)
    # dataset = TuSimpleDataset(**kwargs)
    # dataset = CocoStandard(number_classes=37, **kwargs)
    print(dataset)

    show_random_image(dataset, 6)
