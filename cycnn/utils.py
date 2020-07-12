import os

def generate_fname(dataset, model, polar_transform, augmentation):
    fname = dataset + '-' + model
    if polar_transform is not None:
        fname += '-' + polar_transform
    if augmentation is not None:
        fname += '-' + augmentation
    return fname

def validate(dataset, model, polar_transform, augmentation):
    fname = 'logs/' + generate_fname(dataset, model, polar_transform, augmentation) + '.txt'

    if not os.path.isfile(fname):
        return False

    f = open(fname, 'r')
    content = f.read()

    return 'Training Done!' in content


def soft_validate(dataset, model, polar_transform, augmentation):
    fname = 'saves/' + generate_fname(dataset, model, polar_transform, augmentation) + '.pt'

    return os.path.isfile(fname)
