import os
import sys
import time


def for_files(path, file_func, verbose=True):

    files = os.listdir(path)

    for file, i in zip(files, range(len(files))):

        if verbose:
            print('\rProcessing: ' + file + ' (' + str(i + 1) + '/' + str(len(files)) + ')', end='')

        file_func(path + '/' + file)
    if verbose:
        print()

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    format_str = "{0:." + str(decimals) + "f}"
    percents = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '=' * (filled_length - 1) + '>' + ' ' * (bar_length - filled_length)
    sys.stdout.write('\r%s [%s] %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()