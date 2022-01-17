import shutil
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--start', type=int, default=0, required=False)
parser.add_argument('--end', type=int, required=True)
parser.add_argument('--stops', type=int, default=3)
args = parser.parse_args()


class PathManager:
    def __init__(self, path):
        self.dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        self.filename, self.ext = os.path.splitext(basename)
        print(self.dirname, basename, self.filename, self.ext)

    def get(self, num):
        return os.path.join(self.dirname, f'{self.filename}{str(num).zfill(3)}{self.ext}')

paths = PathManager(args.path)
num = args.end + 1

for _ in range(args.stops):
    shutil.copy2(paths.get(args.end), paths.get(num))
    num += 1

for i in range(args.end, args.start + 1, -1):
    shutil.copy2(paths.get(i), paths.get(num))
    num += 1

for _ in range(args.stops):
    shutil.copy2(paths.get(args.start), paths.get(num))
    num += 1