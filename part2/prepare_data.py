"""Copy the original assignment data/records from the course starter ZIP."""
import argparse
from pathlib import Path, PurePosixPath
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('starter_zip', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
with zipfile.ZipFile(args.starter_zip) as z:
    for item in z.infolist():
        if item.is_dir() or '__MACOSX' in item.filename:
            continue
        parts = PurePosixPath(item.filename).parts
        if 'part2' not in parts:
            continue
        relative = parts[parts.index('part2') + 1:]
        if not relative or relative[0] not in {'data', 'records'}:
            continue
        if '..' in relative:
            raise ValueError('Unsafe archive path')
        destination = root.joinpath(*relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(z.read(item))
for split, count in [('train', 4225), ('dev', 466), ('test', 432)]:
    assert len((root / 'data' / (split + '.nl')).read_text().splitlines()) == count
assert (root / 'data/flight_database.db').is_file()
assert (root / 'records/ground_truth_dev.pkl').is_file()
print('Original course data ready:', root / 'data')
