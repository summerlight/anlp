import argparse
import csv
import itertools
import os.path
import generate_text

def read_csv_file(data_type, path):
    summary_path = os.path.join(path, '{}-summary'.format(data_type))
    with open(summary_path, 'rt') as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def read_text(csv_row, path):
    file_path = os.path.join(path, csv_row['docid'])
    with open(file_path, 'rt') as f:
        text = f.read()
        len1 = int(csv_row['pri_len'])
        len2 = int(csv_row['sec_len'])
        return {
            'text': text,
            'metadata': [
                {'lang': csv_row['pri_lang'], 'begin': 0, 'end': len1},
                {'lang': csv_row['sec_lang'], 'begin': len1, 'end': len1 + len2}
            ]
        }


def read_dataset(data_type, path):
    for row in read_csv_file(data_type, args.path):
        yield read_text(row, os.path.join(args.path, data_type))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adapt ALTW2010 LangID task dataset.')
    parser.add_argument('--path', dest='path',
                        default='../../data/altw2010-langid/',
                        help='a directory contains corpora to be used')
    args = parser.parse_args()
    for doc in read_dataset('dev', args.path):
        print(doc)

