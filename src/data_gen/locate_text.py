import sqlite3
import argparse
import json
import index_db
from collections import defaultdict


def query_doc(cursor, lang, title):
    cursor.execute(index_db.select_lang_title, (lang, title))
    result = cursor.fetchone()
    if not result:
        return None
    return {
        'lang': result[0],
        'doc_id': result[1],
        'doc_path': result[2],
        # 'url': result[3], # I don't think url is needed here...
        'title': result[4],
        'begin': result[5],
        'end': result[6]
    }


def locate_single_topic_texts(lang_title_dict, cursor):
    same_topic = (query_doc(cursor, l, t) for l, t in lang_title_dict.items())
    return sorted(
        (i for i in same_topic if i),
        key=lambda x: x['lang']
    )


def locate_interlanguage_texts(file_path, db_path):
    with open(file_path, 'rt') as f:
        interlangauge = json.load(f)

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        return [locate_single_topic_texts(pairs, c) for pairs in interlangauge]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Locate same topic texts over multiple languages.')
    parser.add_argument('--db', dest='db_path', default=index_db.default_path,
                        help='a sqlite database file generated by index.py')
    parser.add_argument('--input', dest='input_path',
                        default='interlanguage_topics.json',
                        help='a json file containing sets of topics over '
                             'multiple languages')
    parser.add_argument('--output', dest='output_path',
                        default='interlanguage_location.json',
                        help='a json file locating same topic texts over '
                             'multiple languages')
    args = parser.parse_args()
    location_infos = locate_interlanguage_texts(args.input_path, args.db_path)
    with open(args.output_path, 'wt') as f:
        json.dump(location_infos, f)
