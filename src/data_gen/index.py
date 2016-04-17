import os.path
import glob
import re
import sqlite3
import argparse
import traceback
from . import index_db
from itertools import islice, takewhile, count

# directory structure related
glob_pattern = './*wiki-20160305-pages-meta-current*/*/wiki_*'
lang_regex = re.compile(r'([a-z]+)wiki-20160305-pages-meta-current')

# parsing related
doc_begin_regex = \
    re.compile(r'<doc id="([0-9]+)" url="([0-9a-zA-Z.:/?=]+)" title="(.+)">\n')
doc_end_tag = '</doc>'
doc_min_size = 500

chunk_size = 10000

lang_set = set()


# iterate over corpus files generated by wikiextractor
# yields normalized file path and the corresponding language id
# refer glob_pattern/lang_regex to the expected directory structure
def corpus_files_iter(root_path):
    file_pattern = os.path.join(root_path, glob_pattern)
    for file_path in glob.iglob(file_pattern):
        file_path = os.path.normpath(file_path)
        re = lang_regex.search(file_path)
        if not re:
            continue
        yield re.group(1), file_path


# iterate over documents in a single file generated by wikiextractor
# yields document id, url, title, begin_pos, end_pos
# TODO : is there any documents that accidentally contains tags similar
#        to one generated by wikiextractor?
def documents_in_file_iter(text):
    # Unfortunately, the output of wikiextractor is not a valid xml.
    # But this is still well-structured, so we just use regex to parse it
    # TODO : I made a mistake here; begin and end should be byte offset,
    #        rather than string offset. Currently we need to read the whole
    #        file because of this reason.
    for i in doc_begin_regex.finditer(text):
        doc_id = int(i.group(1))
        url = i.group(2)
        title = i.group(3)
        begin = i.span()[1]
        end = text.find(doc_end_tag, begin)
        assert end != -1
        # url might not be needed
        yield doc_id, url, title, begin, end


def is_relevant_docs(doc_title, begin, end):
    # Skip documents that contains ':'
    # Those documents usually do not have useful informations
    if ':' in doc_title:
        return False
    # we want to skip too short documents
    # those are not representative
    if end - begin < doc_min_size:
        return False
    return True


# iterate over all docuements over all files
# note that the order of yielding tuple is consistent to the table column order
def all_documents_iter(root_path):
    for lang, doc_path in corpus_files_iter(root_path):
        lang_set.add(lang)
        try:
            with open(doc_path) as f:
                text = f.read()
                for data in documents_in_file_iter(text):
                    doc_id, url, title, begin, end = data
                    if not is_relevant_docs(title, begin, end):
                        continue
                    yield lang, doc_id, doc_path, url, title, begin, end
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            # even if there is a malicious data, keep going.
            print(doc_path, traceback.format_exc())


def split_every(it, n):
    return takewhile(bool, (list(islice(it, n)) for _ in count(0)))


def build_database(root_path, db_path):
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(index_db.drop_table)
        c.execute(index_db.create_table)
        for i, docs in enumerate(split_every(from .
                                 all_documents_iter(root_path), chunk_size)):
            print("{}th insertion...".format(i * chunk_size))
            c.executemany(index_db.insert_data, docs)
            conn.commit()
        c.execute(index_db.create_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build index DB from the corpora '
                    'generated by wikiextractor.')
    parser.add_argument('--path', dest='path', default='.',
                        help='a directory contains corpora to be used')
    parser.add_argument('--db-path', dest='db_path',
                        default=index_db.default_path,
                        help='a sqlite database file')
    args = parser.parse_args()
    build_database(args.path, args.db_path)
