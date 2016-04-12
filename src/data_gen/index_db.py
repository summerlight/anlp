# This file is just for segregate ugly SQL queries from the codes...

drop_table = 'DROP TABLE IF EXISTS doc_index;'
create_table = '''
CREATE TABLE doc_index (
    lang TEXT,
    doc_id INT,
    doc_path TEXT,
    url TEXT,
    title TEXT,
    begin INT,
    end INT,
    PRIMARY KEY (lang, doc_id)
);'''
create_index = 'CREATE INDEX idx_lang_title ON doc_index (lang, title);'

# some documents are appearing more than once
# so ignore them from the second case
insert_data = 'INSERT OR IGNORE INTO doc_index VALUES (?,?,?,?,?,?,?);'

select_lang_title = '''
SELECT *
FROM doc_index
WHERE lang=? AND title=?;'''

random_selection = '''
SELECT *
FROM doc_index
WHERE rowid = (ABS(RANDOM()) % (SELECT (SELECT MAX(_ROWID_) FROM doc_index)+1)
);'''

default_path = 'index.db'
