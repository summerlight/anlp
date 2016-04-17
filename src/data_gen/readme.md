This folder contains some scripts for data generation. Since their implementations are pretty straightforward, I don't write heavy documents about how to use them. Please refer to the codes.

 * index.py - This script generates a sqlite3 index database for Wikipedia corpora. The primary reason of this script is to speed up other data generation tasks.
 * interlanguage_topics.json - This json data file contains an array of sets containing metadata of the same topic documents over multiple languages. We need this data in order to imitate ALTA-2010 data generation process.
 * locate_text.py - Running this script preprocesses interlanguage_topics.json into a more usable format by using the sqlite3 index database. Running `index.py` should precede to this script.
 * generate_text.py - This module contains a basic framework for generating multilingual documents. It depends on the output of `locate_text.py`.

We use PyICU for word/sentence-level segmentation. You'll need to install it.