#!/bin/bash
mkdir -p wikidata

if [ -f "list.txt" ]
then
	echo 'Using existing list.txt to download'
else
	echo 'Put all wiki data links into one file'
	python wikilist.py > list.txt
fi

# python wikilist.py | while read -r line; do cd wikidata && { curl -O "$line" ; cd -; }; done
while read -r line; do
	echo ${line##*/}
	cd wikidata
	{ curl -O# "$line" ; cd -; }
done <list.txt
# cd wikidata && { curl -O URL ; cd -; }
# cat file... | xargs -n1 command
# <file xargs -n1 command