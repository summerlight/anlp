import re

script_regex = re.compile(
    r'([0-9A-F]{4,5})(..([0-9A-F]{4,5}))? *; ([a-zA-Z]+) # ([a-zA-Z]+)'
)


def parse_script_data():
    with open('../data/Scripts.txt', 'rt') as f:
        for i in script_regex.finditer(f.read()):
            begin = int(i.group(1), 16)
            end = i.group(3)
            # We want to make this range half-open representation
            end = int(end, 16) + 1 if end else begin + 1
            script = i.group(4)
            category = i.group(5)  # This might not be needed, just in case

            yield begin, end, script, category


def merge_identical_ranges(seq):
    begin, end, script, _ = seq[0]

    for i, e, s, _ in seq[1:]:
        assert end <= i
        if end < i:
            # unknown/undefined script
            yield begin, end, script
            yield end, i, 'Unknown'
            begin, end, script = i, e, s
            continue

        assert end == i
        if script == s:
            # merge
            end = e
            continue
        else:
            yield begin, end, script
            begin, end, script = i, e, s
    yield begin, end, script

sorted_ranges = sorted(parse_script_data(), key=lambda x: x[0])
merged_ranges = [i for i in merge_identical_ranges(sorted_ranges)]

with open('script_map.py', 'wt') as f:
    f.write('import numpy\n\n')
    f.write('CODEPOINT = numpy.array([{}])  # noqa\n'.format(
            (', '.join(hex(i[1]-1) for i in merged_ranges))))
    f.write('SCRIPT = [{}, \'Unknown\']  # noqa\n\n\n'.format(
            ', '.join("'{}'".format(i[2]) for i in merged_ranges)))
    f.write('def script(chr):\n')
    f.write('    return SCRIPT[numpy.searchsorted(CODEPOINT, ord(chr))]\n')
