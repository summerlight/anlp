import icu

locale = icu.Locale.getUS()


def by_(s, func):
    boundary = func(locale)
    boundary.setText(s)
    start = boundary.first()
    for end in boundary:
        s = boundary.getText().getText()
        yield s[start:end]
        start = end


def by_words(s):
    return by_(s, icu.BreakIterator.createWordInstance)


def by_sentences(s):
    return by_(s, icu.BreakIterator.createSentenceInstance)
