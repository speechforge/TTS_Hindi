""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  clean_text = _clean_text(text, cleaner_names)
  return _symbols_to_sequence(clean_text)


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  return _symbols_to_sequence(cleaned_text)


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(text):
  tokenized = text.strip().split()
  if tokenized and all(symbol in _symbol_to_id for symbol in tokenized):
    return [_symbol_to_id[symbol] for symbol in tokenized]

  sequence = []
  missing = []
  for symbol in text:
    if symbol in _symbol_to_id:
      sequence.append(_symbol_to_id[symbol])
    else:
      missing.append(symbol)
  if missing:
    missing_unique = sorted(set(missing))
    raise KeyError("Text contains symbols missing from vocabulary: {}".format(missing_unique[:20]))
  return sequence
