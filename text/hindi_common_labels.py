"""Hindi native-script to Common Label Set token mapping."""

import re


SPACE_TOKEN = "SP"

COMMON_LABEL_MAP = {
  "अ": "A", "आ": "AA", "इ": "I", "ई": "II", "उ": "U", "ऊ": "UU",
  "ऋ": "RI", "ए": "E", "ऐ": "AI", "ओ": "O", "औ": "AU",
  "क": "K", "ख": "KH", "ग": "G", "घ": "GH", "ङ": "NG",
  "च": "CH", "छ": "CHH", "ज": "J", "झ": "JH", "ञ": "NY",
  "ट": "T", "ठ": "TH", "ड": "D", "ढ": "DH", "ण": "N",
  "त": "T_d", "थ": "TH_d", "द": "D_d", "ध": "DH_d", "न": "N_d",
  "प": "P", "फ": "PH", "ब": "B", "भ": "BH", "म": "M",
  "य": "Y", "र": "R", "ल": "L", "व": "V", "श": "SH", "ष": "SH_r",
  "स": "S", "ह": "H",
  "क़": "Q", "ख़": "KH_n", "ग़": "GH_n", "ज़": "Z", "ड़": "RD",
  "ढ़": "RDH", "फ़": "F",
  "ा": "AA_m", "ि": "I_m", "ी": "II_m", "ु": "U_m", "ू": "UU_m",
  "ृ": "RI_m", "े": "E_m", "ै": "AI_m", "ो": "O_m", "ौ": "AU_m",
  "ं": "AN", "ँ": "AM", "ः": "AH", "़": "NUQTA", "ऽ": "AVAGRAHA",
  "्": "HALANT",
  "ऑ": "EXT_001", "ऱ": "EXT_002", "ॅ": "EXT_003", "ॉ": "EXT_004",
}

PUNCTUATION_MAP = {
  "।": ".",
  "॥": ".",
  ",": ",",
  ".": ".",
  ";": ";",
  ":": ":",
  "!": "!",
  "?": "?",
}

COMMON_LABEL_SYMBOLS = list(dict.fromkeys(
  list(COMMON_LABEL_MAP.values()) + [SPACE_TOKEN]))

_space_re = re.compile(r"\s+")


def hindi_to_common_labels(text, strict=True):
  labels = []
  text = _space_re.sub(" ", text.strip())
  for char in text:
    if char.isspace():
      if labels and labels[-1] != SPACE_TOKEN:
        labels.append(SPACE_TOKEN)
      continue
    if char in PUNCTUATION_MAP:
      labels.append(PUNCTUATION_MAP[char])
      continue
    if char in COMMON_LABEL_MAP:
      labels.append(COMMON_LABEL_MAP[char])
      continue
    if "\u0900" <= char <= "\u097f":
      if strict:
        raise ValueError("No Common Label mapping for Hindi character {!r}".format(char))
      continue
    # Drop non-Hindi formatting marks or unsupported Latin artifacts.
  while labels and labels[-1] == SPACE_TOKEN:
    labels.pop()
  return labels


def hindi_to_common_label_text(text, strict=True):
  return " ".join(hindi_to_common_labels(text, strict=strict))
