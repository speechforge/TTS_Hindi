from pathlib import Path
import os
import shutil
import zipfile
import unicodedata

import regex


BASE_DIR = Path("</path/to/your/IndicTTS_dataset/speaker1>")
WAV_DIR = BASE_DIR / "wav_22050"
TRANSCRIPT_FILE = BASE_DIR / "transcript.txt"

INDICMFA_DIR = Path("</path/to/your/IndicMFA_folder>")
ACOUSTIC_MODEL_DIR = INDICMFA_DIR / "Hindi_All_Acoustic"

OUT_CORPUS_DIR = INDICMFA_DIR / "Hindi_MFA_Corpus"
OUT_DICTIONARY = INDICMFA_DIR / "Hindi_Dictionary_g2g.txt"
OUT_ACOUSTIC_ZIP = INDICMFA_DIR / "Hindi_All_Acoustic.zip"

USE_SYMLINK = True


PUNCTUATION = set(
    "।॥,;:!?“”\"'`’‘()[]{}<>"
)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if ch in PUNCTUATION else ch for ch in text)
    text = " ".join(text.split())
    return text


def grapheme_space(text: str) -> str:
    text = clean_text(text)
    chars = []

    for token in text.split():
        for g in regex.findall(r"\X", token):
            if g.strip():
                chars.append(g)

    return " ".join(chars)


def parse_transcript_line(line: str):
    line = line.strip()

    if not line:
        return None

    line = line.strip("()").strip()

    q1 = line.find('"')
    q2 = line.rfind('"')

    if q1 == -1 or q2 == -1 or q2 <= q1:
        return None

    utt_id = line[:q1].strip()
    text = line[q1 + 1:q2].strip()

    if not utt_id or not text:
        return None

    return utt_id, text


def prepare_corpus():
    OUT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_missing = 0
    n_bad = 0
    missing_files = []

    lines = TRANSCRIPT_FILE.read_text(encoding="utf-8").splitlines()

    for line in lines:
        parsed = parse_transcript_line(line)

        if parsed is None:
            n_bad += 1
            continue

        utt_id, text = parsed

        src_wav = WAV_DIR / f"{utt_id}.wav"
        dst_wav = OUT_CORPUS_DIR / f"{utt_id}.wav"
        dst_lab = OUT_CORPUS_DIR / f"{utt_id}.lab"

        if not src_wav.exists():
            n_missing += 1
            missing_files.append(str(src_wav))
            continue

        if dst_wav.exists() or dst_wav.is_symlink():
            dst_wav.unlink()

        if USE_SYMLINK:
            os.symlink(src_wav, dst_wav)
        else:
            shutil.copy2(src_wav, dst_wav)

        dst_lab.write_text(grapheme_space(text) + "\n", encoding="utf-8")
        n_ok += 1

    print("Created MFA corpus:", OUT_CORPUS_DIR)
    print("Valid wav/lab pairs:", n_ok)
    print("Missing wav files:", n_missing)
    print("Bad transcript lines skipped:", n_bad)

    if missing_files:
        print("First missing files:")
        print("\n".join(missing_files[:10]))


def create_dictionary():
    grapheme_file = ACOUSTIC_MODEL_DIR / "graphemes.txt"

    if not grapheme_file.exists():
        raise FileNotFoundError(f"Missing grapheme file: {grapheme_file}")

    entries = []

    for line in grapheme_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if not parts:
            continue

        g = parts[0]
        entries.append(f"{g} {g}")

    OUT_DICTIONARY.write_text("\n".join(entries) + "\n", encoding="utf-8")
    print("Created dictionary:", OUT_DICTIONARY)


def create_acoustic_zip():
    if OUT_ACOUSTIC_ZIP.exists():
        OUT_ACOUSTIC_ZIP.unlink()

    with zipfile.ZipFile(OUT_ACOUSTIC_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ACOUSTIC_MODEL_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(ACOUSTIC_MODEL_DIR))

    print("Created acoustic model zip:", OUT_ACOUSTIC_ZIP)


def print_summary():
    wav_count = len(list(OUT_CORPUS_DIR.glob("*.wav")))
    lab_count = len(list(OUT_CORPUS_DIR.glob("*.lab")))

    print("\nSummary")
    print("WAV count:", wav_count)
    print("LAB count:", lab_count)
    print("Dictionary:", OUT_DICTIONARY)
    print("Acoustic model:", OUT_ACOUSTIC_ZIP)

    sample_lab = OUT_CORPUS_DIR / "train_hindimale_00001.lab"

    if sample_lab.exists():
        print("\nSample LAB:")
        print(sample_lab.read_text(encoding="utf-8").strip())


def main():
    prepare_corpus()
    create_dictionary()
    create_acoustic_zip()
    print_summary()


if __name__ == "__main__":
    main()
