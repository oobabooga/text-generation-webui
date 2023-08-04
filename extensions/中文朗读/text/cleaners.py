""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from unidecode import unidecode

from jamo import h2j, j2hcj
from pypinyin import lazy_pinyin, BOPOMOFO
import jieba, cn2an


# This is a list of Korean classifiers preceded by pure Korean numerals.
_korean_classifiers = (
    "군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통"
)

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

# List of (hangul, hangul divided) pairs:
_hangul_divided = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("ㄳ", "ㄱㅅ"),
        ("ㄵ", "ㄴㅈ"),
        ("ㄶ", "ㄴㅎ"),
        ("ㄺ", "ㄹㄱ"),
        ("ㄻ", "ㄹㅁ"),
        ("ㄼ", "ㄹㅂ"),
        ("ㄽ", "ㄹㅅ"),
        ("ㄾ", "ㄹㅌ"),
        ("ㄿ", "ㄹㅍ"),
        ("ㅀ", "ㄹㅎ"),
        ("ㅄ", "ㅂㅅ"),
        ("ㅘ", "ㅗㅏ"),
        ("ㅙ", "ㅗㅐ"),
        ("ㅚ", "ㅗㅣ"),
        ("ㅝ", "ㅜㅓ"),
        ("ㅞ", "ㅜㅔ"),
        ("ㅟ", "ㅜㅣ"),
        ("ㅢ", "ㅡㅣ"),
        ("ㅑ", "ㅣㅏ"),
        ("ㅒ", "ㅣㅐ"),
        ("ㅕ", "ㅣㅓ"),
        ("ㅖ", "ㅣㅔ"),
        ("ㅛ", "ㅣㅗ"),
        ("ㅠ", "ㅣㅜ"),
    ]
]

# List of (Latin alphabet, hangul) pairs:
_latin_to_hangul = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("a", "에이"),
        ("b", "비"),
        ("c", "시"),
        ("d", "디"),
        ("e", "이"),
        ("f", "에프"),
        ("g", "지"),
        ("h", "에이치"),
        ("i", "아이"),
        ("j", "제이"),
        ("k", "케이"),
        ("l", "엘"),
        ("m", "엠"),
        ("n", "엔"),
        ("o", "오"),
        ("p", "피"),
        ("q", "큐"),
        ("r", "아르"),
        ("s", "에스"),
        ("t", "티"),
        ("u", "유"),
        ("v", "브이"),
        ("w", "더블유"),
        ("x", "엑스"),
        ("y", "와이"),
        ("z", "제트"),
    ]
]

# List of (Latin alphabet, bopomofo) pairs:
_latin_to_bopomofo = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("a", "ㄟˉ"),
        ("b", "ㄅㄧˋ"),
        ("c", "ㄙㄧˉ"),
        ("d", "ㄉㄧˋ"),
        ("e", "ㄧˋ"),
        ("f", "ㄝˊㄈㄨˋ"),
        ("g", "ㄐㄧˋ"),
        ("h", "ㄝˇㄑㄩˋ"),
        ("i", "ㄞˋ"),
        ("j", "ㄐㄟˋ"),
        ("k", "ㄎㄟˋ"),
        ("l", "ㄝˊㄛˋ"),
        ("m", "ㄝˊㄇㄨˋ"),
        ("n", "ㄣˉ"),
        ("o", "ㄡˉ"),
        ("p", "ㄆㄧˉ"),
        ("q", "ㄎㄧㄡˉ"),
        ("r", "ㄚˋ"),
        ("s", "ㄝˊㄙˋ"),
        ("t", "ㄊㄧˋ"),
        ("u", "ㄧㄡˉ"),
        ("v", "ㄨㄧˉ"),
        ("w", "ㄉㄚˋㄅㄨˋㄌㄧㄡˋ"),
        ("x", "ㄝˉㄎㄨˋㄙˋ"),
        ("y", "ㄨㄞˋ"),
        ("z", "ㄗㄟˋ"),
    ]
]


# List of (bopomofo, romaji) pairs:
_bopomofo_to_romaji = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("ㄅㄛ", "p⁼wo"),
        ("ㄆㄛ", "pʰwo"),
        ("ㄇㄛ", "mwo"),
        ("ㄈㄛ", "fwo"),
        ("ㄅ", "p⁼"),
        ("ㄆ", "pʰ"),
        ("ㄇ", "m"),
        ("ㄈ", "f"),
        ("ㄉ", "t⁼"),
        ("ㄊ", "tʰ"),
        ("ㄋ", "n"),
        ("ㄌ", "l"),
        ("ㄍ", "k⁼"),
        ("ㄎ", "kʰ"),
        ("ㄏ", "h"),
        ("ㄐ", "ʧ⁼"),
        ("ㄑ", "ʧʰ"),
        ("ㄒ", "ʃ"),
        ("ㄓ", "ʦ`⁼"),
        ("ㄔ", "ʦ`ʰ"),
        ("ㄕ", "s`"),
        ("ㄖ", "ɹ`"),
        ("ㄗ", "ʦ⁼"),
        ("ㄘ", "ʦʰ"),
        ("ㄙ", "s"),
        ("ㄚ", "a"),
        ("ㄛ", "o"),
        ("ㄜ", "ə"),
        ("ㄝ", "e"),
        ("ㄞ", "ai"),
        ("ㄟ", "ei"),
        ("ㄠ", "au"),
        ("ㄡ", "ou"),
        ("ㄧㄢ", "yeNN"),
        ("ㄢ", "aNN"),
        ("ㄧㄣ", "iNN"),
        ("ㄣ", "əNN"),
        ("ㄤ", "aNg"),
        ("ㄧㄥ", "iNg"),
        ("ㄨㄥ", "uNg"),
        ("ㄩㄥ", "yuNg"),
        ("ㄥ", "əNg"),
        ("ㄦ", "əɻ"),
        ("ㄧ", "i"),
        ("ㄨ", "u"),
        ("ㄩ", "ɥ"),
        ("ˉ", "→"),
        ("ˊ", "↑"),
        ("ˇ", "↓↑"),
        ("ˋ", "↓"),
        ("˙", ""),
        ("，", ","),
        ("。", "."),
        ("！", "!"),
        ("？", "?"),
        ("—", "-"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def japanese_to_romaji_with_accent(text):
    """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""

    return text


def latin_to_hangul(text):
    for regex, replacement in _latin_to_hangul:
        text = re.sub(regex, replacement, text)
    return text


def divide_hangul(text):
    for regex, replacement in _hangul_divided:
        text = re.sub(regex, replacement, text)
    return text


def hangul_number(num, sino=True):
    """Reference https://github.com/Kyubyong/g2pK"""
    num = re.sub(",", "", num)

    if num == "0":
        return "영"
    if not sino and num == "20":
        return "스무"

    digits = "123456789"
    names = "일이삼사오육칠팔구"
    digit2name = {d: n for d, n in zip(digits, names)}

    modifiers = "한 두 세 네 다섯 여섯 일곱 여덟 아홉"
    decimals = "열 스물 서른 마흔 쉰 예순 일흔 여든 아흔"
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}

    spelledout = []
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        if sino:
            if i == 0:
                name = digit2name.get(digit, "")
            elif i == 1:
                name = digit2name.get(digit, "") + "십"
                name = name.replace("일십", "십")
        else:
            if i == 0:
                name = digit2mod.get(digit, "")
            elif i == 1:
                name = digit2dec.get(digit, "")
        if digit == "0":
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)) :]
                if "".join(last_three) == "":
                    spelledout.append("")
                    continue
            else:
                spelledout.append("")
                continue
        if i == 2:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 3:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 4:
            name = digit2name.get(digit, "") + "만"
            name = name.replace("일만", "만")
        elif i == 5:
            name = digit2name.get(digit, "") + "십"
            name = name.replace("일십", "십")
        elif i == 6:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 7:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 8:
            name = digit2name.get(digit, "") + "억"
        elif i == 9:
            name = digit2name.get(digit, "") + "십"
        elif i == 10:
            name = digit2name.get(digit, "") + "백"
        elif i == 11:
            name = digit2name.get(digit, "") + "천"
        elif i == 12:
            name = digit2name.get(digit, "") + "조"
        elif i == 13:
            name = digit2name.get(digit, "") + "십"
        elif i == 14:
            name = digit2name.get(digit, "") + "백"
        elif i == 15:
            name = digit2name.get(digit, "") + "천"
        spelledout.append(name)
    return "".join(elem for elem in spelledout)


def number_to_hangul(text):
    """Reference https://github.com/Kyubyong/g2pK"""
    tokens = set(re.findall(r"(\d[\d,]*)([\uac00-\ud71f]+)", text))
    for token in tokens:
        num, classifier = token
        if (
            classifier[:2] in _korean_classifiers
            or classifier[0] in _korean_classifiers
        ):
            spelledout = hangul_number(num, sino=False)
        else:
            spelledout = hangul_number(num, sino=True)
        text = text.replace(f"{num}{classifier}", f"{spelledout}{classifier}")
    # digit by digit for remaining digits
    digits = "0123456789"
    names = "영일이삼사오육칠팔구"
    for d, n in zip(digits, names):
        text = text.replace(d, n)
    return text


def number_to_chinese(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text


def chinese_to_bopomofo(text):
    text = text.replace("、", "，").replace("；", "，").replace("：", "，")
    words = jieba.lcut(text, cut_all=False)
    text = ""
    for word in words:
        bopomofos = lazy_pinyin(word, BOPOMOFO)
        if not re.search("[\u4e00-\u9fff]", word):
            text += word
            continue
        for i in range(len(bopomofos)):
            if re.match("[\u3105-\u3129]", bopomofos[i][-1]):
                bopomofos[i] += "ˉ"
        if text != "":
            text += " "
        text += "".join(bopomofos)
    return text


def latin_to_bopomofo(text):
    for regex, replacement in _latin_to_bopomofo:
        text = re.sub(regex, replacement, text)
    return text


def bopomofo_to_romaji(text):
    for regex, replacement in _bopomofo_to_romaji:
        text = re.sub(regex, replacement, text)
    return text


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def japanese_cleaners(text):
    text = japanese_to_romaji_with_accent(text)
    if re.match("[A-Za-z]", text[-1]):
        text += "."
    return text


def japanese_cleaners2(text):
    return japanese_cleaners(text).replace("ts", "ʦ").replace("...", "…")


def korean_cleaners(text):
    """Pipeline for Korean text"""
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = j2hcj(h2j(text))
    text = divide_hangul(text)
    if re.match("[\u3131-\u3163]", text[-1]):
        text += "."
    return text


def chinese_cleaners(text):
    """Pipeline for Chinese text"""
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    if re.match("[ˉˊˇˋ˙]", text[-1]):
        text += "。"
    return text


def zh_ja_mixture_cleaners(text):
    chinese_texts = re.findall(r"\[ZH\].*?\[ZH\]", text)
    japanese_texts = re.findall(r"\[JA\].*?\[JA\]", text)
    for chinese_text in chinese_texts:
        cleaned_text = number_to_chinese(chinese_text[4:-4])
        cleaned_text = chinese_to_bopomofo(cleaned_text)
        cleaned_text = latin_to_bopomofo(cleaned_text)
        cleaned_text = bopomofo_to_romaji(cleaned_text)
        cleaned_text = re.sub("i[aoe]", lambda x: "y" + x.group(0)[1:], cleaned_text)
        cleaned_text = re.sub("u[aoəe]", lambda x: "w" + x.group(0)[1:], cleaned_text)
        cleaned_text = re.sub(
            "([ʦsɹ]`[⁼ʰ]?)([→↓↑]+)",
            lambda x: x.group(1) + "ɹ`" + x.group(2),
            cleaned_text,
        ).replace("ɻ", "ɹ`")
        cleaned_text = re.sub(
            "([ʦs][⁼ʰ]?)([→↓↑]+)", lambda x: x.group(1) + "ɹ" + x.group(2), cleaned_text
        )
        text = text.replace(chinese_text, cleaned_text + " ", 1)
    for japanese_text in japanese_texts:
        cleaned_text = (
            japanese_to_romaji_with_accent(japanese_text[4:-4])
            .replace("ts", "ʦ")
            .replace("u", "ɯ")
            .replace("...", "…")
        )
        text = text.replace(japanese_text, cleaned_text + " ", 1)
    text = text[:-1]
    if re.match("[A-Za-zɯɹəɥ→↓↑]", text[-1]):
        text += "."
    return text
