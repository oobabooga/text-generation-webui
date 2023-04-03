import re
from num2words import num2words


alphabet_map = {
    "A": " Ei ",
    "B": " Bee ",
    "C": " See ",
    "D": " Dee ",
    "E": " Ii ",
    "F": " Eff ",
    "G": " Jee ",
    "H": " Eich ",
    "I": " Eye ",
    "J": " Jay ",
    "K": " Kay ",
    "L": " El ",
    "M": " Emm ",
    "N": " Enn ",
    "O": " Ohh ",
    "P": " Pii ",
    "Q": " Queue ",
    "R": " Are ",
    "S": " Ess ",
    "T": " Tee ",
    "U": " You ",
    "V": " Vii ",
    "W": " Double You ",
    "X": " Ex ",
    "Y": " Why ",
    "Z": "Zed"  # Zed is weird, as I (da3dsoul) am American, but most of the voice models sound British, so it matches
}


def preprocess(string):
    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = remove_commas(string)
    string = replace_roman(string)
    string = hyphen_range_to(string)
    string = num_to_words(string)

    # TODO Try to use a ML predictor to expand abbreviations. It's hard, dependent on context, and whether to actually
    # try to say the abbreviation or spell it out as I've done below is not agreed upon

    # For now, expand abbreviations to pronunciations
    string = replace_abbreviations(string)

    string = string.strip()
    return string


def remove_surrounded_chars(string):
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub(r'\*[^*]*?(\*|$)', '', string)


def replace_roman(string):
    pattern = re.compile(r'\s[IVXLCDM]+[\s,.?!)"\'\]>]')
    result = string
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start+1] + str(roman_to_int(result[start+1:end-1])) + result[end-1:len(result)]

    return result


def roman_to_int(s):
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val


def hyphen_range_to(text):
    pattern = re.compile(r'(\d+)[-–](\d+)')
    result = pattern.sub(lambda x: x.group(1) + ' to ' + x.group(2), text)
    return result


def num_to_words(text):
    pattern = re.compile(r'\d+')
    result = pattern.sub(lambda x: num2words(int(x.group())), text)
    return result


def replace_abbreviations(string):
    pattern = re.compile(r'[\s("\'\[<][A-Z]{2,4}[\s,.?!)"\'\]>]')
    result = string
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start] + replace_abbreviation(result[start:end]) + result[end:len(result)]

    return result


def replace_abbreviation(string):
    result = ""
    for char in string:
        result = match_mapping(char, result)

    return result


def match_mapping(char, result):
    for mapping in alphabet_map.keys():
        if char == mapping:
            return result + alphabet_map[char]

    return result + char


def remove_commas(text):
    import re
    pattern = re.compile(r'(\d),(\d)')
    result = pattern.sub(r'\1\2', text)
    return result


def __main__(args):
    print(preprocess(args[1]))


if __name__ == "__main__":
    import sys
    __main__(sys.argv)
