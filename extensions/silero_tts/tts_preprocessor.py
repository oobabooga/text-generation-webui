import re
import locale
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
    "Z": " Zed "  # Zed is weird, as I (da3dsoul) am American, but most of the voice models sound British, so it matches
}


def preprocess(string):
    # the order for some of these matter
    # For example, you need to remove the commas in numbers before expanding them
    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = convert_num_locale(string)
    string = replace_negative(string)
    string = replace_roman(string)
    string = hyphen_range_to(string)
    string = num_to_words(string)

    # TODO Try to use a ML predictor to expand abbreviations. It's hard, dependent on context, and whether to actually
    # try to say the abbreviation or spell it out as I've done below is not agreed upon

    # For now, expand abbreviations to pronunciations
    # replace_abbreviations adds a lot of unnecessary whitespace to ensure separation
    string = replace_abbreviations(string)

    # cleanup whitespaces
    # remove whitespace before punctuation
    string = re.sub(r'\s+([,.?!\'])', r'\1', string)
    string = string.strip()
    # compact whitespace
    string = ' '.join(string.split())

    return string


def remove_surrounded_chars(string):
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub(r'\*[^*]*?(\*|$)', '', string)


def replace_negative(string):
    # handles situations like -5. -5 would become negative 5, which would then be expanded to negative five
    return re.sub(r'(\s)(-)(\d+)([\s,.?!)"\'\]>])', r'\1negative \3\4', string)


def replace_roman(string):
    # find a string of roman numerals.
    # Only 2 or more, to avoid capturing I and single character abbreviations, like names
    pattern = re.compile(r'\s[IVXLCDM]{2,}[\s,.?!)"\'\]>]')
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
    # 1000 or 10.23
    pattern = re.compile(r'\d+\.\d+|\d+')
    result = pattern.sub(lambda x: num2words(float(x.group())), text)
    return result


def replace_abbreviations(string):
    # abbreviations 1 to 4 characters long. It will get things like A and I, but those are pronounced with their letter
    pattern = re.compile(r'(^|[\s("\'\[<])([A-Z]{1,4})([\s,.?!)"\'\]>]|$)')
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


def convert_num_locale(text):
    # This detects locale and converts it to American without comma separators
    pattern = re.compile(r'(?:\s|^)\d{1,3}(?:\.\d{3})*(?:,\d+)?(?:\s|$)')
    result = text
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start] + result[start:end].replace('.', '').replace(',', '.') + result[end:len(result)]

    # removes comma separators from existing American numbers
    pattern = re.compile(r'(\d),(\d)')
    result = pattern.sub(r'\1\2', result)

    return result


def __main__(args):
    print(preprocess(args[1]))


if __name__ == "__main__":
    import sys
    __main__(sys.argv)
