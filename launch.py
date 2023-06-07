import os
import configparser

if __name__ == '__main__':
    config = configparser.ConfigParser()
    path = 'flags.ini'
    config.read(path)
    print('DEFAULT FLAGS:')
    print(*config.items('DEFAULT FLAGS'), sep='\n')
    print('\n')
    option = input("Please choose a flags #number or input flags:")
    try:
        int(option)
        flags = config.get('DEFAULT FLAGS', 'flags#' + option)
    except ValueError:
        flags = option
        flags_num = input("Would you like to save these flags as default? (y/n)")
        if flags_num == 'y':
            flags_num = input("Please input the #number of these flags you want to save to:")
            config.set('DEFAULT FLAGS', 'flags#' + flags_num, flags)
            config.write(open(path, 'w'))

    os.system("python server.py " + flags)