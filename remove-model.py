import argparse
import os
import sys
import yaml
import shutil
import time

'''
QnD Removes models from models/username_modelname.

Example:
$ python remove-model.py modelGG-1.3b

'''


def model_folder(input_model, base_folder=None):
    if base_folder is None:
        return os.path.abspath(os.getcwd()) + os.sep + "models" + os.sep + input_model
    else:
        return base_folder + os.sep + input_model


def config_folder():
    return os.path.abspath(os.getcwd()) + os.sep + "models"


def get_keys_yaml(output_folder, configfile="config-user.yaml"):
    conf = str(output_folder) + os.sep + configfile
    with open(conf, 'r') as fp:
        config_data = yaml.safe_load(fp)
    return conf, config_data


def delete_from_config(dump, fp):
    yaml.dump(dump, fp)
    fp.truncate()


def handle_keys(output_folder, model, configfile):

    backup_config_file = configfile + ".bak." + str(int(time.time())) + ".yaml"

    shutil.copyfile(configfile, backup_config_file)
    f = model_folder(model, output_folder)
    config_model_name = model + "$"

    with open(configfile, 'r+') as fp:
        content = yaml.safe_load(fp)

        try:
            del content[config_model_name]
            print(f"Deleting : {model}")
        except KeyError as e:
            print(f"KeyError {e} while deleting {model}")
        finally:
            fp.seek(0)

        try:
            if os.listdir(f):
                try:
                    print(f"Deleting files in {f}")
                    shutil.rmtree(f)
                except:
                    pass
                finally:
                    print(f"Removing {model} from config file {configfile}")
                    delete_from_config(content, fp)

        except FileNotFoundError:
            print("Directory not found.")
            try:
                print(f"Removing {model} leftover from config file {configfile}")
                delete_from_config(content, fp)
            except KeyError as e:
                print(f"KeyError {e} while deleting model from config")
        finally:
            print(f"Model {model} deleted.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL', type=str, default=None, nargs='?', help="Model to be removed")
    parser.add_argument('--list', action='store_true', default=None, help='Only list the models found in config file')
    args = parser.parse_args()
    config_folder = config_folder()

    config, data = get_keys_yaml(config_folder)

    if args.list:
        for k in data:
            print(k)
        sys.exit()

    if args.MODEL is None:
        print("Error: Please specify the model you'd like to remove ('python remove-model.py modelGG-1.3b').")
        sys.exit()

    handle_keys(config_folder, args.MODEL, config)

