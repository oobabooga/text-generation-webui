import time
import datetime
import modules.shared as shared

if __name__ == "__main__":
    while True:
        print(shared.settings['name2'])
        print(shared.gradio['name2']) if 'name2' in shared.gradio else None
        print(datetime.datetime.now())
        time.sleep(1)  # print values every 10 seconds
