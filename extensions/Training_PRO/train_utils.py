import os
from modules import shared, utils
from pathlib import Path
import requests
import tqdm
import json

'''
def get_gpu_memory_usage(rank):
    return {
        'total': round(torch.cuda.get_device_properties(rank).total_memory / (1024**3), 2),
        'max': round(torch.cuda.max_memory_allocated(rank) / (1024**3), 2),
        'reserved': round(torch.cuda.memory_reserved(rank) / (1024**3), 2),
        'allocated': round(torch.cuda.memory_allocated(rank) / (1024**3), 2)
    }
'''

def list_subfoldersByTime(directory):

    if not directory.endswith('/'):
        directory += '/'
    subfolders = []
    subfolders.append('None') 
    path = directory
    name_list = os.listdir(path)
    full_list = [os.path.join(path,i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime,reverse=True)

    for entry in time_sorted_list:
        if os.path.isdir(entry):
            entry_str = f"{entry}"  # Convert entry to a string
            full_path = entry_str
            entry_str = entry_str.replace('\\','/')
            entry_str = entry_str.replace(f"{directory}", "")  # Remove directory part
            subfolders.append(entry_str)

    return subfolders

def get_available_loras_local(_sortedByTime):
    
    model_dir = shared.args.lora_dir  # Update with the appropriate directory path
    subfolders = []
    if _sortedByTime:
        subfolders = list_subfoldersByTime(model_dir)
    else:
        subfolders = utils.get_available_loras()        

    return subfolders


# FPHAM SPLIT BY SENTENCE BLOCK ===============
     
def split_sentences(text: str, cutoff_len: int):
    sentences = []
    sentence = ''
    delimiters = ['. ', '? ', '! ', '... ', '.\n', '?\n', '!\n','...\n','</s>','<//>']
    abbreviations = ['Mr. ', 'Mrs. ', 'Dr. ', 'Ms. ', 'St. ', 'Prof. ', 'Jr. ', 'Ltd. ', 'Capt. ', 'Col. ', 'Gen. ', 'Ave. ', 'Blvd. ', 'Co. ', 'Corp. ', 'Dept. ', 'Est. ', 'Gov. ', 'Inc. ', 'Ph.D. ', 'Univ. ']
    errors = 0
    max_cut = cutoff_len-1
    prev_char = ''  

    for char in text:
        sentence += char

    
        if (any(sentence.endswith(delimiter) for delimiter in delimiters) and
            not (prev_char.isupper() and len(sentence) >= 3 and sentence[-3] != ' ') and 
            not any(sentence.endswith(abbreviation) for abbreviation in abbreviations)):
            tokens = shared.tokenizer.encode(sentence)
            
            if len(tokens) > max_cut:
                tokens = tokens[:max_cut]
                sentence = shared.tokenizer.decode(tokens, skip_special_tokens=True)
                errors = errors + 1

            sentences.append({'text': sentence, 'size': len(tokens)})
            
            sentence = ''

        prev_char = char

    if sentence:
        tokens = shared.tokenizer.encode(sentence)
        if len(tokens) > max_cut:
            tokens = tokens[:max_cut]
            sentence = shared.tokenizer.decode(tokens, skip_special_tokens=True)  
            errors = errors + 1

        sentences.append({'text': sentence, 'size': len(tokens)})

    if errors > 0:
        print(f"Trimmed sentences beyond Cutoff Length: {errors}")

    return sentences

# The goal of following code is to create blocks of text + overlapping blocks while:
# respects sentence boundaries
# always uses all the text 
# hard cut defined by hard_cut_string or </s> will always end at the end of data block
# no overlapping blocks will be created across hard cut or across </s> token

def precise_cut(text: str, overlap: bool, min_chars_cut: int, eos_to_hc: bool, cutoff_len: int, hard_cut_string: str, debug_slicer:bool):

    EOSX_str = '<//>' #hardcut placeholder
    EOS_str = '</s>' 
    print("Precise raw text slicer: ON")
    
    cut_string = hard_cut_string.replace('\\n', '\n')
    text = text.replace(cut_string, EOSX_str)
    sentences = split_sentences(text, cutoff_len)

    print(f"Sentences: {len(sentences)}")
    sentencelist = []
    currentSentence = ''
    totalLength = 0
    max_cut = cutoff_len-1
    half_cut = cutoff_len//2
    halfcut_length = 0

    edgeindex = []
    half_index = 0

    for index, item in enumerate(sentences):
        
        if halfcut_length+ item['size'] < half_cut:
            halfcut_length += item['size']
            half_index = index
        else:
            edgeindex.append(half_index)
            halfcut_length = -2 * max_cut


        if totalLength + item['size'] < max_cut and not currentSentence.endswith(EOSX_str): 
            currentSentence += item['text']
            totalLength += item['size']
        else:

            if len(currentSentence.strip()) > min_chars_cut:
                sentencelist.append(currentSentence.strip())

            currentSentence = item['text']
            totalLength = item['size']
            halfcut_length = item['size']
            
    if len(currentSentence.strip()) > min_chars_cut:    
        sentencelist.append(currentSentence.strip())

    unique_blocks = len(sentencelist)
    print(f"Text Blocks: {unique_blocks}")

    #overlap strategies: 
    # don't overlap across HARD CUT (EOSX)
    if overlap:
        for edge_idx in edgeindex:
            currentSentence = ''
            totalLength = 0

            for item in sentences[edge_idx:]:
                if totalLength + item['size'] < max_cut:
                    currentSentence += item['text']
                    totalLength += item['size']
                else:
                    #if by chance EOSX is at the end then it's acceptable
                    if currentSentence.endswith(EOSX_str) and len(currentSentence.strip()) > min_chars_cut:
                            sentencelist.append(currentSentence.strip())    
                    # otherwise don't cross hard cut    
                    elif EOSX_str not in currentSentence and len(currentSentence.strip()) > min_chars_cut:
                        sentencelist.append(currentSentence.strip())
                    
                    currentSentence = ''
                    totalLength = 0
                    break
        
        print(f"+ Overlapping blocks: {len(sentencelist)-unique_blocks}")

    num_EOS = 0
    for i in range(len(sentencelist)):
        if eos_to_hc:
            sentencelist[i] = sentencelist[i].replace(EOSX_str, EOS_str)
        else:
            sentencelist[i] = sentencelist[i].replace(EOSX_str, '')
        
        #someone may have had stop strings in the raw text...
        sentencelist[i] = sentencelist[i].replace("</s></s>", EOS_str)
        num_EOS += sentencelist[i].count(EOS_str)

    if num_EOS > 0:
        print(f"+ EOS count: {num_EOS}")

    #final check for useless lines
    sentencelist = [item for item in sentencelist if item.strip() != "</s>"]
    sentencelist = [item for item in sentencelist if item.strip() != ""]


    if debug_slicer:
                    # Write the log file
        Path('user_data/logs').mkdir(exist_ok=True)
        sentencelist_dict = {index: sentence for index, sentence in enumerate(sentencelist)}
        output_file = "user_data/logs/sentencelist.json"
        with open(output_file, 'w') as f:
            json.dump(sentencelist_dict, f,indent=2)
        
        print("Saved sentencelist.json in user_data/logs folder")
    
    return sentencelist   


def sliding_block_cut(text: str, min_chars_cut: int, eos_to_hc: bool, cutoff_len: int, hard_cut_string: str, debug_slicer:bool):

    EOSX_str = '<//>' #hardcut placeholder
    EOS_str = '</s>' 
    print("Mega Block Overlap: ON")
    
    cut_string = hard_cut_string.replace('\\n', '\n')
    text = text.replace(cut_string, EOSX_str)
    sentences = split_sentences(text, cutoff_len)

    print(f"Sentences: {len(sentences)}")
    sentencelist = []
    
    max_cut = cutoff_len-1

    #print(f"max_cut: {max_cut}")
    advancing_to = 0

    prev_block_lastsentence = ""
    

    for i in range(len(sentences)):
        totalLength = 0
        currentSentence = ''
        lastsentence = ""
        
        if i >= advancing_to:
            for k in range(i, len(sentences)):
                
                current_length = sentences[k]['size']

                if totalLength + current_length <= max_cut and not currentSentence.endswith(EOSX_str):
                    currentSentence += sentences[k]['text']
                    totalLength += current_length
                    lastsentence = sentences[k]['text']
                else:
                    if len(currentSentence.strip()) > min_chars_cut:
                        if prev_block_lastsentence!=lastsentence:
                            sentencelist.append(currentSentence.strip())
                            prev_block_lastsentence = lastsentence
                        
                    advancing_to = 0
                    if currentSentence.endswith(EOSX_str):
                        advancing_to = k

                    currentSentence = ""
                    totalLength = 0
                    break
            
            if currentSentence != "":
                if len(currentSentence.strip()) > min_chars_cut:
                    sentencelist.append(currentSentence.strip())

    unique_blocks = len(sentencelist)
    print(f"Text Blocks: {unique_blocks}")
    num_EOS = 0
    for i in range(len(sentencelist)):
        if eos_to_hc:
            sentencelist[i] = sentencelist[i].replace(EOSX_str, EOS_str)
        else:
            sentencelist[i] = sentencelist[i].replace(EOSX_str, '')
        
        #someone may have had stop strings in the raw text...
        sentencelist[i] = sentencelist[i].replace("</s></s>", EOS_str)
        num_EOS += sentencelist[i].count(EOS_str)

    if num_EOS > 0:
        print(f"+ EOS count: {num_EOS}")

    #final check for useless lines
    sentencelist = [item for item in sentencelist if item.strip() != "</s>"]
    sentencelist = [item for item in sentencelist if item.strip() != ""]


    if debug_slicer:
                    # Write the log file
        Path('user_data/logs').mkdir(exist_ok=True)
        sentencelist_dict = {index: sentence for index, sentence in enumerate(sentencelist)}
        output_file = "user_data/logs/sentencelist.json"
        with open(output_file, 'w') as f:
            json.dump(sentencelist_dict, f,indent=2)
        
        print("Saved sentencelist.json in user_data/logs folder")
    
    return sentencelist   

# Example usage:
# download_file_from_url('https://example.com/path/to/your/file.ext', '/output/directory')

def download_file_from_url(url, overwrite, output_dir_in, valid_extensions = {'.txt', '.json'}):
    try:
    # Validate and sanitize the URL
    #parsed_url = urllib.parse.urlparse(url)
    #if not parsed_url.netloc:
    #    raise ValueError("Invalid URL")
    #filename = os.path.basename(parsed_url.path)

    # Get the filename from the URL

        session = requests.Session()
        headers = {}
        mode = 'wb'
        filename = url.split('/')[-1]

        output_dir = str(output_dir_in)
        # Construct the full path to the output file
        local_filename = os.path.join(output_dir, filename)

        # Check if the local file already exists
        overw = ''
        if os.path.exists(local_filename):
            if not overwrite:
                yield f"File '{local_filename}' already exists. Aborting."
                return
            else:
                overw = ' [Overwrite existing]'

        filename_lower = filename.lower()

        # Send an HTTP GET request to the URL with a timeout
        file_extension = os.path.splitext(filename_lower)[-1]
        
        if file_extension not in valid_extensions:
            yield f"Invalid file extension: {file_extension}. Only {valid_extensions} files are supported."
            return

        with session.get(url, stream=True, headers=headers, timeout=10) as r:
            r.raise_for_status() 
            # total size can be wildly inaccurate
            #total_size = int(r.headers.get('content-length', 0))
            
            block_size = 1024 * 4  
            with open(local_filename, mode) as f:
                count = 0
                for data in r.iter_content(block_size):
                    f.write(data)
                    count += len(data)

                    yield f"Downloaded: {count} " + overw

            # Verify file size if possible
            if os.path.exists(local_filename):
                downloaded_size = os.path.getsize(local_filename)
                if downloaded_size > 0:
                    yield f"File '{filename}' downloaded to '{output_dir}' ({downloaded_size} bytes)."
                    print("File Downloaded")
                else:
                    print("Downloaded file is zero")
                    yield f"Failed. Downloaded file size is zero)."
            else:
                print(f"Error: {local_filename} failed to download.")
                yield f"Error: {local_filename} failed to download"

    except Exception as e:
        print(f"An error occurred: {e}")
        yield f"An error occurred: {e}"

    finally:
        # Close the session to release resources
        session.close()

