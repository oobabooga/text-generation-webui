import requests

# For local streaming, the websockets are hosted without ssl - http://
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/completions'

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'


def run(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 250,
        'do_sample': True,
        'temperature': 1.3,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()
        print(result)


if __name__ == '__main__':
    prompt = "In order to make homemade bread, follow these steps:\n1)"
    run(prompt)


    """
    {
   "choices":[
      {
         "finish_reason":"stop",
         "index":0,
         "logprobs":"None",
         "text":" Mix the flour and salt in a bowl. Add water gradually until you have a soft dough 
         that is not sticky or too dry. Knead for 5 minutes by hand (or use an electric mixer). Cover 
         with plastic wrap and let rise at room temperature for about one hour.\n2) Punch down the dough 
         and divide into two pieces. Roll each piece out on a lightly-floured surface so it’s about half as 
         thick as your thumb. Cut into desired shapes using cookie cutters or knives. Place on greased baking
         sheets and allow them to rest again while they are rising. Bake at 375 degrees Fahrenheit for 
         approximately 40 minutes.\nThe best part of making homemade bread? The smell! It fills up my whole 
         house when I am kneading the dough. And then there is nothing like biting into freshly made bread 
         right from the oven…mmm mmm good!"
      }
   ],
   "created":1686072316,
   "id":"cmpl-shTPUqteQFT7ENxh98YzzKdeNH",
   "model":"decapoda-research_llama-7b-hf",
   "object":"text_completion",
   "usage":{
      "completion_tokens":208,
      "prompt_tokens":17,
      "total_tokens":225
   }
}
    
    """