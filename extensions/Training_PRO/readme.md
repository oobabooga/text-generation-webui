This is an expanded Training tab


- Chunking: precise raw text slicer (PRTS) uses sentence slicing and making sure things are clean on all ends
- overlap chunking - this special overlapping will make additional overlap block based on logical rules (aka no overlap block on hard cut)
- custom scheduler (follow the code to make your own) In LR Scheduler select FP_low_epoch_annealing - this scheduler will keep the LR constant for first epoch then use cosine for the rest - this part would be best to spawn into a new py file
- save loss threshold - will not save the "Save every n steps" checkpoints until this threshold is reached (I definitely don't need multiple checkpoints that are 2.5 loss - I'm usually interested in checkpoints between say 1.5 and 1.9 loss)
- saves graph png file at the end with learning rate and loss per epoch
- adding EOS to each block or to hard cut only
- automatically lowers gradient accumulation if you go overboard and set gradient accumulation that will be higher than actual data - transformers would then throw error (or they used to, not sure if still true) but in any way, it will fix bad data
- turn BOS on and OFF
- target selector

###Notes:

This uses it's own chunking code for raw text based on sentence splitting. This will avoid weird cuts in the chunks and each chunk should now start with sentence and end on some sentence. It works hand in hand with Hard Cut.
A propper use is to structure your text into logical blocks (ideas) separated by three \n then use three \n in hard cut.
This way each chunk will contain only one flow of ideas and not derail in the thoughts. 
And Overlapping code will create overlapped blocks on sentence basis too, but not cross hard cut, thus not cross different ideas either. 
Does it make any sense? No? Hmmmm...

###Targets

Normal LORA is q, v and that's what you should use.
You can use (q k v o) or (q k v) and it will give you a lot more trainable parameters. The benefit is that you can keep rank lower and still attain the same coherency as q v with high rank. Guanaco has been trained with QLORA and q k v o for example and they swear by it.
I also added k-v-down which is lifted from IA3, which is very odd one to use for LORA, but it created adorable style craziness when training on raw structured text and bringing the loss all the way down to 1.1 . It didn't overfit (q-v would be just writing entire novels at loss 1.1) and it followed the instruction seeping from the previous fine-tuning. YMMW of course.
Using All will train all 7 targets q-k-v-o-up,down, gate - not sure if there is much benefit from attention only qkvo. It sure makes LORA huge. If that's what you like.
