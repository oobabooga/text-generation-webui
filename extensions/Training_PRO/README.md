# Training_PRO

This is an expanded Training tab
Maintained by FP

https://github.com/FartyPants/Training_PRO

- Chunking: precise raw text slicer (PRTS) uses sentence slicing and making sure things are clean on all ends
- overlap chunking - this special overlapping will make additional overlap block based on logical rules (aka no overlap block on hard cut)
- custom scheduler (follow the code to make your own) In LR Scheduler select FP_low_epoch_annealing - this scheduler will keep the LR constant for first epoch then use cosine for the rest - this part would be best to spawn into a new py file
- saves graph png file at the end with learning rate and loss per epoch
- adding EOS to each block or to hard cut only
- automatically lowers gradient accumulation if you go overboard and set gradient accumulation that will be higher than actual data - transformers would then throw error (or they used to, not sure if still true) but in any way, it will fix bad data
- turn BOS on and OFF
- target selector
- DEMENTOR LEARNING (experimental) Deep Memorization Enforcement Through Overlapping and Repetition. This is an experiment for long-text learning using low epochs (basically use 1 epoch with constant LR or 2 epochs with FP_low_epoch_annealing LR scheduler)
- Getting rid of micro batch size/batch size confusion. Now there is True Batch Size and Gradient accumulation slider, consisten with all the other training out there
- Ability to save Checkpoint during training with a button
- Ability to change Stop Loss during training
- different modes of checkpoint auto saving
- Function to Check Dataset and suggest parameters such as warmup and checkpoint save frequency before training
  
### Notes:

This uses it's own chunking code for raw text based on sentence splitting. This will avoid weird cuts in the chunks and each chunk should now start with sentence and end on some sentence. It works hand in hand with Hard Cut. A propper use is to structure your text into logical blocks (ideas) separated by three \n then use three \n in hard cut. This way each chunk will contain only one flow of ideas and not derail in the thoughts. And Overlapping code will create overlapped blocks on sentence basis too, but not cross hard cut, thus not cross different ideas either. Does it make any sense? No? Hmmmm...

### Targets

Normal LORA is q, v and that's what you should use. You can use (q k v o) or (q k v) and it will give you a lot more trainable parameters. The benefit is that you can keep rank lower and still attain the same coherency as q v with high rank. Guanaco has been trained with QLORA and q k v o for example and they swear by it.

### DEMENTOR LEARNING (experimental) Deep Memorization Enforcement Through Overlapping and Repetition

This is and experimental chunking to train long-form text in low number of epochs (basically 1) with sliding repetition. The depth of learning directly depends on the cutoff_length. Increasing cutoff length will also increase number of blocks created from long-form text (which is contrary to normal training). It is based on my own wild experiments. 

### Getting rid of batch size and micro batch size

Keeping consistency with everyone else. 

Listen, There is only ONE batch size - the True batch size (called previously micro-batch size in WebUI) - this is how many blocks are processed at once (during a single step). It eats GPU, but it really helps with the quality training (in fact the ideal batch size would be the same as number of blocks - which is unrealistic) - so the idea is to cram as much True Batch Size before your GPU blows with OOM. On 24GB this is about 10 for 13b (loaded with 4-bit)

So no micro batch size - it is now called True Batch Size, because that's what it is.

The other thing is Gradient Accumulation - this is an emulation of the above Batch size - a virtual batch size, if you will. If your GPU can't handle real batch size then you may fake it using Gradient Accumulation. This will accumulate the gradients over so many steps defined here and then update the weights at the end without increase in GPU.
Gradient accumulation is like a virtual Batch size multiplier without the GPU penalty.

If your batch size is 4 and your gradient accumulation is 2 then it sort of behaves as if we have batch size 8. *Sort of* because Batch size of 4 and GA of 2 is NOT the same as batch size of 2 and GA of 4. (It produces different weights - hence it's not an equivalent). The idea is that if you don't have GPU - using GA to extend batch size is the next best thing (good enough) since you have no other choice.

If all you can afford is 1 batch size, then increasing GA will likely make the learning better in some range of GA (it's not always more is better).

However - GA is not some golden goose. As said, it isn't the same as batch size. In fact GA may worsen your learning as well.

I would suggest a series of experiment where you would put batch size as high as possible without OOM, set GA 1, then repeat training while increasing the GA (2, 4...), and see how the model changes. It's likely that it would follow some sort of curve where GA will seem to help before it will make it worse. Some people believe that if you can squeeze 6 BATCH Size, then you should not bother with GA at all... YMMW

High Batch Size vs High GA would also likely produce different results in terms of learning  words vs style. How? Hmmmm... good question.

One optical "benefit" of GA is that the loss will fluctuate less (because of all the gradient accumulation, which works as a form of noise smoothing as well).
