This is an expanded Training tab


- Chunking: precise raw text slicer (PRTS) uses sentence slicing and making sure things are clean on all ends
- overlap chunking - this special overlapping will make additional overlap block based on logical rules (aka no overlap block on hard cut)
- custom scheduler (follow the code to make your own) In LR Scheduler select FP_low_epoch_annealing - this scheduler will keep the LR constant for first epoch then use cosine for the rest - this part would be best to spawn into a new py file
- save loss threshold - will not save the "Save every n steps" checkpoints until this threshold is reached (I definitely don't need multiple checkpoints that are 2.5 loss - I'm usually interested in checkpoints between say 1.5 and 1.9 loss)
- saves graph png file at the end with learning rate and loss per epoch
- adding EOS to each block or to hard cut only
- automatically lowers gradient accumulation if you go overboard and set gradient accumulation that will be higher than actual data - transformers would then throw error (or they used to, not sure if still true) but in any way, it will fix bad data
- turn BOS on and OFF
