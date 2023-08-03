mkdir oobabooga_{windows,linux,macos,wsl}
for p in windows macos linux wsl; do
  if [ "$p" == "wsl" ]; then cp {*$p*\.*,webui.py,INSTRUCTIONS-WSL.txt} oobabooga_$p;
  else cp {*$p*\.*,webui.py,INSTRUCTIONS.txt,CMD_FLAGS.txt} oobabooga_$p; fi
  zip -r oobabooga_$p.zip oobabooga_$p;
done
