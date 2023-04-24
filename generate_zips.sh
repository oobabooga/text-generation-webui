mkdir oobabooga_{windows,linux,macos}
for p in windows macos linux; do
  cp {*$p*\.*,webui.py,INSTRUCTIONS.TXT} oobabooga_$p;
  zip -r oobabooga_$p.zip oobabooga_$p;
done
