git clone -n --depth=1 --filter=tree:0 https://git.ecker.tech/mrq/tortoise-tts.git tortoise
cd tortoise
git sparse-checkout set --no-cone requirements.txt tortoise
awk '{gsub(/(^\s+--hash=[a-z:0-9]+\s*)|(\\)|(@HEAD)|(==[0-9A-z.]+)/,"")} 1' ./requirements.txt > ./req.txt
python -m pip install -r ./req.txt
cd ../
