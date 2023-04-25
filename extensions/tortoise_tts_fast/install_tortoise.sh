git clone https://github.com/152334H/tortoise-tts-fast.git tortoise
cd tortoise
awk '{gsub(/(^\s+--hash=[a-z:0-9]+\s*)|(\\)|(@HEAD)|(==[0-9A-z.]+)/,"")} 1' ./requirements.txt > ./req.txt
sed -i '/^\s*$/d' ./req.txt
python -m pip install -r ./req.txt
cd ../
