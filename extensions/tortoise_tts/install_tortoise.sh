git clone -n --depth=1 --filter=tree:0 https://github.com/neonbjb/tortoise-tts.git tortoise
cd tortoise
git sparse-checkout set --no-cone requirements.txt tortoise
python -m pip install -r ./requirements.txt
cd ../
