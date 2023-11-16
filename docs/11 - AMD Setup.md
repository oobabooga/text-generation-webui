## Using an AMD GPU in Linux

Requires ROCm SDK 5.4.2 or 5.4.3 to be installed. Some systems may also
need: 

```
sudo apt-get install libstdc++-12-dev
```

Edit the "one_click.py" script using a text editor and un-comment and
modify the lines near the top of the script according to your setup. In
particular, modify the `os.environ["ROCM_PATH"] = '/opt/rocm'` line to
point to your ROCm installation.
