#1.compile and inference
python3 lynxi_inference.py --batch_size 1 --model metaspikformer_8_512  --time_steps 4 --device apu:0 --compile True

#2.inference only
python3 lynxi_inference.py --batch_size 1 --model metaspikformer_8_512  --time_steps 4 --device apu:0 