#1.编译+推理
python3 lynxi_inference.py --batch_size 1 --model metaspikformer_8_512  --time_steps 4 --device apu:0 --compile True

#2.推理
python3 lynxi_inference.py --batch_size 1 --model metaspikformer_8_512  --time_steps 4 --device apu:0 
