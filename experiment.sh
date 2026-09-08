# Activate the environment
source .venv/bin/activate

python agent.py ReLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra

exit 0

python agent.py GELUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py Convolution2DFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SoftmaxFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py DotProductAttentionFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py RMSNormFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py ReductionSumFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SiLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SwiGLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra


exit 0


# tmux new -s dev  
# tmux attach -t dev  
# tmux kill-session -t dev

python agent.py FusedGEMMAddGELUFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py MatrixMultiplicationFloat16 --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py FlashAttentionFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py MatrixVectorMultiplicationFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py Convolution2DFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SoftmaxFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py DotProductAttentionFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py RMSNormFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py ReductionSumFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SiLUFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SwiGLUFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py GELUFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra