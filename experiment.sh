# Activate the environment
source .venv/bin/activate

python3 agent.py RMSNormFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python3 agent.py RMSNormFloat8Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python3 agent.py Convolution2DFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python3 agent.py Convolution2DFloat8Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python3 agent.py MatrixVectorMultiplicationFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python3 agent.py MatrixVectorMultiplicationFloat8Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra


exit 0


# OpenRouter pricing is input/output USD per million tokens as of 2026-09-15.
# Gemini 3.8 Flash
python -m agent MatrixMultiplicationFloat16 --provider openrouter --model google/gemini-3.8-flash --reasoning-effort max --max-tool-rounds 0 --trace-path astra
# Qwen 3.8 Max (0902): $2.00 / $6.00
python -m agent MatrixMultiplicationFloat16 --provider openrouter --model qwen/qwen3.8-max-0902 --reasoning-effort max --max-tool-rounds 0 --trace-path astra
# GLM 5.3: $0.8775 / $2.97
python -m agent MatrixMultiplicationFloat16 --provider openrouter --model z-ai/glm-5.3 --reasoning-effort max --max-tool-rounds 0 --trace-path astra
# GLM 5.3 Flash: $0.075 / $0.25
python -m agent MatrixMultiplicationFloat16 --provider openrouter --model z-ai/glm-5.3-flash --reasoning-effort max --max-tool-rounds 0 --trace-path astra
# DeepSeek V4.1 Flash: $0.15 / $0.60
python -m agent MatrixMultiplicationFloat16 --provider openrouter --model deepseek/deepseek-v4.1-flash --reasoning-effort max --max-tool-rounds 0 --trace-path astra
# Kimi K3: $2.10 / $10.95
python -m agent MatrixMultiplicationFloat16 --provider openrouter --model moonshotai/kimi-k3 --reasoning-effort max --max-tool-rounds 0 --trace-path astra
# Xiaomi MiMo-V2.5: $0.119 / $0.238
python -m agent MatrixMultiplicationFloat16 --provider openrouter --model xiaomi/mimo-v2.5 --reasoning-effort max --max-tool-rounds 0 --trace-path astra
# NVIDIA Nemotron 3 Ultra: $0.50 / $2.20
python -m agent MatrixMultiplicationFloat16 --provider openrouter --model nvidia/nemotron-3-ultra-550b-a55b --reasoning-effort max --max-tool-rounds 0 --trace-path astra


exit 0

python agent.py RoPEFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra 
python agent.py RoPEFloat8Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra 



exit 0

ython agent.py ReLUFloat8Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra --max-budget 3.0
python agent.py GELUFloat8Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra --max-budget 3.0

exit 0

python agent.py RMSNormFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra --max-budget 5.0
python agent.py SoftmaxFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra --max-budget 5.0
python agent.py RMSNormFloat8Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra --max-budget 5.0
python agent.py SoftmaxFloat8Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra --max-budget 5.0
python agent.py ReductionSumFloat8Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra --max-budget 5.0
python agent.py GELUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra 
python agent.py GELUFloat8Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra 
python agent.py SiLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SiLUFloat8Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra 
python agent.py SwiGLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SwiGLUFloat8Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py ReLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra 
python agent.py ReLUFloat8Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra 
python agent.py GELUFloat8Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra 

python agent.py ReLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
#curl -fsSL https://vast.ai/install.sh | bash
#vastai stop instance $CONTAINER_ID
exit 0

python agent.py MatrixMultiplicationFloat16 --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model claude-opus-5 --max-budget 15.0 --provider anthropic
python agent.py MatrixMultiplicationFloat16 --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model claude-haiku-4-5-20251001 --max-budget 15.0 --provider anthropic
exit 0

python agent.py MatrixMultiplicationFloat16 --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra 

python agent.py MatrixMultiplicationFloat16 --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model claude-fable-5-1 --max-budget 15.0 --provider anthropic
python agent.py ReLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model claude-fable-5 --max-budget 15.0 --provider anthropic

python agent.py MatrixMultiplicationFloat16 --reasoning-effort max --max-tool-rounds 1 --trace-path astra --model claude-fable-5 --max-budget 2s5.0 --provider anthropic
exit 0

python agent.py GELUFloat16Kernel --reasoning-effort medium --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py Convolution2DFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SoftmaxFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
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
python agent.py RMSNormFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py ReductionSumFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SiLUFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py SwiGLUFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
python agent.py GELUFloat16Kernel --reasoning-effort max --max-tool-rounds 0 --trace-path astra --model gpt-6-astra
