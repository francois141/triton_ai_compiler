# Activate the environment
source .venv/bin/activate

#python agent.py Convolution2DFloat16Kernel --reasoning-effort max --max-tool-rounds 1 --trace-path paper_results --start-json paper_results/260826070938_Convolution2DFloat16Kernel_gpt-5.6-sol_max/iteration_001_candidate_02_try_00_candidate_speedup_vs_triton_0.9908x.json


python agent.py FlashAttentionFloat16Kernel --reasoning-effort max --max-tool-rounds 3 --trace-path paper_results --start-json paper_results/260826122348_FlashAttentionFloat16Kernel_gpt-5.6-sol_max/iteration_001_candidate_02_try_00_candidate_speedup_vs_triton_0.9412x.json

exit 0

python agent.py MatrixVectorMultiplicationFloat16Kernel --reasoning-effort max --max-tool-rounds 3 --trace-path paper_results
exit 0


python agent.py FusedGEMMAddGELUFloat16Kernel --reasoning-effort max --max-tool-rounds 3 --trace-path paper_results
python agent.py FusedGEMMAddGELUKernel --reReLUFloat16Kernelasoning-effort max --max-tool-rounds 3 --trace-path paper_results
exit 0

python agent.py ReductionSumKernel --reasoning-effort medium --max-tool-rounds 2 --trace-path paper_results
python agent.py ReductionSumFloat16Kernel --reasoning-effort medium --max-tool-rounds 2 --trace-path paper_results
python agent.py MatrixVectorMultiplicationFloat16Kernel --reasoning-effort max --max-tool-rounds 1 --trace-path paper_results
exit 0
python agent.py RMSNormFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py RMSNormKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py SoftmaxKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py SoftmaxFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
exit 0

python agent.py MatrixVectorMultiplicationFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py RMSNormFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py ReLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py ReductionSumFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py SiLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results

python agent.py SwiGLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results

python agent.py DotProductAttentionFloat16Kernel --reasoning-effort max --max-tool-rounds 2 --trace-path paper_results
python agent.py Convolution2DFloat16Kernel --reasoning-effort max --max-tool-rounds 2 --trace-path paper_results
python agent.py SoftmaxFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results

exit 0

python agent.py FusedGEMMAddGELUKernel --reasoning-effort max --max-tool-rounds 0 --trace-path paper_results
python agent.py DotProductAttentionKernel --reasoning-effort max --max-tool-rounds 0 --trace-path paper_results
python agent.py MatrixMultiplicationKernel --reasoning-effort max --max-tool-rounds 0 --trace-path paper_results
python agent.py Convolution2DKernel --reasoning-effort max --max-tool-rounds 0 --trace-path paper_results

python agent.py SoftmaxKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py ReductionSumKernel --reasoning-effort medium --max-tool-rounds 2 --trace-path paper_results

python agent.py GELUFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py MatrixVectorMultiplicationFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py RMSNormFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py ReLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py ReductionSumFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py SiLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results

python agent.py SwiGLUFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results

python agent.py DotProductAttentionFloat16Kernel --reasoning-effort max --max-tool-rounds 2 --trace-path paper_results
python agent.py Convolution2DFloat16Kernel --reasoning-effort max --max-tool-rounds 2 --trace-path paper_results
python agent.py SoftmaxFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results


exit 0

python agent.py SoftmaxFloat16Kernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results

# Run level1 experiments for simple values
python agent.py GELUKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py MatrixVectorMultiplicationKernel --reasoning-effort medium --max-tool-rounds 3 --trace-path paper_results
python agent.py RMSNormKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py ReLUKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py ReductionSumKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py SiLUKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py SoftmaxKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results
python agent.py SwiGLUKernel --reasoning-effort medium --max-tool-rounds 1 --trace-path paper_results

# Run level2 experiments for simple values


# tmux new -s dev  
# tmux attach -t dev  
# tmux kill-session -t dev
