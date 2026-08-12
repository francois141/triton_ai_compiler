source /matx/u/franc141/client/.venv/bin/activate

NCU_PATH=/usr/local/cuda-13.3/nsight-compute-2026.2.0/ncu python agent.py MatrixMultiplicationFloat16  --max-tool-rounds 1 --max-repair-attempts 0 --start-json output_traces/260811142249_MatrixMultiplicationFloat16_gpt-5.6-sol_max/iteration_000_candidate_00_try_00_initial_candidate_speedup_vs_triton_0.9438x.json
NCU_PATH=/usr/local/cuda-13.3/nsight-compute-2026.2.0/ncu python agent.py MatrixMultiplicationFloat16  --max-tool-rounds 0 --max-repair-attempts 0
NCU_PATH=/usr/local/cuda-13.3/nsight-compute-2026.2.0/ncu python agent.py MatrixMultiplicationFloat16  --max-tool-rounds 0 --max-repair-attempts 0
