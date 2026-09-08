# Get results from the various places
rsync -avP franc141@matx3:/matx/u/franc141/client/astra .
rsync -avP \
  -e "ssh -p 14744 -o ClearAllForwardings=yes" \
  root@208.64.254.185:/workspace/triton_ptx_client/astra \
  .

python plot_speedup_cost_frontier.py