#!/bin/bash
# GLM-5.2 chat server. Model loads ONCE (~12 min) then stays resident, so every
# message after that costs only generation time instead of a fresh 12-min mmap.
cd ~/glm52
exec env GGML_TRUNK_LOOKAHEAD=64 nice -n 5 ~/ik_llama.cpp/build/bin/llama-server \
  -m model/glm52-1.63bpw.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 4096 -t 16 --prefetch-experts --prefetch-experts-threads 8 \
  --temp 1.0 --top-p 0.95 --repeat-penalty 1.05 \
  --metrics --jinja \
  >> ~/glm52/out/server.log 2>&1
