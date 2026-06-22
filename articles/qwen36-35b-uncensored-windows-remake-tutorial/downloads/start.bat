@echo off
chcp 65001 >nul
cd /d "%~dp0"

llama-server.exe ^
-m "models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf" ^
--mmproj "models\mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf" ^
--mmproj-offload ^
--image-min-tokens 1024 ^
--image-max-tokens 1024 ^
-ngl all ^
--fit off ^
--flash-attn on ^
--cache-type-k q8_0 ^
--cache-type-v q8_0 ^
-c 8192 ^
-n 4096 ^
-np 1 ^
--host 127.0.0.1 ^
--port 8080

pause
