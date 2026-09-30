LLM2CLIP CLIP sources vendored from microsoft/LLM2CLIP-Openai-L-14-336,
revision 92512331f393a003c3d98404677f991c188162c9.

The unused CLIPOnnxConfig and its imports are omitted because Transformers 5
removed transformers.onnx. Weight initialization uses Transformers 5 initialization helpers to preserve loaded
weights and reconstruct non-persistent position IDs after meta-device loading.
The forward computations are unchanged.
