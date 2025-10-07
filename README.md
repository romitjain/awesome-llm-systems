# Awesome llm systems [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository aims to consolidate resources for learning about systems for LLM. I have attempted to compile a list of resources (blogs/papers) that are essential for building a fundamental knowledge of the field. This is by no means exhaustive. The criteria for a resource to be in this list are:

1. It is simple (not necessarily easy!) to follow
2. It is fundamental to the domain of systems and LLM, i.e, it is either widely adopted or has a critical idea explored
3. It is good for someone starting in the area or someone with intermediate knowledge in the field

## Basics

- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
  - An introduction to batching in LLMs
- [GPU Glossary](https://modal.com/gpu-glossary)
  - A starting point for understanding about GPUs and terms used in GPU programming
- [Domain specific architectures for AI inference](https://fleetwood.dev/posts/domain-specific-architectures)
  - A primer on what a good GPU architecture looks like
- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
  - Derivation of Flash attention, starting from softmax
- [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)

### Parameter arithmetic

IMO, understanding parameter arithmetic is the key to performance optimization in LLMs.

- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
  - Understanding compute bound vs memory bound
- [How is LLaMa.cpp possible?](https://finbarr.ca/how-is-llama-cpp-possible/)
  - Real world example of what it means to be memory bound
- [LLM Inference Economics from First Principles](https://www.tensoreconomics.com/p/llm-inference-economics-from-first)
  - Finally merging parameter arithmetic with costs

## Quantization

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

## Throughput

- [Throughput is all you need](https://cmeraki.github.io/throughput-is-all-you-need.html)
  - A primer on how to think about throughput in LLM systems. Talks about continuous batching, paged attention and the basics of vLLM orchestrator
- [Throughput is Not All You Need: Maximizing Goodput in LLM Serving using Prefill-Decode Disaggregation](https://hao-ai-lab.github.io/blogs/distserve/)

## Training

- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
  - One of the best resource to understand distributed training
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Visualizing 6D Mesh Parallelism](https://main-horse.github.io/posts/visualizing-6d/)
- [1.5x faster MoE training with custom MXFP8 kernels](https://cursor.com/blog/kernels)

## Inference

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/)
- [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
- [One Kernel for All Your GPUs](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)
- [We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4)

## Kernels

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [Writing Speed-of-Light Flash Attention for 5090 in CUDA C++](https://gau-nernst.github.io/fa-5090/)
- [How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html)
- [Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
