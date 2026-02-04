# Awesome LLM systems [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository aims to consolidate resources for learning about systems for LLM. I have attempted to compile a list of resources (blogs/papers) that are essential for building a fundamental knowledge of the field. This is by no means exhaustive. The criteria for a resource to be in this list are:

1. It is simple (not necessarily easy!) to follow
2. It is fundamental to the domain of systems and LLM, i.e, it is either widely adopted or has a critical idea explored
3. It is good for someone starting in the area or someone with intermediate knowledge in the field

## Basics

- [What Every Developer Should Know About GPU Computing](https://blog.codingconfessions.com/p/gpu-computing)
- [GPU Glossary](https://modal.com/gpu-glossary)
  - A starting point for understanding about GPUs and terms used in GPU programming
- [Domain specific architectures for AI inference](https://fleetwood.dev/posts/domain-specific-architectures)
  - A primer on what a good GPU architecture looks like
- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
  - Derivation of Flash attention, starting from softmax
- [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [PyTorch and CPU-GPU Synchronizations](https://tomasruizt.github.io/posts/08_cpu_gpu_synchronization/)

IMO, understanding parameter arithmetic is the key to performance optimization in LLMs.

- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
  - Understanding compute bound vs memory bound
- [How is LLaMa.cpp possible?](https://finbarr.ca/how-is-llama-cpp-possible/)
  - Real world example of what it means to be memory bound
- [LLM Inference Economics from First Principles](https://www.tensoreconomics.com/p/llm-inference-economics-from-first)
  - Finally merging parameter arithmetic with costs

## Architecture

- [TransMLA: MLA Is All You Need](https://arxiv.org/abs/2502.07864)
- [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms)

## Quantization

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- [SmoothQuant](https://www.youtube.com/watch?v=U0yvqjdMfr0)

## Training

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
  - One of the best resource to understand distributed training
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Visualizing 6D Mesh Parallelism](https://main-horse.github.io/posts/visualizing-6d/)
- [1.5x faster MoE training with custom MXFP8 kernels](https://cursor.com/blog/kernels)
- [Accelerate ND-Parallel](https://huggingface.co/blog/accelerate-nd-parallel)

## Inference

- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
  - An introduction to batching in LLMs
- [Continuous batching from first principles](https://huggingface.co/blog/continuous_batching)
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai/)
- [Throughput is all you need](https://cmeraki.github.io/throughput-is-all-you-need.html)
  - A primer on how to think about throughput in LLM systems. Talks about continuous batching, paged attention and the basics of vLLM orchestrator
- [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/)
- [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
- [Throughput is Not All You Need: Maximizing Goodput in LLM Serving using Prefill-Decode Disaggregation](https://hao-ai-lab.github.io/blogs/distserve/)
- [One Kernel for All Your GPUs](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/pdf/2407.00079)
- [Inside Kaiju](https://blog.character.ai/technical/inside-kaiju-building-conversational-models-at-scale/)

## Kernels

- [Tiled Matrix Multiplication](https://penny-xu.github.io/blog/tiled-matrix-multiplication)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [Writing Speed-of-Light Flash Attention for 5090 in CUDA C++](https://gau-nernst.github.io/fa-5090/)
- [How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html)
- [GPUs Go Brrr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
- [Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)
- [Matrix Multiplication on Blackwell: Part 1 - Introduction](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)
- [Dissecting FlashInfer - A Systems Perspective on High-Performance LLM Inference](https://ydnyshhh.github.io/posts/flash_infer/)
- [Notes About Nvidia GPU Shared Memory Banks](https://feldmann.nyc/blog/smem-microbenchmarks#/)
- [Chasing 6+ TB/s: an MXFP8 quantizer on Blackwell](https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/#/)
    - Notes on packing quantization scales in a format required by downstream GEMM

## Communication

- [GPU networking basics](https://www.chipstrat.com/p/gpu-networking-basics-part-1)
- [A Beginner's Guide to Interconnects in AI Datacenters](https://substack.com/home/post/p-173009699)
- [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://www.alphaxiv.org/abs/2507.04786)

## Lecture series

- [Faster LLMs](https://faster-llms.vercel.app/)
    - Has multiple lectures from industry leaders on topics around serving LLMs as applications and how they are different from traditional ML models and regular web services

## Misc

- [Democratizing AI Compute](https://www.modular.com/democratizing-ai-compute)
- [We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4)

## Labs

List of labs working on LLM systems

- [Dao AI Lab](https://github.com/Dao-AILab)
- [Hazy Research](https://hazyresearch.stanford.edu/)
- [Alistarh Group](https://ista.ac.at/en/research/alistarh-group/)
- [HaoAI Lab](https://hao-ai-lab.github.io/)
- [HanLab](https://hanlab.mit.edu/)
- [LMSys Org](https://lmsys.org/)
- [SkyRL](https://sky.cs.berkeley.edu/project/skyrl/)
- [MaDSys](https://madsys.cs.tsinghua.edu.cn/)
