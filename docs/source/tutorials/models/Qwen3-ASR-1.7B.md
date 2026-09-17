# Qwen3-ASR-1.7B

## Introduction

The released Qwen3-ASR-1.7B is a lightweight, high-performance automatic speech recognition (ASR) model developed by the Qwen Team. It delivers industry-leading recognition accuracy across Chinese/English multi-scene speech, Chinese dialects, multilingual and singing voice scenarios, with native support for long audio and streaming inference, and deep optimization for Ascend NPU hardware.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

## Environment Preparation

### Model Weight

`Qwen3-ASR-1.7B`(BF16 version): requires 1 Ascend 910B (with 1 x 64G NPUs).

`Qwen3-ASR-1.7B`(BF16 version): requires 1 Ascend 310P (with 1 x 48G NPUs).
[Download model weight](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### Installation

<<<<<<< HEAD
|  Weight Version    | Hardware Requirements                                            | Download Links |
|--------------------|------------------------------------------------------------------|----------------|
| BF16               | one Ascend Atlas 800I A2 64 GB NPU or one Ascend Atlas 300I DUO 48 GB NPU | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) |

Download the weights to a directory that is accessible from the deployment environment. For multi-node deployments, use a shared directory; for example, `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

## 4 Installation

### 4.1 Docker Image Installation

Use the vLLM-Ascend Docker image that corresponds to your hardware. Replace the model-weight mount with the path used in your environment.

=== "Atlas A2 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net host \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it -d $IMAGE bash
    ```

=== "Atlas 300I DUO"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p

    docker run --rm \
        --name vllm-ascend \
        --shm-size=10g \
        --net host \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it -d $IMAGE bash
    ```

Verify that the container is running and that the installed package version matches the image tag:

```bash
docker ps --filter name=vllm-ascend
pip show vllm vllm-ascend
```

Expected result: `docker ps` lists the container with status `Up`, and `pip show` displays version information for both packages.

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM-Ascend following the [Installation Guide](../../getting_started/installation.md).

!!! note

    For Atlas 300I DUO, source installation may pull in `triton` and `triton-ascend`. Uninstall them before running vLLM-Ascend on Atlas 300I DUO:

    ```bash
    pip uninstall -y triton-ascend triton
    ```

To verify the source installation:

```bash
pip show vllm-ascend
```

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment runs both audio prefill and decoding on one NPU, making it suitable for development, testing, and small-scale ASR services. Replace `your_model_path` with the local model directory, or use `Qwen/Qwen3-ASR-1.7B` to download the model through the configured model hub.

=== "Atlas A2 inference products"

    ```shell
    # Ensure the model path matches the directory recorded during download
    vllm serve your_model_path \
      --served-model-name qwen3-asr \
      --tensor-parallel-size 1 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.9 \
      --enforce-eager \
      --port 8000
    ```

=== "Atlas 300I DUO"

    ```shell
    # Ensure the model path matches the directory recorded during download
    vllm serve your_model_path \
      --served-model-name qwen3-asr \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.9 \
      --dtype float16 \
      --max-model-len 4096 \
      --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false,"enable_npu_graph_ex":false}}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,4]}' \
      --port 8000
    ```

    !!! note

        - `--tensor-parallel-size 1` uses one NPU. Increase it only after confirming that the hardware and deployment topology support the chosen parallel configuration.
        - `--max-model-len 4096` limits the maximum sequence length. On Atlas 300I DUO, always specify a conservative value explicitly; automatic detection can allocate an oversized attention mask and cause an out-of-memory error.
        - `--gpu-memory-utilization 0.9` sets the fraction of device memory available to the vLLM executor. Lower this value if other workloads share the NPU.
        - `--enforce-eager` disables graph execution. It is used in the Atlas 300I A2 2UP example for compatibility.

When the service starts successfully, the log contains `Application startup complete`. If startup fails, see the [Public FAQs](../../faqs.md).

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
=======
`Qwen3-ASR-1.7B` is supported in `vllm-ascend`.

You can use our official docker image to run `Qwen3-ASR-1.7B` directly.

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
  --name vllm-ascend \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v /data/vllm-workspace/models:/data/vllm-workspace/models \
  -p 8000:8000 \
  -it $IMAGE bash
```

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

### Atlas 300I A2 2UP

```shell
vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000
```

### Atlas 300I DUO

```shell
vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --tensor-parallel-size 1 \
  --gpu_memory_utilization 0.9 \
  --dtype float16 \
  --max_model_len 4096 \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,4]}' \
  --port 8000
```

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://localhost:8000/v1/chat/completions
    -H "Content-Type: application/json"
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
    -d '{
    "messages": [
    {"role": "user", "content": [
        {"type": "audio_url",
        "audio_url":
        {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
    ]}
    ]
}'
```

## Accuracy Evaluation

After all samples were processed, transcription quality was measured using:

- WER (Word Error Rate) for word-level recognition accuracy
- CER (Character Error Rate) for character-level recognition accuracy

### Remarks

This result reflects end-to-end serving performance, including audio preprocessing, request construction, API communication, inference, and response parsing. Actual performance may vary depending on hardware, concurrency, audio length, and deployment configuration.

<<<<<<< HEAD
Actual performance varies with hardware, audio duration, concurrency, and deployment configuration. Evaluate short audio, long audio, and concurrent requests separately before selecting a production configuration.

## 9 Performance Tuning

The following settings are starting points rather than globally optimal configurations. Tune them according to audio duration, concurrency, latency requirements, and available NPU memory.

| Scenario | Recommended Starting Point | Key Considerations |
| --- | --- | --- |
| Low latency | `--tensor-parallel-size 1`, `--max-model-len 4096` | Use short audio inputs and avoid sharing the NPU with other workloads. |
| High throughput | Increase request concurrency after establishing the latency baseline | Monitor NPU memory and end-to-end latency; do not use synthetic text-only requests as a proxy for ASR traffic. |
| Long audio | Increase `--max-model-len` only as required | On Atlas 300I DUO, keep the value conservative because attention-mask memory grows with the configured maximum length. |

For general parameter tuning, refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

## 10 FAQ

For common environment, installation, and general parameter issues, see the [Public FAQs](../../faqs.md). This section covers model- and hardware-specific guidance.

### Atlas 300I DUO runs out of memory during startup

**Symptom:** The server fails with an out-of-memory error while initializing attention.

**Cause:** On Atlas 300I DUO, an automatically detected large context length can create a full causal attention mask whose memory consumption grows quadratically with `max_model_len`.

**Solution:** Always set `--max-model-len` explicitly to a conservative value, such as `4096`, and increase it only after verifying available NPU memory.
=======
Further benchmarking is recommended for latency distribution, concurrent throughput, long-audio scenarios, and system resource utilization.
>>>>>>> 36d8a3435 ([Feature][KV Offload] Add offload capability abstraction with warmup proxy and finalization fixes)
