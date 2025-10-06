# runpod/pytorch

**PyTorch-optimized images for deep learning workflows.**

Built on our base images, these PyTorch containers come pre-configured with specific PyTorch versions and CUDA support, eliminating the guesswork of compatibility and setup time. Whether you're training neural networks, running inference, or developing ML models, these images provide the exact PyTorch environment you need.

**What makes these optimized:**
- **Precision-matched versions** — Each image contains a specific PyTorch version paired with the optimal CUDA toolkit
- **Zero configuration** — PyTorch is installed and ready to import, no pip installs or environment setup required
- **GPU-accelerated** — All images include CUDA support for immediate GPU acceleration
- **Production-ready** — Built on our stable base images with all the development tools and workspace setup you need

**Choose your combination:**
- **PyTorch versions:** 2.4.0 through 2.7.1
- **CUDA versions:** 12.4.1 through 12.9.0
- **Ubuntu versions:** 20.04, 22.04, and 24.04

Perfect for research, development, and production PyTorch workloads without the setup overhead.

Please also see [../base/README.md](../base/README.md)

<div class="base-images">

## Generated PyTorch Images

### CUDA 12.4.1:
- Torch 2.4.0:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch240-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1241-torch240`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch240-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1241-torch240`
- Torch 2.4.1:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch241-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1241-torch241`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch241-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1241-torch241`
- Torch 2.5.0:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch250-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1241-torch250`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch250-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1241-torch250`
- Torch 2.5.1:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch251-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1241-torch251`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch251-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1241-torch251`
- Torch 2.6.0:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch260-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1241-torch260`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1241-torch260-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1241-torch260`

### CUDA 12.5.1:
- Torch 2.5.1:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1251-torch251-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1251-torch251`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1251-torch251-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1251-torch251`
- Torch 2.6.0:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1251-torch260-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1251-torch260`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1251-torch260-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1251-torch260`

### CUDA 12.6.3:
- Torch 2.6.0:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1263-torch260-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1263-torch260`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1263-torch260-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1263-torch260`
  - Ubuntu 24.04:
    - `runpod/pytorch:0.7.0-dev-cu1263-torch260-ubuntu2404`
    - `runpod/pytorch:0.7.0-dev-ubuntu2404-cu1263-torch260`
- Torch 2.7.1:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1263-torch271-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1263-torch271`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1263-torch271-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1263-torch271`
  - Ubuntu 24.04:
    - `runpod/pytorch:0.7.0-dev-cu1263-torch271-ubuntu2404`
    - `runpod/pytorch:0.7.0-dev-ubuntu2404-cu1263-torch271`

### CUDA 12.8.1:
- Torch 2.6.0:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1281-torch260-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1281-torch260`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1281-torch260-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1281-torch260`
  - Ubuntu 24.04:
    - `runpod/pytorch:0.7.0-dev-cu1281-torch260-ubuntu2404`
    - `runpod/pytorch:0.7.0-dev-ubuntu2404-cu1281-torch260`
- Torch 2.7.1:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1281-torch271`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1281-torch271`
  - Ubuntu 24.04:
    - `runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2404`
    - `runpod/pytorch:0.7.0-dev-ubuntu2404-cu1281-torch271`

### CUDA 12.9.0:
- Torch 2.6.0:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1290-torch260-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1290-torch260`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1290-torch260-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1290-torch260`
  - Ubuntu 24.04:
    - `runpod/pytorch:0.7.0-dev-cu1290-torch260-ubuntu2404`
    - `runpod/pytorch:0.7.0-dev-ubuntu2404-cu1290-torch260`
- Torch 2.7.1:
  - Ubuntu 20.04:
    - `runpod/pytorch:0.7.0-dev-cu1290-torch271-ubuntu2004`
    - `runpod/pytorch:0.7.0-dev-ubuntu2004-cu1290-torch271`
  - Ubuntu 22.04:
    - `runpod/pytorch:0.7.0-dev-cu1290-torch271-ubuntu2204`
    - `runpod/pytorch:0.7.0-dev-ubuntu2204-cu1290-torch271`
  - Ubuntu 24.04:
    - `runpod/pytorch:0.7.0-dev-cu1290-torch271-ubuntu2404`
    - `runpod/pytorch:0.7.0-dev-ubuntu2404-cu1290-torch271`

</div>

---

## Dockerfile

```
ARG BASE_IMAGE=non-existing
FROM ${BASE_IMAGE}

ARG WHEEL_SRC
ARG TORCH

RUN python3.10 -m pip install --resume-retries 3 --no-cache-dir --upgrade ${TORCH} --index-url https://download.pytorch.org/whl/cu${WHEEL_SRC}
```

## docker-bake.hcl

```
variable "TORCH_META" {
  default = {
    "2.7.1" = {
      torchvision = "0.22.1"
    }
    "2.6.0" = {
      torchvision = "0.21.0"
    }
    "2.5.1" = {
      torchvision = "0.20.1"
    }
    "2.5.0" = {
      torchvision = "0.20.0"
    }
    "2.4.1" = {
      torchvision = "0.19.1"
    }
    "2.4.0" = {
      torchvision = "0.19.0"
    }
  }
}

# We need to grab the most compatible wheel for a given CUDA version and Torch version pair
# At times, this requires grabbing a wheel built for a different CUDA version.
variable "CUDA_TORCH_COMBINATIONS" {
  default = [ 
    { cuda_version = "12.4.1", torch = "2.4.0", whl_src = "121" },
    { cuda_version = "12.4.1", torch = "2.4.1", whl_src = "121" },
    { cuda_version = "12.4.1", torch = "2.5.0", whl_src = "124" },
    { cuda_version = "12.4.1", torch = "2.5.1", whl_src = "124" },
    { cuda_version = "12.4.1", torch = "2.6.0", whl_src = "124" },
    
    { cuda_version = "12.5.1", torch = "2.5.1", whl_src = "121" },
    { cuda_version = "12.5.1", torch = "2.6.0", whl_src = "124" },
    
    { cuda_version = "12.6.3", torch = "2.6.0", whl_src = "126" },
    { cuda_version = "12.6.3", torch = "2.7.1", whl_src = "126" },
    
    { cuda_version = "12.8.1", torch = "2.6.0", whl_src = "126" },
    { cuda_version = "12.8.1", torch = "2.7.1", whl_src = "128" },
    
    { cuda_version = "12.9.0", torch = "2.6.0", whl_src = "126" },
    { cuda_version = "12.9.0", torch = "2.7.1", whl_src = "128" },
  ]
}

variable "COMPATIBLE_BUILDS" {
  default = flatten([
    for combo in CUDA_TORCH_COMBINATIONS : [
      for cuda in CUDA_VERSIONS : [
        for ubuntu in UBUNTU_VERSIONS : {
          ubuntu_version = ubuntu.version
          ubuntu_name    = ubuntu.name
          ubuntu_alias   = ubuntu.alias
          cuda_version   = cuda.version
          cuda_code      = replace(cuda.version, ".", "")
          wheel_src      = combo.whl_src
          torch          = combo.torch
          torch_code     = replace(combo.torch, ".", "")
          torch_vision   = TORCH_META[combo.torch].torchvision
        } if cuda.version == combo.cuda_version && contains(cuda.ubuntu, ubuntu.version)
      ]
    ]
  ])
}

group "dev" {
  targets = ["pytorch-ubuntu2204-cu1241-torch260"]
}

group "default" {
  targets = [
    for build in COMPATIBLE_BUILDS:
      "pytorch-${build.ubuntu_name}-cu${replace(build.cuda_version, ".", "")}-torch${build.torch_code}"
  ]
}

target "pytorch-base" {
  context = "official-templates/pytorch"
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
}

target "pytorch-matrix" {
  matrix = {
    build = COMPATIBLE_BUILDS
  }
  
  name = "pytorch-${build.ubuntu_name}-cu${build.cuda_code}-torch${build.torch_code}"
  
  inherits = ["pytorch-base"]
  
  args = {
    BASE_IMAGE = "runpod/base:${RELEASE_VERSION}-${build.ubuntu_name}-cuda${build.cuda_code}"
    WHEEL_SRC = build.wheel_src
    TORCH = "torch==${build.torch} torchvision==${build.torch_vision} torchaudio==${build.torch}"
  }
  
  tags = [
    "runpod/pytorch:${RELEASE_VERSION}-${build.ubuntu_name}-cu${build.cuda_code}-torch${build.torch_code}",
    "runpod/pytorch:${RELEASE_VERSION}-cu${build.cuda_code}-torch${build.torch_code}-${build.ubuntu_name}",
  ]
}
```