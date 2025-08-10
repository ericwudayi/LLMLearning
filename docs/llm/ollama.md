

# **An In-Depth Analysis of Ollama's Hybrid CPU/GPU Inference Strategy**

## **Section 1: Foundational Architecture: The Role of GGML and the GGUF Format**

To comprehend the strategy by which Ollama orchestrates CPU and GPU resources, it is essential to first analyze its foundational architecture. Ollama's approach to performance is not rooted in a proprietary, from-scratch inference engine. Instead, its power and flexibility are derived from its strategic integration with the GGML tensor library and the GGUF model format. Ollama's primary innovation lies in creating a highly accessible and user-friendly ecosystem around these powerful, low-level technologies, effectively abstracting their complexity from the end-user.

### **1.1 Ollama's Core Dependency: The GGML Library**

The core of Ollama's computational capability is the GGML library. While early perceptions might have labeled Ollama as a simple wrapper around the popular llama.cpp project, the relationship is more nuanced. Ollama has evolved its own sophisticated server, API, and management logic, but it continues to rely on GGML for the fundamental operations of model inference.1 GGML is a C-based tensor library specifically engineered to run large language models (LLMs) efficiently on consumer-grade hardware. Its design philosophy prioritizes CPU-based inference as a primary, viable execution path while incorporating GPU offloading as a powerful mechanism for acceleration.2

This reliance on GGML is not merely incidental; it is the cornerstone of Ollama's entire performance strategy. The features that define Ollama's hybrid execution model—such as partial model offloading and support for quantized models—are inherited directly from the capabilities of GGML. Examination of the Ollama source code confirms this deep integration. For instance, the ggml.go file within the repository contains explicit references to GGML-specific file magic constants, such as FILE\_MAGIC\_GGML and FILE\_MAGIC\_GGJT, which are used to identify and parse different GGML-based model formats.4 This demonstrates that Ollama interacts with these models at a fundamental level, leveraging the structures defined by the GGML ecosystem.

### **1.2 The GGUF Format: The Blueprint for Layer Offloading**

The practical implementation of hybrid CPU/GPU execution is enabled by the GGUF (GGML Universal Format). Ollama's documentation and command-line interface explicitly highlight its support for importing models in the GGUF format through the use of a Modelfile.5 GGUF is a self-contained, single-file format designed for portability and ease of use. It bundles not only the model's weights but also crucial metadata, including the model's architecture, required context length, and a detailed manifest of its constituent tensors.6

The most critical aspect of the GGUF format for this analysis is its organization of the model into a discrete sequence of layers or tensors. This layered structure is the technical prerequisite for partial offloading. It allows an inference engine like Ollama to read the model file and load it into memory chunk by chunk. During this loading process, the engine can make a strategic decision for each individual layer: place it in the GPU's high-speed VRAM for maximum performance or consign it to the system's main RAM for processing by the CPU. This layer-by-layer decision-making process is the fundamental mechanism that underpins Ollama's ability to run models that are larger than the available VRAM on a given GPU.

### **1.3 Quantization: A Synergistic Technology**

Working in concert with the GGUF format and layer offloading is the technique of quantization. Quantization is a model compression method that reduces the memory footprint of an LLM by storing its weights using lower-precision numerical formats, such as 4-bit or 8-bit integers, instead of the standard 16-bit or 32-bit floating-point numbers.2 This reduction in size has a direct and profound impact on performance when using consumer hardware.

The synergy between quantization and layer offloading is a critical component of Ollama's strategy. By using a quantized version of a model, its overall size is significantly reduced. This allows a greater number of layers—or in many cases, the entire model—to fit within the limited VRAM of a typical consumer GPU.2 Since operations performed on the GPU are orders of magnitude faster than those on the CPU for LLM workloads, fitting more of the model into VRAM leads to a dramatic increase in token generation speed.2 Ollama's extensive model library includes numerous pre-quantized models (e.g., with tags like

q4\_0, q5\_K\_S, q8\_0), a direct result of its GGML foundation and a key enabler for its performance on mainstream hardware.5 The ability to run a 7B parameter model smoothly on a GPU with 8 GB of VRAM is a testament to the combined power of quantization and GPU offloading.2

The decision to build upon GGML and its associated technologies reveals the core of Ollama's strategic vision. Rather than reinventing the highly complex and specialized field of low-level inference optimization, the Ollama project has focused on encapsulation and user experience. It takes the raw, formidable power of the GGML library and packages it within a polished, accessible interface. This allows developers and hobbyists to leverage advanced features like hybrid execution through simple commands like ollama run and a clean REST API, without needing to manually compile llama.cpp or grapple with its complex command-line arguments.1 This strategic choice is arguably the single most significant factor behind Ollama's widespread and rapid adoption. Consequently, the performance characteristics, capabilities, and even the limitations of Ollama are inextricably linked to the ongoing development of the underlying GGML/

llama.cpp ecosystem.

## **Section 2: The Core Strategy: Partial Model Offloading on a Single GPU**

The central pillar of Ollama's performance enhancement strategy is the mechanism of partial model offloading. This technique involves intelligently distributing the layers of a large language model between a computer's high-speed GPU Video RAM (VRAM) and its larger, but slower, main system RAM. This hybrid approach allows users to run models that would otherwise be too large to fit entirely within the VRAM of their GPU, striking a pragmatic balance between performance and hardware limitations.

### **2.1 The VRAM/RAM Hierarchy**

The rationale for this strategy is rooted in the fundamental differences between VRAM and system RAM. GPU VRAM is a type of memory designed for extremely high-bandwidth access, optimized for the massively parallel computations that characterize both graphics rendering and the matrix multiplications at the heart of LLM inference. However, this high performance comes at the cost of limited capacity, with consumer GPUs typically offering between 8 GB and 24 GB of VRAM.2 In contrast, system RAM is available in much larger quantities (32 GB, 64 GB, or more is common) but operates at significantly lower bandwidth, making it a bottleneck for the parallel processing tasks at which GPUs excel.2

The objective of Ollama's offloading strategy is therefore straightforward: to load as many of the model's most computationally intensive layers as possible into the fastest available memory—the GPU's VRAM. Each transformer layer successfully offloaded to the GPU provides a substantial acceleration in inference speed, drastically reducing the time it takes to generate each token of the response.2

### **2.2 The Mechanics of Layer Offloading**

When a user initiates a command like ollama run \<model\>, a carefully orchestrated sequence of events unfolds to implement this hybrid strategy:

1. **Hardware Discovery:** Ollama begins by querying the system to identify the available computational hardware. It detects the CPU, the type of GPU (NVIDIA, AMD, Apple Metal), and, most critically, the amount of available VRAM.10  
2. **Model Inspection:** It then reads the metadata from the model's GGUF file to understand its architecture, its total number of layers, and the size of each layer.  
3. **Sequential Layer Loading:** Ollama proceeds to load the model's layers one by one. The most performance-critical components, the transformer blocks that perform the vast majority of the matrix multiplication operations, are prioritized for offloading to the GPU.  
4. **VRAM Management:** With each layer it considers for offloading, Ollama checks against the available VRAM. As long as there is sufficient space, the layer is loaded onto the GPU. This process continues until the VRAM is nearly full.  
5. **CPU Fallback:** Once the available VRAM is exhausted, any remaining layers of the model are loaded into the system's main RAM. These layers will be processed by the CPU during inference.12

This process results in a hybrid execution model. During the generation of a single token, the inference process will flow through all layers of the model. The parts of the model residing in VRAM are processed rapidly by the GPU, while the parts in RAM are processed more slowly by the CPU. The final performance is a weighted average of these two speeds. In addition to the model weights, the KV cache—a memory structure that stores the attention state of the conversation to avoid re-computation—also consumes a significant and variable amount of VRAM, which must be factored into the memory allocation calculations.13

### **2.3 Automatic Layer Calculation: "It Just Works"... Sometimes**

By default, Ollama is designed to automate this entire process. It attempts to calculate the optimal number of layers to offload to the GPU based on its detection of available VRAM, aiming to maximize GPU utilization without triggering an Out-of-Memory (OOM) error.12 For many users and common models, this "just works" approach provides a good balance of performance and ease of use.

However, a substantial body of user-reported evidence from GitHub issues and community forums indicates that this automatic calculation is frequently conservative and suboptimal.8 In many cases, Ollama underestimates how many layers can safely fit into VRAM, leaving valuable GPU memory unused and, consequently, leaving performance on the table. This is a critical point of friction for users seeking to maximize the performance of their hardware.

Ollama's own logs provide clear evidence of this behavior. A user might have a GPU with 24 GB of VRAM and a model they know can fit entirely, yet the logs will show a message such as offloaded 49/63 layers to GPU, indicating that Ollama has conservatively decided to keep 14 layers on the much slower CPU, even though space was available.8 This discrepancy between potential and actual utilization is the primary reason why advanced users must move beyond the automatic configuration and engage with manual tuning parameters. This reveals a fundamental tension in the Ollama experience: its core value proposition of simplicity can be at odds with the goal of achieving maximum performance, compelling users to delve into the very complexities the tool was designed to abstract away. The performance strategy for a user thus becomes an active optimization problem: selecting the highest-quality quantized model that can be made to fit almost entirely within their VRAM, as a 7B model fully resident on an 8 GB GPU will invariably outperform a larger 13B model that is split heavily between the GPU and CPU.

## **Section 3: User Control and Configuration: Mastering the num\_gpu Parameter**

While Ollama strives for automated, "zero-config" operation, its true power for advanced users lies in the ability to override the default behavior. The primary mechanism for this manual control is the num\_gpu parameter. Understanding and correctly utilizing this parameter is the single most important skill for tuning Ollama's performance and resolving issues of underutilized GPU resources. However, its function is widely misunderstood, and its documentation within the official Ollama project is conspicuously absent, creating a knowledge gap that users must often bridge through community-driven discovery.

### **3.1 Demystifying num\_gpu: Layers, Not Devices**

The most critical fact about the num\_gpu parameter is that it specifies the **number of model layers** to offload to the GPU(s).16 It does not, despite its name, refer to the number of physical GPU devices installed in the system.

This naming choice has been a significant source of confusion. A detailed user report on a public forum illustrates this pitfall perfectly: a user with a powerful dual RTX 3090 setup initially set num\_gpu to 2, logically assuming it corresponded to their two physical GPUs. The result was abysmal performance, as only the first two layers of their model were being offloaded to the GPUs, with the vast majority being processed by the much slower CPU. It was only through extensive trial and error that they discovered the parameter's true function was to control layer count. By setting num\_gpu to 64 (the number of layers in their model), performance increased dramatically as the entire model was loaded into VRAM.16

Documentation found in forks of the Ollama repository and older documentation versions explicitly confirms this definition: num\_gpu. The number of layers to send to the GPU(s).17 Power users often employ a simple trick: setting

num\_gpu to a very high number (e.g., 100 or 999\) to effectively tell Ollama to offload as many layers as can possibly fit into VRAM, thereby overriding any conservative automatic calculations.14

### **3.2 Methods of Configuration**

Users have three primary methods to set the num\_gpu parameter and other performance-tuning options:

1. **Modelfile:** The most persistent and recommended method for creating a model tailored to specific hardware is to use a Modelfile. By including the instruction PARAMETER num\_gpu \<value\>, a user can create a new custom model tag that will always use the specified number of layers when run.8 This is ideal for creating a permanent, optimized version of a public model.  
2. **API Options:** For dynamic, on-the-fly tuning, the Ollama REST API is the preferred method. Both the /api/generate and /api/chat endpoints accept a JSON body containing an options object. This object can include a num\_gpu key to override the model's default for a single request, allowing for programmatic control without modifying the underlying model file.19  
3. **Interactive CLI Commands:** When using the Ollama command-line interface interactively (ollama run \<model\>), the setting can be adjusted for the current session using the command /set parameter num\_gpu \<value\>.20 This is useful for quick experiments and testing.

### **3.3 The Case of the Missing Documentation**

A perplexing issue for users attempting to master these controls is the apparent removal of the num\_gpu parameter from Ollama's official documentation. A Reddit thread initiated by a knowledgeable user directly questions this absence, with another user confirming, "They removed it".21 This stands in stark contrast to the parameter's continued functionality and its documented presence in the API specification and older or forked documentation.17 This suggests either a documentation oversight or a deliberate product decision to de-emphasize advanced, potentially confusing parameters to streamline the experience for new users. Regardless of the reason, the effect is that the primary tool for overcoming Ollama's most common performance bottleneck (suboptimal automatic layer allocation) is rendered an "expert secret" discoverable only through community channels. This reality highlights the tension between Ollama's goal of simplicity and the needs of power users who require granular control to extract maximum value from their hardware.

To demystify this landscape, the following table centralizes the key parameters for managing Ollama's hybrid compute behavior.

| Parameter | Scope | Description | Example |
| :---- | :---- | :---- | :---- |
| num\_gpu | Modelfile, API | Sets the number of model layers to offload to the GPU(s). A value of \-1 or a very large number often means "offload as many as possible." 16 | PARAMETER num\_gpu 43 |
| num\_thread | Modelfile, API | Sets the number of CPU threads used for computation. Recommended to be set to the number of physical CPU cores for optimal performance. 17 | PARAMETER num\_thread 8 |
| main\_gpu | Environment Variable / API Option | In a multi-GPU setup, specifies the index of the GPU that stores the KV cache and handles small, non-split tensors. 18 | OLLAMA\_LLAMA\_MAIN\_GPU=1 |
| tensor\_split | Environment Variable / API Option | In a multi-GPU setup, defines the proportional split of tensor data across GPUs. The values are relative proportions. 18 | OLLAMA\_LLAMA\_TENSOR\_SPLIT=3,1 |
| CUDA\_VISIBLE\_DEVICES | Environment Variable | A system-level NVIDIA variable that restricts which GPU(s) are visible to an application. Crucial for forcing model placement. 22 | CUDA\_VISIBLE\_DEVICES=0 |

## **Section 4: Advanced Strategy: Multi-GPU Execution and Load Balancing**

While Ollama's single-GPU performance strategy is relatively mature, its approach to utilizing multiple GPUs simultaneously represents a more nascent and complex area of its functionality. For users with advanced hardware setups, such as workstations with two or more GPUs, achieving optimal performance requires moving beyond Ollama's default behaviors and employing system-level controls and backend-specific parameters to manually orchestrate the workload.

### **4.1 Default Multi-GPU Behavior**

When Ollama detects the presence of multiple GPUs, its default behavior is to attempt to distribute a single large model across them. The underlying logic aims to split the model's layers in a way that balances the VRAM usage on each available card.18 While this approach can be effective for systems with identical GPUs, it frequently leads to suboptimal performance in heterogeneous environments—setups with GPUs of different models and capabilities (e.g., an RTX 4090 paired with an RTX 3090).

A significant flaw, identified by users in detailed bug reports, is that the default allocation logic appears to prioritize VRAM capacity over raw performance. In one documented case, a user with both an RTX 4090 (faster, but with 24564 MiB of VRAM) and an RTX 3090 (slower, but with 24576 MiB of VRAM) found that Ollama would default to using the slower RTX 3090 simply because it had marginally more memory. Furthermore, when a model was too large for the faster GPU, Ollama would fall back to using system RAM rather than intelligently splitting the model across both available GPUs.24 The expected and more logical behavior would be to prioritize the fastest GPU for any model that fits within its VRAM, and only engage the secondary GPU when the model's size necessitates a split.24 This rudimentary default logic means that advanced users cannot rely on Ollama's automatic configuration for multi-GPU setups and must take manual control.

### **4.2 Advanced Control for Multi-GPU Setups**

To overcome the limitations of the default behavior, users must leverage a combination of environment variables and backend parameters, effectively breaking through the Ollama abstraction layer to control the underlying system.

* **CUDA\_VISIBLE\_DEVICES**: This standard NVIDIA environment variable is the most powerful and commonly used tool for multi-GPU management. It allows a user to control which physical GPUs are "visible" to the Ollama process. For example, by setting CUDA\_VISIBLE\_DEVICES=0, a user can force Ollama to see and use only the GPU at index 0 (typically the fastest card in the primary PCIe slot). This is the standard workaround to ensure that smaller models run exclusively on the most performant GPU, preventing unwanted splitting.22  
* **tensor\_split**: This is a parameter inherited from the underlying llama.cpp backend that provides fine-grained control over how a model's layers are partitioned across multiple GPUs. It takes a comma-separated list of values representing the proportion of the model to be placed on each GPU. For instance, OLLAMA\_LLAMA\_TENSOR\_SPLIT=3,1 would instruct the backend to place 75% of the offloaded layers on the first visible GPU and 25% on the second.18 This is essential for manually balancing the load on heterogeneous GPUs according to their respective VRAM capacities.  
* **main\_gpu**: Another llama.cpp parameter, main\_gpu designates a specific GPU to handle tasks that are not worth splitting across multiple devices, such as storing the KV cache and processing intermediate results for small tensors. By default, this is GPU 0\. In a carefully tuned setup, specifying a different main\_gpu can help optimize VRAM allocation and reduce overhead.18

The current state of multi-GPU support in Ollama demonstrates that while the potential for multi-GPU execution exists, a sophisticated and automated management strategy has not yet been implemented. The responsibility for optimization falls entirely on the user, who must possess a deep understanding of their hardware and the underlying backend tools to achieve desirable results.

### **Table 2: Multi-GPU Scenarios and Solutions**

The following table provides a practical, problem-oriented guide for users navigating common multi-GPU challenges.

| Scenario | Default Ollama Behavior | Recommended Solution | Example Command/Configuration |
| :---- | :---- | :---- | :---- |
| **Run a small model on the fastest of two GPUs (e.g., 4090 and 3090).** | May split the model across both GPUs or incorrectly choose the slower GPU if it has slightly more VRAM. 24 | Use CUDA\_VISIBLE\_DEVICES to make only the fastest GPU visible to the Ollama process. | CUDA\_VISIBLE\_DEVICES=0 ollama run \<model\> |
| **Split a very large model across two heterogeneous GPUs (e.g., 24GB and 12GB).** | Attempts to balance VRAM usage, but may not be optimal for different capacities. | Use tensor\_split to manually define the load proportion based on the VRAM ratio of the GPUs. | OLLAMA\_LLAMA\_TENSOR\_SPLIT=2,1 ollama run \<model\> |
| **Run two different models concurrently, one on each GPU.** | Not directly supported by a single Ollama server instance, which will try to use all visible GPUs for one model. | Run two separate Ollama server instances on different network ports, each with a unique CUDA\_VISIBLE\_DEVICES setting. 22 | **Terminal 1:** CUDA\_VISIBLE\_DEVICES=0 ollama serve \--port 11434 **Terminal 2:** CUDA\_VISIBLE\_DEVICES=1 ollama serve \--port 11435 |

## **Section 5: System-Level Dependencies and Interactions**

Ollama's performance is not determined in a vacuum. Its ability to effectively leverage a hybrid CPU/GPU strategy is critically dependent on a correctly configured host system. A significant portion of user-reported performance issues, particularly those where the GPU is not being utilized at all, can be traced back to problems within the surrounding environment rather than bugs in the Ollama application itself. These dependencies range from hardware drivers to containerization settings and advanced memory management options.

### **5.1 Hardware Drivers: The Non-Negotiable Prerequisite**

The most fundamental requirement for GPU acceleration is the proper installation of vendor-specific hardware drivers. Without them, Ollama will be unable to detect or communicate with the GPU and will silently fall back to a CPU-only execution mode.

* **NVIDIA:** For NVIDIA GPUs, a compatible version of the CUDA Toolkit and the corresponding NVIDIA display driver must be installed. The standard command-line utility nvidia-smi is the definitive tool for verifying a successful installation. If this command executes correctly and displays details about the installed GPU, the drivers are likely functional.23 Many user support threads begin and end with troubleshooting driver installations.10  
* **AMD:** For AMD GPUs, support is enabled through the ROCm (Radeon Open Compute) platform. Ollama provides a specific installation package that includes the necessary ROCm libraries.25 The project officially supports a range of modern Radeon RX series and professional AMD Instinct cards, and users must ensure they have the correct ROCm drivers installed for their specific hardware.28

### **5.2 Containerization: GPU Acceleration in Docker**

Deploying Ollama within a Docker container is a popular method for ensuring a consistent and isolated environment.29 However, enabling GPU access from within a container requires additional host system configuration. By default, Docker containers are isolated from the host's hardware devices. To grant a containerized Ollama instance access to the GPU, the host system must have the NVIDIA Container Toolkit (or an equivalent for AMD GPUs) installed.

Furthermore, the Docker container or service must be explicitly configured to request GPU resources. In a docker-compose.yml file, this is typically achieved by adding a deploy section to the Ollama service definition. This section must include a resources block with reservations for devices that specify the nvidia driver and a count.30 If this configuration is omitted, the Ollama container will run in CPU-only mode, a common point of confusion for users new to GPU-accelerated container workflows.

### **5.3 Memory Management: mmap and mlock**

Deeper analysis of user-reported performance issues reveals the influence of lower-level memory management settings inherited from the llama.cpp backend: use\_mmap and use\_mlock.14 These settings control how the model's weights are loaded from disk into memory and can have a significant impact on performance, especially on systems with limited RAM.

* use\_mmap: true: This setting instructs the operating system to use memory-mapping to load the model file. Instead of reading the entire file into RAM at once, it maps the file directly into the process's virtual address space. This is highly efficient, as the OS only needs to load pages from the disk into physical RAM as they are accessed. This is particularly beneficial for hybrid execution, as the layers destined for the CPU may not need to be loaded into physical RAM immediately.  
* use\_mlock: true: This setting locks the model's memory pages into physical RAM, preventing the operating system from swapping them out to disk. On a system under memory pressure, the OS might otherwise page out parts of the model to make room for other applications, leading to significant performance degradation (known as page thrashing) when those parts need to be read back from the slow disk. Using mlock ensures consistent performance at the cost of permanently reserving a large chunk of physical RAM.

The interplay of these settings is crucial for diagnosing performance bottlenecks on memory-constrained systems. An incorrect configuration could lead to excessive disk I/O, negating the benefits of even partial GPU offloading. This illustrates that Ollama's performance strategy is not self-contained; it is deeply intertwined with and dependent upon a correctly configured and well-understood host system, extending the user's responsibility far beyond simply executing the ollama run command.

## **Section 6: Synthesis and Practical Recommendations**

The comprehensive analysis of Ollama's architecture, configuration parameters, and system dependencies reveals a multi-faceted performance strategy. It is a strategy built on a foundation of abstraction, designed for accessibility, but one that provides powerful, if sometimes obscured, levers for manual optimization. This final section synthesizes these findings into a cohesive overview and provides a practical, actionable workflow for users to tune their Ollama experience for maximum performance.

### **6.1 Recapping the Strategy: Abstraction, Offloading, and Manual Overrides**

Ollama's approach to mixing CPU and GPU usage can be understood as a three-tiered strategy:

1. **Abstraction:** The primary strategic decision was to build upon the powerful and mature GGML library rather than developing a new inference engine. Ollama's core value is the creation of a simple, polished, and user-friendly interface (CLI, API, Docker images) that abstracts away the underlying complexity of llama.cpp and its dependencies.1  
2. **Automatic Offloading:** For the user, the default experience is one of automation. Ollama attempts to intelligently detect system hardware and automatically split model layers between the GPU's VRAM and the system's RAM. The goal is to provide a "good enough" performance out-of-the-box with zero configuration.12  
3. **Manual Overrides:** Recognizing that automatic logic is imperfect, especially with diverse hardware and models, the strategy relies on providing "escape hatches" for power users. These take the form of parameters like num\_gpu and the use of system-level tools like the CUDA\_VISIBLE\_DEVICES environment variable. These overrides allow knowledgeable users to bypass the automatic configuration and fine-tune the hybrid execution to match their specific needs, often unlocking significant performance gains.8

This layered strategy successfully caters to a wide spectrum of users. It allows beginners to get up and running with a single command, while simultaneously providing the necessary tools for experts to push their hardware to its limits.

### **6.2 A Practical Workflow for Performance Tuning**

To achieve optimal performance with Ollama, users should adopt a systematic tuning workflow that moves from system-level verification to fine-grained parameter adjustment.

1. **System Verification:** Before attempting to run any model, verify the foundational system configuration. Use nvidia-smi (for NVIDIA) or rocminfo (for AMD) to confirm that the GPU drivers are installed correctly and the hardware is detected.23 If using Docker, ensure the NVIDIA Container Toolkit is installed and that the  
   docker-compose.yml or docker run command includes the necessary flags to grant the container GPU access.30  
2. **Model Selection:** The most significant factor in performance is the model itself. Consult model size charts and VRAM requirement tables to select a model and quantization level that is slightly smaller than your GPU's available VRAM.2 A smaller model that runs entirely on the GPU will always be faster than a larger model that spills over to the CPU.  
3. **Initial Run & Monitoring:** Run the selected model with default settings. Immediately inspect the Ollama server logs for the offloaded X/Y layers to GPU message.14 This will reveal how many layers the automatic logic decided to offload. Concurrently, use the  
   ollama ps command in another terminal to see the real-time CPU/GPU split percentage for the running model.11  
4. **Manual Tuning with num\_gpu:** If the logs and ollama ps indicate that VRAM is being underutilized (e.g., only 80% of VRAM is used, and not all layers are offloaded), it is time for manual intervention. Create a custom Modelfile for your model and add the PARAMETER num\_gpu \<value\> line. Start with a value slightly higher than what Ollama chose automatically. Re-create the model (ollama create...) and run it again, monitoring the logs and VRAM usage. Incrementally increase the num\_gpu value until performance is maximized or an Out-of-Memory (OOM) error occurs, then back off slightly.  
5. **Multi-GPU Optimization:** For systems with multiple GPUs, begin by simplifying the problem. Use the CUDA\_VISIBLE\_DEVICES environment variable to force Ollama to use only your fastest GPU.22 This is the optimal strategy for any model that can fit on a single card. Only engage multiple GPUs for models that are too large for your biggest single GPU. In these cases, use the  
   tensor\_split parameter to manually define the layer distribution in proportion to each GPU's VRAM.18

### **6.3 The Future of Ollama's Performance Strategy**

The analysis of current capabilities and user-reported issues points toward several areas for future improvement in Ollama's performance strategy. Active development, such as the "multi-GPU performance improvements" noted in recent release announcements 31, suggests these are known areas of focus.

Potential future enhancements could include:

* **Performance-Aware GPU Selection:** Implementing more sophisticated multi-GPU logic that considers not just VRAM capacity but also the raw performance characteristics (e.g., core count, clock speed, memory bandwidth) of each GPU when making allocation decisions, as requested by users.24  
* **Improved Documentation and First-Class Parameters:** Elevating advanced parameters like num\_gpu, main\_gpu, and tensor\_split to be first-class, well-documented options within the core Ollama framework. This would close the knowledge gap that currently sends users to community forums for critical information.  
* **More Accurate Automatic Offloading:** Refining the algorithm that automatically calculates the number of layers to offload. A more accurate algorithm that better accounts for different model architectures and memory overheads would reduce the need for manual tuning and improve the out-of-the-box experience for all users.

In conclusion, Ollama's strategy for hybrid computation is a masterful exercise in balancing competing goals. It successfully democratizes access to powerful LLMs through an elegant layer of abstraction, while its reliance on the robust GGML backend provides the underlying power and flexibility for expert users. The path to mastering Ollama's performance lies in understanding this duality—knowing when to trust the automation and when to peel back the abstraction to take direct control of the powerful engine within.

#### **引用的著作**

1. Ollama Turbo | Hacker News, 檢索日期：8月 10, 2025， [https://news.ycombinator.com/item?id=44802414](https://news.ycombinator.com/item?id=44802414)  
2. Does Ollama Need a GPU? \- Collabnix, 檢索日期：8月 10, 2025， [https://collabnix.com/does-ollama-need-a-gpu/](https://collabnix.com/does-ollama-need-a-gpu/)  
3. TheBloke/guanaco-33B-GGML · Using CPU only \- Hugging Face, 檢索日期：8月 10, 2025， [https://huggingface.co/TheBloke/guanaco-33B-GGML/discussions/1](https://huggingface.co/TheBloke/guanaco-33B-GGML/discussions/1)  
4. ollama/fs/ggml/ggml.go at main \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ollama/ollama/blob/main/fs/ggml/ggml.go](https://github.com/ollama/ollama/blob/main/fs/ggml/ggml.go)  
5. ollama/ollama: Get up and running with OpenAI gpt-oss, DeepSeek-R1, Gemma 3 and other models. \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ollama/ollama](https://github.com/ollama/ollama)  
6. Modelfile Reference \- Ollama English Documentation, 檢索日期：8月 10, 2025， [https://ollama.readthedocs.io/en/modelfile/](https://ollama.readthedocs.io/en/modelfile/)  
7. Choosing the Right GPU for LLMs on Ollama \- Database Mart, 檢索日期：8月 10, 2025， [https://www.databasemart.com/blog/choosing-the-right-gpu-for-popluar-llms-on-ollama](https://www.databasemart.com/blog/choosing-the-right-gpu-for-popluar-llms-on-ollama)  
8. Ollama not respecting num\_gpu to load entire model into VRAM for a model that I know should fit into 24GB. \#1906 \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/jmorganca/ollama/issues/1906](https://github.com/jmorganca/ollama/issues/1906)  
9. Need help understanding how offloading layers to GPU works : r/Oobabooga \- Reddit, 檢索日期：8月 10, 2025， [https://www.reddit.com/r/Oobabooga/comments/13y7nqt/need\_help\_understanding\_how\_offloading\_layers\_to/](https://www.reddit.com/r/Oobabooga/comments/13y7nqt/need_help_understanding_how_offloading_layers_to/)  
10. Ollama not using GPU (RTX 3090\) anymore on Ubuntu 20.04 – (it previously worked) \#9842, 檢索日期：8月 10, 2025， [https://github.com/ollama/ollama/issues/9842](https://github.com/ollama/ollama/issues/9842)  
11. High GPU and CPU usage · Issue \#6816 \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ollama/ollama/issues/6816](https://github.com/ollama/ollama/issues/6816)  
12. openai/gpt-oss-20b · Is CPU offloading possible? \- Hugging Face, 檢索日期：8月 10, 2025， [https://huggingface.co/openai/gpt-oss-20b/discussions/26](https://huggingface.co/openai/gpt-oss-20b/discussions/26)  
13. Automatically choosing optimal amount of layers to offload to GPUs · ggml-org llama.cpp · Discussion \#4049 \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ggerganov/llama.cpp/discussions/4049](https://github.com/ggerganov/llama.cpp/discussions/4049)  
14. \[QUESTION\] Why is gpu not using full power or mid to 80% while processing requests ? · Issue \#8850 \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ollama/ollama/issues/8850](https://github.com/ollama/ollama/issues/8850)  
15. Strange processing after update to 0.7 · Issue \#10752 · ollama/ollama \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ollama/ollama/issues/10752](https://github.com/ollama/ollama/issues/10752)  
16. num\_gpu (Ollama) parameter in settings seems to be poorly explained by the hint. \#8348, 檢索日期：8月 10, 2025， [https://github.com/open-webui/open-webui/discussions/8348](https://github.com/open-webui/open-webui/discussions/8348)  
17. docs/modelfile.md \- Ollama \- GitLab, 檢索日期：8月 10, 2025， [https://gitlab.informatik.uni-halle.de/ambcj/ollama/-/blob/ab6be852c77064d7abeffb0b03c096aab90e95fe/docs/modelfile.md](https://gitlab.informatik.uni-halle.de/ambcj/ollama/-/blob/ab6be852c77064d7abeffb0b03c096aab90e95fe/docs/modelfile.md)  
18. Fine grained control of GPU offloading · ggml-org llama.cpp · Discussion \#7678 \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ggml-org/llama.cpp/discussions/7678](https://github.com/ggml-org/llama.cpp/discussions/7678)  
19. API Reference \- Ollama English Documentation, 檢索日期：8月 10, 2025， [https://ollama.readthedocs.io/en/api/](https://ollama.readthedocs.io/en/api/)  
20. huihui\_ai/deepseek-r1-pruned \- Ollama, 檢索日期：8月 10, 2025， [https://ollama.com/huihui\_ai/deepseek-r1-pruned](https://ollama.com/huihui_ai/deepseek-r1-pruned)  
21. What happen with PARAMETER num\_gpu \- ollama \- Reddit, 檢索日期：8月 10, 2025， [https://www.reddit.com/r/ollama/comments/1d29wdx/what\_happen\_with\_parameter\_num\_gpu/](https://www.reddit.com/r/ollama/comments/1d29wdx/what_happen_with_parameter_num_gpu/)  
22. How to run Ollama only on a dedicated GPU? (Instead of all GPUs) · Issue \#1813 \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ollama/ollama/issues/1813](https://github.com/ollama/ollama/issues/1813)  
23. Running Ollama on NVIDIA GPUs: A Comprehensive Guide \- Arsturn, 檢索日期：8月 10, 2025， [https://www.arsturn.com/blog/running-ollama-on-nvidia-gpus-a-comprehensive-guide](https://www.arsturn.com/blog/running-ollama-on-nvidia-gpus-a-comprehensive-guide)  
24. Ollama logic for GPU choice is suboptimal. · Issue \#9462 \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ollama/ollama/issues/9462](https://github.com/ollama/ollama/issues/9462)  
25. ollama/docs/linux.md at main \- GitHub, 檢索日期：8月 10, 2025， [https://github.com/ollama/ollama/blob/main/docs/linux.md](https://github.com/ollama/ollama/blob/main/docs/linux.md)  
26. Is Ollama supposed to run on your GPU? : r/LocalLLaMA \- Reddit, 檢索日期：8月 10, 2025， [https://www.reddit.com/r/LocalLLaMA/comments/1cew9fb/is\_ollama\_supposed\_to\_run\_on\_your\_gpu/](https://www.reddit.com/r/LocalLLaMA/comments/1cew9fb/is_ollama_supposed_to_run_on_your_gpu/)  
27. Ollama refuses to use GPU even on 1.5b parameter models \- Reddit, 檢索日期：8月 10, 2025， [https://www.reddit.com/r/ollama/comments/1l08k4w/ollama\_refuses\_to\_use\_gpu\_even\_on\_15b\_parameter/](https://www.reddit.com/r/ollama/comments/1l08k4w/ollama_refuses_to_use_gpu_even_on_15b_parameter/)  
28. Ollama now supports AMD graphics cards, 檢索日期：8月 10, 2025， [https://ollama.com/blog/amd-preview](https://ollama.com/blog/amd-preview)  
29. Open WebUI: Home, 檢索日期：8月 10, 2025， [https://docs.openwebui.com/](https://docs.openwebui.com/)  
30. How to run Ollama locally on GPU with Docker | by Sujith R Pillai \- Medium, 檢索日期：8月 10, 2025， [https://medium.com/@srpillai/how-to-run-ollama-locally-on-gpu-with-docker-a1ebabe451e0](https://medium.com/@srpillai/how-to-run-ollama-locally-on-gpu-with-docker-a1ebabe451e0)  
31. What GPU do I need to run A.I. (Ollama)? : r/selfhosted \- Reddit, 檢索日期：8月 10, 2025， [https://www.reddit.com/r/selfhosted/comments/1g0rpfp/what\_gpu\_do\_i\_need\_to\_run\_ai\_ollama/](https://www.reddit.com/r/selfhosted/comments/1g0rpfp/what_gpu_do_i_need_to_run_ai_ollama/)
