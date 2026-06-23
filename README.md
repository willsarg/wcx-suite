<div align="center">

# wcx-suite

**Will's CUDA Suite** — a VRAM/stress bench for local CUDA inference on NVIDIA GPUs.

*Find each model's safe context ceiling without exhausting the card.*

</div>

---

wcx-suite is the CUDA sibling of [wmx-suite](https://github.com/willsarg/wmx-suite). Same
discipline, other side of the hardware fence: it finds the **safe context ceiling** for a
model+context on **your** GPU by extrapolating from measurements taken below the hardware
wall — never by probing into it.

On Apple Silicon the wall is the unified-memory working set; here it's the GPU's **VRAM**.
A CUDA over-allocation usually raises an out-of-memory error and kills the *workload* rather
than the machine — but on a GPU that also drives a display, exhausting VRAM can freeze the
desktop or trip a driver reset. The rule holds: stay under the wall.

## Status

Barebones. Today it reads your VRAM wall live (via `nvidia-smi`, zero dependencies):

```bash
uv sync
uv run wcx-suite system     # GPU, VRAM total/free, the wall, and the safe budget
```

Characterization (safe probe → fitted context ceiling), a model registry, and the launch
gate are being built out next, mirroring wmx-suite.

## Where it fits

This is the engine [project-ara](https://github.com/willsarg/project-ara) installs on NVIDIA
hardware (`ara install --engine wcx`), the way wmx-suite is its Apple-Silicon engine.

## License

[Apache 2.0](./LICENSE) © Will Sarg
