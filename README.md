# Not All Sequence Model Created Equal: Revealing the Inductive Biases of Sequence Models via Elementary Cellular Automata

| Arch | Weights | Deduction: Class 1 | Deduction: Class 2 | Deduction: Class 3 | Deduction: Class 4 | Deduction: Total | Induction: Class 1 | Induction: Class 2 | Induction: Class 3 | Induction: Class 4 | Induction: Total | Abduction: Class 1 | Abduction: Class 2 | Abduction: Class 3 | Abduction: Class 4 | Abduction: Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Llama Huggingface (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/transformer-baseline-1-epoch) | 43.4% | 28.3% | 26.0% | 28.0% | 28.9% | 72.6% | 52.0% | 27.3% | 29.2% | 37.2% | 38.3% | 40.5% | 13.1% | 17.1% | 20.9% |
| Llama Huggingface (3ep) | [Weights](https://huggingface.co/ChavyvAkvar/transformer-baseline-3-epoch) | 31.0% | 27.4% | 35.9% | 27.6% | 30.8% | 30.1% | 54.6% | 60.9% | 49.1% | 52.5% | 14.9% | 8.1% | 12.4% | 13.4% | 12.1% |
| LFM2 Huggingface (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/lfm2-baseline-1-epoch) | 55.8% | 43.8% | 47.6% | 39.5% | 44.8% | 95.6% | 18.9% | 16.2% | 15.6% | 24.2% | 14.9% | 4.3% | 4.6% | 6.5% | 5.8% |
| **LFM2 Huggingface (3ep)** | [Weights](https://huggingface.co/ChavyvAkvar/lfm2-baseline-3-epoch) | **72.6%** | **66.8%** | **81.6%** | **72.9%** | **74.8%** | **92.0%** | **65.6%** | **74.9%** | **71.8%** | **73.6%** | **23.4%** | **44.3%** | **18.3%** | **19.9%** | **23.9%** |
| Transformer FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/transformer-fla-baseline-1-epoch) | 67.3% | 18.1% | 11.4% | 12.7% | 18.6% | 61.1% | 11.5% | 17.4% | 7.3% | 16.9% | 34.0% | 34.1% | 20.1% | 28.4% | 26.5% |
| Transformer FLA (3ep) | | | | | | | | | | | | | | | | |
| Multi Head Latent Attn FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/mla-baseline-1-epoch) | 82.3% | 36.3% | 28.4% | 32.9% | 36.8% | 92.0% | 15.0% | 2.7% | 0.2% | 12.9% | 25.5% | 31.9% | 9.8% | 14.2% | 16.3% |
| Multi Head Latent Attn FLA (3ep) | | | | | | | | | | | | | | | | |
| Kimi Delta Attn FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/kda-baseline-1-epoch) | 50.4% | 44.2% | 36.7% | 38.8% | 40.2% | 37.2% | 8.8% | 1.9% | 0.2% | 6.1% | 19.1% | 23.8% | 5.7% | 4.4% | 9.1% |
| Kimi Delta Attn FLA (3ep) | | | | | | | | | | | | | | | | |
| *PaTH FLA (1ep)* | [Weights](https://huggingface.co/ChavyvAkvar/pathfox-baseline-1-epoch) | 53.1% | 20.4% | 7.8% | 5.4% | 13.8% | *100.0%* | *89.0%* | *96.1%* | *94.2%* | *94.4%* | *59.6%* | *62.7%* | *28.1%* | *34.1%* | *38.2%* |
| PaTH FLA (3ep) | | | | | | | | | | | | | | | | |
| MesaNet FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/mesanet-baseline-1-epoch) | 14.2% | 11.5% | 11.9% | 12.4% | 12.2% | 0.0% | 1.8% | 1.0% | 0.2% | 0.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| MesaNet FLA (3ep) | | | | | | | | | | | | | | | | |
| Native Sparse Attn FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/nsa-baseline-1-epoch) | | | | | | | | | | | | | | | | |
| Native Sparse Attn FLA (3ep) | | | | | | | | | | | | | | | | |
| Mixture of Memories FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/mom-baseline-1-epoch) | | | | | | | | | | | | | | | | |
| Mixture of Memories FLA (3ep) | | | | | | | | | | | | | | | | |