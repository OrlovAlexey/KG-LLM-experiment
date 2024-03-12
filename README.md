# KG-LLM: Knowledge Graph Large Language Model for Link Prediction

## Abstract:
The task of predicting multiple links within knowledge graphs (KGs) stands as a challenge in the field of knowledge graph analysis, a challenge increasingly resolvable due to advancements in natural language processing (NLP) and KG embedding techniques. This paper introduces a novel methodology, the Knowledge Graph Large Language Model Framework (KG-LLM), which leverages pivotal NLP paradigms, including chain-of-thought (CoT) prompting and in-context learning (ICL), to enhance multi-hop link prediction in KGs. By converting the KG to a CoT prompt, our framework is designed to discern and learn the latent representations of entities and their interrelations. To show the efficacy of the KG-LLM Framework, we fine-tune three leading Large Language Models (LLMs) within this framework, employing both non-ICL and ICL tasks for a comprehensive evaluation. Further, we explore the framework's potential to provide LLMs with zero-shot capabilities for handling previously unseen prompts. Our experimental findings discover that integrating ICL and CoT not only augments the performance of our approach but also significantly boosts the models' generalization capacity, thereby ensuring more precise predictions in unfamiliar scenarios.

## Requirements:
- Set up your Python environment
  
  > Install the requirements.txt file by running
  > ```python
  > pip install -r requirements.txt
  > ```

- Requesting model access from META and Google

  > Visit this [link](https://ai.meta.com/llama/) and request the access to the **Llama-2** models
  
  > Visit this [link](https://blog.google/technology/developers/gemma-open-models/) and request the access to the **Gemma** models

- Requesting model access from HuggingFace

  > Once request is approved, use the same email adrress to get the access of the model from HF [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b) and [Gemma](https://huggingface.co/google/gemma-7b).

- Authorising HF token

  > Once HF request to access the model has been approved, create hugging face token [here](https://huggingface.co/settings/tokens)

  > Run below code and enter your token. It will authenticate your HF account
  ```python
  >>> huggingface-cli login
  
  or
  
  >>> from huggingface_hub import login
  >>> login(YOUR_HF_TOKEN)
  ```

## Dataset:
We conduct experiments over *two* real-world datasets, WN18RR and NELL-995, which are constructed and released by the [OpenKE library](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch).
| Dataset  | #Entities | #Triples  | #Relations  |
| ------------- |:-------------:| -----:| -----:|
| [WN18RR](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/WN18RR)      | 40,943 | 86,835 | 11 |
| [NELL-995](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/NELL-995)      | 75,492      |   149,678 |   200 |

## Preprocessing

1. Clone Repository: Clone this repository to your local machine.

   ```python
   git clone
   ```

3. Download Data: Download the dataset from the previously mentioned library.

4. Place Files: Put the downloaded dataset in the same directory as the `KGLLM_preprocess.py` script.

5. Run Script: Open a terminal or command prompt, navigate to the directory containing the script and files, and run the following command:
  ```python
  python KGLLM_preprocess.py
  ```
5. Check Output: After running the script, you should find some new CSV files containing the preprocessed training data and testing data in the same directory.

## Finetune
TODO

## Test
TODO
