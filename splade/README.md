# Neural Conversational Search - SPLADE

---

This section of the code relies on the [SPLADE GitHub repository](https://github.com/naver/splade) from the original SPLADE authors, Formal et al. We only truncated all unnecessary functions in the context of this tutorial. This `README.md` file is specific to the SPLADE repository and outlines the organization of the various files contained within.

### Repository Overview

- **Indexing and Evaluation**:
  - Indexing and evaluation are performed using a Numba inverted index. All methods and functions for these tasks are defined in `/splade/evaluation/`.

**Configuration**: The SPLADE repository utilizes Hydra for configuration management. You can find and modify the configuration files in `/splade/conf/`.

We removed all code for training SPLADE in this repo.

### Key Scripts

- **`retrieve.py`**: Script to perform retrieval.
- **`evaluate.py`**: Script to evaluate the model.


- **`retrieve_interactive.py`**: Script to perform retrieval interactively.

