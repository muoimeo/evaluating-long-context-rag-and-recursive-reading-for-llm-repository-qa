# Dataset Selection for Context-Rot Study

This document outlines the rationale for selecting the datasets used to evaluate the AI Agent's performance across different retrieval and reasoning modes.

## 1. Code Repository: FastAPI
- **Source:** [https://github.com/fastapi/fastapi](https://github.com/fastapi/fastapi)
- **Scale:** ~50,000+ Lines of Code (LOC) and hundreds of files.
- **Complexity:** High. It involves complex Python type hints, dependency injection patterns, and modular routing.
- **Rationale:** The modular nature of FastAPI allows for testing **multi-hop reasoning**, where the agent must traverse multiple files (e.g., from a route definition to a dependency utility) to answer a query.

## 2. Technical Documentation: AWS Lambda Developer Guide
- **Source:** [https://github.com/awsdocs/aws-lambda-developer-guide](https://github.com/awsdocs/aws-lambda-developer-guide)
- **Format:** Sample applications (Python, Node.js, Go, Java, etc.), IAM policy templates, and CloudFormation/SAM templates.
- **Scale:** 22 sample apps, 31 IAM policy files, and 5 CloudFormation templates, reaching the target corpus size of 200k–1M tokens when combined with the repository.
- **Rationale:** The multi-language sample apps (Python, Node.js, Go, Java) provide a rich source for **cross-file reasoning** and language comparison questions. IAM policies and CloudFormation templates allow testing the Agent's ability to cite specific configuration sections accurately.

## 3. Context-Rot Suitability
The combined size of these datasets allows for a controlled "context-rot" experiment by:
1. Incrementally increasing the number of files/sections provided in the prompt.
2. Injecting "distractor" files from other parts of the AWS ecosystem to test the Agent's noise-filtering capabilities.