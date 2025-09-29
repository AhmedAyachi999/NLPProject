# Project Overview

This project demonstrates a framework for comparing and evaluating different models.

## Architecture

- **Base Model**: The main architecture can be understood from the `models` folder. All models inherit or use this base model.
- **Model Comparator**: The `model_comparator` is the main function. It takes different models as input and executes them for comparison.
- **Dependencies**: Install required packages listed in `requirements.txt`.

## Components

- **Contradiction Example**: Uses the Facebook BART model for detecting contradictions.
- **Pattern Discoverer**: Code for generating a knowledge graph from data.
- **Synthetic Manufacturing Data**: Example data generated via ChatGPT, based on predefined rules and randomness.
