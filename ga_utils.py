# ga_utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from model_train import evaluate_population, generate_initial_population, select_parents, crossover, mutate, simulate_strategy

def download_yahoo_data(ticker, start="2018-01-01", end=None):
    df = yf.download(ticker, start=start, end=end)
    if df.empty or 'Close' not in df:
        raise ValueError(f"無法下載資料: {ticker}")
    df.dropna(inplace=True)
    return df

def run_ga(df, generations=20, population_size=25):
    population = generate_initial_population(population_size)
    best_fitness = -np.inf
    best_gene = None

    for generation in range(generations):
        fitness_scores = evaluate_population(population, df)
        top_idx = np.argmax(fitness_scores)
        if fitness_scores[top_idx] > best_fitness:
            best_fitness = fitness_scores[top_idx]
            best_gene = population[top_idx]

        parents = select_parents(population, fitness_scores)
        next_generation = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))

        population = np.array(next_generation)

    return best_gene, best_fitness
