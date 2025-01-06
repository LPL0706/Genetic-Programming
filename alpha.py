import yfinance as yf
import pandas as pd
import numpy as np
from deap import base, creator, gp, tools
from scipy.stats import spearmanr
from functools import partial
from data import *

import warnings
warnings.filterwarnings('ignore')

def setup():   
    pset = gp.PrimitiveSet("MAIN", 20)
    pset.renameArguments(
        ARG0="PriceMomentum5", ARG1="VolumeMomentum5", ARG2="Volatility5",
        ARG3="PriceVolumeCorr5", ARG4="Channel5", ARG5="RSI5", ARG6="Bollinger5",
        ARG7="PriceMomentum10", ARG8="VolumeMomentum10", ARG9="Volatility10",
        ARG10="PriceVolumeCorr10", ARG11="Channel10", ARG12="RSI10", ARG13="Bollinger10",
        ARG14="VolumeImbalance", ARG15="High", ARG16="Open", ARG17="Low", ARG18="Close", ARG19="Volume"
    )
    
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="subtract")
    pset.addPrimitive(np.multiply, 2, name="multiply")
    pset.addPrimitive(safe_divide, 2, name="div")
    pset.addPrimitive(np.negative, 1, name="neg")
    pset.addPrimitive(np.sqrt, 1, name="sqrt")
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.minimum, 2, name="min")
    
    pset.addPrimitive(lambda x: ts_mean(x, 5), 1, name="ts_mean_5")
    pset.addPrimitive(lambda x: ts_std(x, 5), 1, name="ts_std_5")
    pset.addPrimitive(lambda x: ts_rank(x, 5), 1, name="ts_rank_5")
    pset.addPrimitive(lambda x, y: ts_corr(x, y, 5), 2, name="ts_corr_5")
    pset.addPrimitive(lambda x: ts_max(x, 5), 1, name="ts_max_5")
    pset.addPrimitive(lambda x: ts_min(x, 5), 1, name="ts_min_5")
    
    pset.addPrimitive(lambda x: ts_mean(x, 10), 1, name="ts_mean_10")
    pset.addPrimitive(lambda x: ts_std(x, 10), 1, name="ts_std_10")
    pset.addPrimitive(lambda x: ts_rank(x, 10), 1, name="ts_rank_10")
    pset.addPrimitive(lambda x, y: ts_corr(x, y, 10), 2, name="ts_corr_10")
    pset.addPrimitive(lambda x: ts_max(x, 10), 1, name="ts_max_10")
    pset.addPrimitive(lambda x: ts_min(x, 10), 1, name="ts_min_10")

    pset.addEphemeralConstant("rand", lambda: np.random.uniform(-1, 1))
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox, pset

def fitness(individual, data, toolbox):
    compiled_expr = toolbox.compile(expr=individual)
    all_ic = []
    
    for date in data[list(data.keys())[0]].index:
        try:
            cross_section = []
            
            for ticker, df in data.items():
                if date in df.index:
                    feature_values = compiled_expr(
                        df.loc[date, 'PriceMomentum5'],
                        df.loc[date, 'VolumeMomentum5'],
                        df.loc[date, 'Volatility5'],
                        df.loc[date, 'PriceVolumeCorr5'],
                        df.loc[date, 'Channel5'],
                        df.loc[date, 'RSI5'],
                        df.loc[date, 'Bollinger5'],
                        df.loc[date, 'PriceMomentum10'],
                        df.loc[date, 'VolumeMomentum10'],
                        df.loc[date, 'Volatility10'],
                        df.loc[date, 'PriceVolumeCorr10'],
                        df.loc[date, 'Channel10'],
                        df.loc[date, 'RSI10'],
                        df.loc[date, 'Bollinger10'],
                        df.loc[date, 'VolumeImbalance'],
                        df.loc[date, 'High'],
                        df.loc[date, 'Open'],
                        df.loc[date, 'Low'],
                        df.loc[date, 'Close'],
                        df.loc[date, 'Volume']
                    )
                    cross_section.append((feature_values, df.loc[date, 'Return']))
            
            if len(cross_section) > 1:
                features, targets = zip(*cross_section)
                corr, _ = spearmanr(features, targets)
                if not np.isnan(corr):
                    all_ic.append(corr)
        
        except Exception as e:
            continue
    
    return np.nanmean(all_ic) if all_ic else 0,

def run_evolution(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def run(data, n=100, generations=50, cxpb=0.7, mutpb=0.3):
    processed_data = preprocess_data(data)

    toolbox, pset = setup()
    population = toolbox.population(n=n)
    generations = generations
    hof = tools.HallOfFame(1)

    fitness_func = partial(fitness, data=processed_data, toolbox=toolbox)
    toolbox.register("evaluate", fitness_func)
    
    run_evolution(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=None, halloffame=hof, verbose=True)

    print("Best Expression:", hof[0])
    print("Best Fitness:", fitness_func(hof[0])[0])
    return hof

def main():
    tickers = get_tickers()
    start = "2024-08-01"
    end = "2024-11-30"
    data = download_data(tickers, start, end)
    run(data)

if __name__ == '__main__':
    main()
