include("./util.jl")
include("./parsers.jl")
include("./Algorithms/TEAVAR.jl")

env = Gurobi.Env()

topology = "B4"
weibull = true
shape = 0.8
scale = 0.01
paths = "SMORE"
demand_num = 1
beta=0.90


links, capacity, link_probs, nodes = readTopology(topology)
demand, flows = readDemand("$(topology)/demand", length(nodes), demand_num, matrix=true)
T, Tf, k = parsePaths("$(topology)/paths/$(paths)", links, flows)

if weibull
  probabilities = weibullProbs(length(links), shape=shape, scale=scale)
else
  probabilities = link_probs
end
cutoff = (sum(probabilities)/length(probabilities))^2
scenarios, scenario_probs = subScenariosRecursion(probabilities, cutoff)
TEAVAR(env, links, capacity, flows, demand, beta, k,
    T, Tf, scenarios, scenario_probs, explain=true, verbose=true,
        utilization=true)


