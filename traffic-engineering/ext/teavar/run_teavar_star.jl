include("./util.jl")
include("./parsers.jl")
include("./Algorithms/TEAVAR_Star.jl")
using Dates;

env = Gurobi.Env()
setparam!(env, "Method", 2) # choose only barrier; since crossover is needed for concurrent and that takes too long
setparam!(env, "Crossover", 0) # disable barrier crossover

print("method= ", getparam(env, "Method"), "; crossover= ", getparam(env, "Crossover"))

topology = ARGS[1]
# failure probabilities must always come from topology
weibull = false
shape = 0.8
scale = 0.01

# we will change demand numerals and beta
demand_num = parse(Int, ARGS[2]) # used to be 1
beta=parse(Float64, ARGS[3]) # used to be 0.99
max_cf=ARGS[4] == "mcf" 
paths = ARGS[5] # used to be "SMORE" should be "SMORE4" or "SMORE8"
what_to_read = ARGS[6]
cutoff_downscale = parse(Float64, ARGS[7]) # used to 10

# output file for result analysis and debugging
outputfile=string(topology, "_d", demand_num, "_beta", beta, "_mcf", max_cf, "_paths", paths, "_topo", what_to_read, "_cutoffDS", cutoff_downscale)


links, capacity, link_probs, nodes = readTopology(topology, what_to_read=what_to_read)
demand, flows = readDemand("$(topology)/demand", length(nodes), demand_num, matrix=true)
T, Tf, k = parsePaths("$(topology)/paths/$(paths)", links, flows)

if weibull
  probabilities = weibullProbs(length(links), shape=shape, scale=scale)
else
  probabilities = link_probs
end
println("FailureProbs= ", probabilities)

beginning_cutoff = parse(Float64, ARGS[8])

let cutoff = beginning_cutoff
	while true
		println(Dates.format(now(), "HH:MM:SS"), " going to subScenariosRecursion with cutoff=", cutoff)
		flush(stdout)

		task_cutoff_finder = @task global scenarios, scenario_probs = subScenariosRecursion(probabilities, cutoff)
		t = Timer(60)
		# run for 60s or until task cutoff finder finishes
		schedule(task_cutoff_finder)
		while (!@isdefined(scenarios) || length(scenarios) == 0 || isopen(t)) && !istaskdone(task_cutoff_finder)
			# println(Dates.format(now(), "HH::MM::SS"), ": testing")
			# flush(stdout)
			yield()
		end

		nscenarios = length(scenarios)
		total_scenario_prob = sum(scenario_probs)
	
		println(Dates.format(now(), "HH:MM:SS"), " cutoff =", cutoff, " #scenarios=", nscenarios, " total_scenar_prob=", total_scenario_prob)
		flush(stdout)

		if total_scenario_prob >= 1 - (1-beta)/ cutoff_downscale
			break
		end

		if !isopen(t)
			println(Dates.format(now(), "HH:MM:SS"), ": timed out!")
			flush(stdout)
			break
		end

		cutoff = cutoff / cutoff_downscale
	end
end

TEAVAR_Star(env, links, capacity, flows, demand, beta, k,
    T, Tf, scenarios, scenario_probs, outputfile, explain=true, verbose=true,
        utilization=true, max_concurrent_flow=max_cf)


