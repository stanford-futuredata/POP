include("../util.jl")

using JuMP, Gurobi

function TEAVAR(env,
                edges,
                capacity,
                flows,
                demand,
                beta,
                k,
                T,
                Tf,
                scenarios,
                scenario_probs,
				outputfilename;
                explain=false,
                verbose=false,
                utilization=false,
                average=false)
                
    nedges = length(edges)
    nflows = length(flows)
    ntunnels = length(T)
    nscenarios = length(scenarios)
    p = scenario_probs
	
	println(Dates.format(now(), "HH:MM:SS"), ": #edges ", nedges, " #flows ", nflows, " #tunnels ", ntunnels, " #scenarios ", nscenarios)

    #CREATE TUNNEL SCENARIO MATRIX
    X  = ones(nscenarios,ntunnels)
    for s in 1:nscenarios
        for t in 1:ntunnels
            if size(T[t],1) == 0
                X[s,t] = 0
            else
                for e in 1:nedges
                    if scenarios[s][e] == 0
                        back_edge = findfirst(x -> x == (edges[e][2],edges[e][1]), edges)
                        if in(e, T[t]) || in(back_edge, T[t])
                        # if in(e, T[t])
                            X[s,t] = 0
                        end
                    end
                end
            end
        end
    end
	
	println(Dates.format(now(), "HH:MM:SS"), ": created tunnel scenario matrix")

    #CREATE TUNNEL EDGE MATRIX
    L = zeros(ntunnels, nedges)
    for t in 1:ntunnels
        for e in 1:nedges
            if in(e, T[t])
                L[t,e] = 1
            end
        end
    end

    println(Dates.format(now(), "HH:MM:SS"), ": created tunnel edge matrix")

    model = Model(solver=GurobiSolver(env, OutputFlag=1))
    # flow per commodity per path variables
    @variable(model, a[1:nflows, 1:k] >= 0, basename="a", category=:SemiCont)
    # alpha variable
    @variable(model, alpha >= 0, basename="alpha", category=:SemiCont)
    # maximum flow lost in that scenario
    @variable(model, umax[1:nscenarios] >= 0, basename="umax")
    # flow lost per commod per scenario
    @variable(model, u[1:nscenarios, 1:nflows] >= 0, basename="u")
 
    # capacity constraints for final flow assigned to "a" variables
    for e in 1:nedges
        @constraint(model, sum(a[f,t] * L[Tf[f][t],e] for f in 1:nflows, t in 1:size(Tf[f],1)) <= capacity[e])
    end

      # FLOW LEVEL LOSS
      @expression(model, satisfied[s=1:nscenarios, f=1:nflows], sum(a[f,t] * X[s,Tf[f][t]] for t in 1:size(Tf[f],1)) / demand[f])

      for s in 1:nscenarios
          for f in 1:nflows
              # @constraint(model, (demand[f] - sum(a[f,t] * X[s,Tf[f][t]] for t in 1:size(Tf[f],1))) / demand[f] <= u[s,f])
              @constraint(model, u[s,f] >= 1 - satisfied[s,f])
          end
      end

      for s in 1:nscenarios
          if average
              @constraint(model, umax[s] + alpha >= (sum(u[s,f] for f in 1:nflows)) / nflows)
              # @constraint(model, umax[s] + alpha >= avg_loss[s])
          else
              for f in 1:nflows
                  @constraint(model, umax[s] + alpha >= u[s,f])
              end
          end
      end
      @objective(model, Min, alpha + (1 / (1 - beta)) * sum((p[s] * umax[s] for s in 1:nscenarios)))

    println(Dates.format(now(), "HH:MM:SS"), ": ready to solve")

    solve(model)

    println(Dates.format(now(), "HH:MM:SS"), ": solver finished; explaining")

    if (explain)
        println("Runtime: ", getsolvetime(model))
	println("beta: ", beta)
	println("#flows: ", nflows)
	println("#edges: ", nedges)
	println("#tunnels: ", ntunnels)
	println("#demands: ", nflows, " total demand=", sum(demand[f] for f in 1:nflows))
	println("#scenarios: ", nscenarios, " total_prob= ", sum(p[s] for s in 1:nscenarios))
	println("total_scenario_prob= ", sum(p[s] for s in 1:nscenarios))

	result_u = getvalue(u)
	result_umax = getvalue(umax)
	result_satisfied = getvalue(satisfied)
	result_alpha = getvalue(alpha)
	result_a = getvalue(a)
	
	# compute allocations and weights
	flow_allocation_1 = Array{Float64}(undef, nflows)
	numtunnel_perflow = 0
	for f in 1:nflows
		numtunnel_perflow = max(numtunnel_perflow, size(Tf[f], 1))
	end
	flow_tunnel_weights = Matrix{Float64}(undef, nflows, numtunnel_perflow)
	all_flow_alloc = 0
	
	for f in 1:nflows
		flow_totalalloc = 0
		for t in 1:size(Tf[f], 1)
			flow_totalalloc += result_a[f,t]
		end
		
		all_flow_alloc += flow_totalalloc
		flow_allocation_1[f] = (1 - result_alpha) * min(flow_totalalloc, demand[f])
		for t in 1:size(Tf[f], 1)
			weight = 0
			if flow_totalalloc > 0
				weight = result_a[f, t]/ flow_totalalloc
			end
			flow_tunnel_weights[f, t] = weight
		end
	end
	flow_allocation_1[flow_allocation_1 .< 0] .= 0
	
	println("totalalloc= ", all_flow_alloc, " netalloc1= ", sum(flow_allocation_1[f] for f in 1:nflows))

	output = open(outputfilename, "w")
	write(output, "---flow allocation with alpha---\n")
	writedlm(output, flow_allocation_1, ',')
	write(output, "---weights per flow tunnel---\n")
	writedlm(output, flow_tunnel_weights, ',')
	flush(output)
	close(output)
	
        printResults(getobjectivevalue(model), result_alpha, result_a, result_u, result_umax, edges, scenarios, T, Tf, L, capacity, p, demand, verbose=verbose, utilization=utilization)
    end
    
    return (result_alpha, getobjectivevalue(model), result_a, result_umax, getsolvetime(model))
end

