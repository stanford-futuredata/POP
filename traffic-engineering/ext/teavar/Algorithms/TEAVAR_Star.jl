include("../util.jl")

using JuMP, Gurobi, MathOptFormat

function TEAVAR_Star(env,
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
                max_concurrent_flow=true,
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

    model = Model(solver=GurobiSolver(env, OutputFlag=1, Method=2, Crossover=0))
    # flow per commodity per path variables
    # @variable(model, a[1:nflows, 1:k] >= 0, basename="a", category=:SemiCont)
	@variable(model, a[1:nflows, 1:k] >= 0, basename="a")

    # alpha variable
    # @variable(model, alpha >= 0, basename="alpha", category=:SemiCont)
    @variable(model, alpha >= 0, basename="alpha")
	
    # maximum flow lost in that scenario
    @variable(model, umax[1:nscenarios] >= 0, basename="umax")
    # flow lost per commod per scenario
    @variable(model, u[1:nscenarios, 1:nflows] >= 0, basename="u")

    # capacity constraints for final flow assigned to "a" variables
    for e in 1:nedges
        @constraint(model, sum(a[f,t] * L[Tf[f][t],e] for f in 1:nflows, t in 1:size(Tf[f],1)) <= capacity[e])
    end

    println(Dates.format(now(), "HH:MM:SS"), ": some variables and constraints are created")
    total_scenario_prob = sum(p[s] for s in 1:nscenarios)
    println("total_scenario_prob= ", total_scenario_prob)
    total_demand = sum(demand[f] for f in 1:nflows)
 
    if max_concurrent_flow
      println("Running max concurrent flow")
      # FLOW LEVEL LOSS
      @expression(model, satisfied[s=1:nscenarios, f=1:nflows], sum(a[f,t] * X[s,Tf[f][t]] for t in 1:size(Tf[f],1)) / demand[f])

      for s in 1:nscenarios
          for f in 1:nflows
              @constraint(model, u[s,f] >= 1 - satisfied[s,f])
          end
      end

      for s in 1:nscenarios
          if average
              @constraint(model, umax[s] + alpha >= (sum(u[s,f] for f in 1:nflows)) / nflows)
          else
              for f in 1:nflows
                  @constraint(model, umax[s] + alpha >= u[s,f])
              end
          end
      end
      
      @objective(model, Min, alpha + (1 / (1 - beta)) * (sum((p[s] * umax[s] for s in 1:nscenarios)) + ((1-total_scenario_prob) * (1-alpha) )))

    else
      # TODO: do these loss formulas hold if demands are not satisfiable?
      println("Running max total flow")
      @expression(model, satisfied[s=1:nscenarios, f=1:nflows], sum(a[f,t] * X[s, Tf[f][t]] for t in 1:size(Tf[f],1)))

      for s in 1:nscenarios
	@constraint(model, umax[s] >= 0)
	for f in 1:nflows
	  # note: satisfied[s,f] can be more than demand[f]
	  @constraint(model, u[s, f] >= demand[f] - satisfied[s,f])
	end
	# @expression(model, sum_u_f[s=1:nscenarios], sum(u[s,f] for f in 1:nflows))
	@constraint(model, umax[s] + alpha >= sum(u[s, f] for f in 1:nflows))
      end


      @objective(model, Min, alpha + (1/ (1-beta)) * (sum((p[s] * umax[s] for s in 1:nscenarios)) + ((1-total_scenario_prob)*(total_demand-alpha))))
    end

    # println(Dates.format(now(), "HH:MM:SS"), ": writing to file")
    # writeMPS(model, "blahblahjl.mps")
    # MOI.write_to_file(model, "blahblahjl.lp")

    println(Dates.format(now(), "HH:MM:SS"), ": ready to solve")
    flush(stdout)

    solve(model)

    println(Dates.format(now(), "HH:MM:SS"), ": solver finished; explaining")
    if (explain)
        println("Runtime: ", getsolvetime(model))

	
	# some simple debugging information
	println("beta: ", beta)
	println("#flows: ", nflows)
	println("#edges: ", nedges)
	println("#tunnels: ", ntunnels)
	println("#demands: ", nflows, " total demand=", total_demand)
	println("#scenarios: ", nscenarios, " total_prob= ", sum(p[s] for s in 1:nscenarios))

	result_u = getvalue(u)
	result_umax = getvalue(umax)
	result_satisfied = getvalue(satisfied)
	result_alpha = getvalue(alpha)

	for f in 1:nflows
		sat_vector=Array{Float64}(undef, nscenarios)
		for s in 1:nscenarios
			sat_vector[s] = result_satisfied[s,f]
		end
		println("satisfied[*,", f, "]: ", sat_vector)
	end

	per_scenario_losses = Array{Tuple{Float64,Int64,Float64}}(undef, nscenarios)

	if max_concurrent_flow
		loss_s_f = ones(nflows)' .- result_satisfied
	else
		loss_s_f = demand' .- result_satisfied
	end
	loss_s_f[loss_s_f .< 0] .= 0

	for s in 1:nscenarios
	     if max_concurrent_flow
		loss_s = maximum(loss_s_f[s, :])
	     else
		loss_s = sum(loss_s_f[s, :])
	     end
	     # max_loss_s = maximum(loss_s_f[s, :])
	     # max_u_s = maximum(result_u[s, :])
	     # println(s, ": max_loss_s/max_u_s/result_umax_s= ", max_loss_s, " ", max_u_s, " ", result_umax[s])
	     per_scenario_losses[s] = (loss_s, s, p[s])
        end
	println("psl before sort=", per_scenario_losses)
	per_scenario_losses = sort(per_scenario_losses, by = first)
	println("psl after sort=", per_scenario_losses)


	println("Alpha after obj=", result_alpha)

	result_a = getvalue(a)

	total_prob_thus_far = 0
	ell_k_beta = -1
	k_beta = -1
	for ss in 1:nscenarios
		tup = per_scenario_losses[ss]
		total_prob_thus_far += p[tup[2]]
		if total_prob_thus_far > beta
			ell_k_beta = tup[1]
			k_beta = ss
			break;
		end
	end
	if total_prob_thus_far <= beta
		ell_k_beta = result_alpha
	end
	println("Sorting scenarios by loss: beta= ", beta, " k_beta= ", k_beta, " ell_k_beta= ", ell_k_beta)

	if max_concurrent_flow
	else
		# x
		result_alpha /= total_demand
		ell_k_beta /= total_demand
	end


	# compute net flow
	numtunnel_perflow = 0
	for f in 1:nflows
		numtunnel_perflow = max(numtunnel_perflow, size(Tf[f], 1))
	end

	flow_allocation_1 = Array{Float64}(undef, nflows)
	flow_allocation_2 = Array{Float64}(undef, nflows)
	flow_tunnel_weights = Matrix{Float64}(undef, nflows, numtunnel_perflow)
	all_flow_alloc = 0

	for f in 1:nflows
		flow_totalalloc = 0
	 	for t in 1:size(Tf[f], 1)
	  		flow_totalalloc += result_a[f,t]
         	end

	 	all_flow_alloc += flow_totalalloc

	 	flow_allocation_1[f] = (1 - result_alpha) * min(flow_totalalloc, demand[f])
	 	flow_allocation_2[f] = (1 - ell_k_beta) * min(flow_totalalloc, demand[f])
	 	for t in 1:size(Tf[f], 1)
	   		weight = 0
	   		if flow_totalalloc > 0
				weight = result_a[f,t]/ flow_totalalloc
	   		end
	   		flow_tunnel_weights[f,t] = weight
        end
	end
	
	flow_allocation_3 = Array{Float64}(undef, nflows)
	for f in 1:nflows
		flow_per_scenario_losses = Array{Tuple{Float64,Int64,Float64}}(undef, nscenarios)
		for ss in 1:nscenarios
			flow_per_scenario_losses[ss] = (loss_s_f[ss, f], ss, p[ss])
		end
		
		flow_per_scenario_losses = sort(flow_per_scenario_losses, by = first)
		
		flow_total_prob_thus_far = 0
		flow_loss = 0
		flow_loss_cutoff_iter = 0
		if flow_total_prob_thus_far < beta
			for ss in 1:nscenarios
				tup = flow_per_scenario_losses[ss]
				flow_total_prob_thus_far += tup[3]
				if flow_total_prob_thus_far >= beta
					flow_loss = tup[1]
					flow_loss_cutoff_iter = ss
					break
				end
			end
			if flow_loss_cutoff_iter == 0
				println("WARN!!!! did not get flow loss")
			else
				println("Flow ", f, " loss = ", flow_loss, " at cutoffindex= ", flow_loss_cutoff_iter, " totprob= ", flow_total_prob_thus_far)
				println("Flow losses sorted: ", flow_per_scenario_losses)
			end
		end
		
		if max_concurrent_flow
			flow_allocation_3[f] = demand[f] * (1- flow_loss)
		else
			flow_allocation_3[f] = demand[f] - flow_loss
		end
	end
	
	# to account for minor off-by-epsilon negative values
	# should be asserting
	flow_allocation_1[flow_allocation_1 .< 0] .= 0
	flow_allocation_2[flow_allocation_2 .< 0] .= 0
	flow_allocation_3[flow_allocation_3 .< 0] .= 0


	println("totalalloc= ", all_flow_alloc, " netalloc1/2/3= ", 
		sum(flow_allocation_1[f] for f in 1:nflows), " ", 
		sum(flow_allocation_2[f] for f in 1:nflows), " ", 
		sum(flow_allocation_3[f] for f in 1:nflows))

	output=open(outputfilename, "w")
	write(output, "---flow_allocation with alpha---\n")
	writedlm(output, flow_allocation_1, ',')
	write(output, "---flow_allocation with ell_k_beta---\n")
	writedlm(output, flow_allocation_2, ',')
	write(output, "---flow_allocation with per_flow_loss---\n")
	writedlm(output, flow_allocation_3, ',')
	write(output, "---weights per flow per tunnel---\n")
	writedlm(output, flow_tunnel_weights, ',')
	flush(output)
	close(output)

        printResults(getobjectivevalue(model), result_alpha, result_a, result_u, result_umax, edges, scenarios, T, Tf, L, capacity, p, demand, verbose=verbose, utilization=utilization)
    end
    
    return (getvalue(alpha), getobjectivevalue(model), getvalue(a), getvalue(umax), getsolvetime(model))
end

