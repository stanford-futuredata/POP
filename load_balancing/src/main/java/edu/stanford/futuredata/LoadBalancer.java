package edu.stanford.futuredata;

import ilog.concert.*;
import ilog.cplex.IloCplex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

public class LoadBalancer {

    private static final Logger logger = LoggerFactory.getLogger(LoadBalancer.class);
    public static boolean verbose = true;
    public static int minReplicationFactor = 1;

    List<double[]> lastR;
    List<double[]> lastX;
    double[] lastM;
    int lastNumServers = 0;
    int lastNumShards = 0;

    final static double mipGap = 0.1;

    public List<double[]> balanceLoad(Integer numShards, Integer numServers,
                                      int[] shardLoads, int[] shardMemoryUsages, int[][] currentLocations,
                                      Map<Set<Integer>, Integer> sampleQueries,
                                      Integer maxMemory, int splitFactor) throws IloException {
        assert(numShards % splitFactor == 0);
        assert(numServers % splitFactor == 0);
        shardLoads = shardLoads.clone();
        double totalLoad = Arrays.stream(shardLoads).sum();
        int serversPerSplit = numServers / splitFactor;
        int loadPerSplit = (int) Math.round(totalLoad /splitFactor);
        List<double[]> finalRs = new ArrayList<>();
        for(int i = 0; i < numServers; i++) {
            finalRs.add(new double[numShards]);
        }
        Map<Integer, List<Integer>> previousSplitsToShards = new HashMap<>();
        for (int serverNum = 0; serverNum < numServers; serverNum++) {
            int splitNum = serverNum / serversPerSplit;
            previousSplitsToShards.putIfAbsent(splitNum, new ArrayList<>());
            for (int shardNum = 0; shardNum < numShards; shardNum++) {
                if (currentLocations[serverNum][shardNum] == 1 && !previousSplitsToShards.get(splitNum).contains(shardNum)) {
                    previousSplitsToShards.get(splitNum).add(shardNum);
                }
            }
        }
        Map<Integer, List<Integer>> newSplitsToShards = new HashMap<>();
        Map<Integer, List<Integer>> newSplitsToLoads = new HashMap<>();
        Map<Integer, Integer> splitsNeedingMore = new HashMap<>();
        for (int splitNum = 0; splitNum < splitFactor; splitNum++) {
            newSplitsToShards.put(splitNum, new ArrayList<>());
            newSplitsToLoads.put(splitNum, new ArrayList<>());
            int currentLoad = 0;
            int[] finalShardLoads = shardLoads;
            previousSplitsToShards.get(splitNum).sort(Comparator.comparing(i -> finalShardLoads[i]));
            for (int nextShard: previousSplitsToShards.get(splitNum)) {
                int shardLoad = shardLoads[nextShard];
                if (shardLoad == 0) {
                    continue;
                }
                if (currentLoad + shardLoad <= loadPerSplit) {
                    currentLoad += shardLoad;
                    newSplitsToLoads.get(splitNum).add(shardLoad);
                    shardLoads[nextShard] = 0;
                    newSplitsToShards.get(splitNum).add(nextShard);
                    if (currentLoad == loadPerSplit) {
                        break;
                    }
                } else {
                    assert(currentLoad < loadPerSplit);
                    newSplitsToLoads.get(splitNum).add(loadPerSplit - currentLoad);
                    currentLoad = loadPerSplit;
                    shardLoads[nextShard] = currentLoad + shardLoad - loadPerSplit;
                    assert(shardLoads[nextShard] > 0);
                    newSplitsToShards.get(splitNum).add(nextShard);
                    break;
                }
            }
            if (currentLoad < loadPerSplit) {
                splitsNeedingMore.put(splitNum, currentLoad);
            }
        }
        for (int splitNum: splitsNeedingMore.keySet()) {
            int currentLoad = splitsNeedingMore.get(splitNum);
            for (int shardNum = 0; shardNum < numShards; shardNum++) {
                int shardLoad = shardLoads[shardNum];
                if (shardLoad == 0) {
                    continue;
                }
                if (currentLoad + shardLoad <= loadPerSplit) {
                    currentLoad += shardLoad;
                    newSplitsToLoads.get(splitNum).add(shardLoad);
                    shardLoads[shardNum] = 0;
                    newSplitsToShards.get(splitNum).add(shardNum);
                    if (currentLoad == loadPerSplit) {
                        break;
                    }
                } else {
                    assert(currentLoad < loadPerSplit);
                    newSplitsToLoads.get(splitNum).add(loadPerSplit - currentLoad);
                    currentLoad = loadPerSplit;
                    shardLoads[shardNum] = currentLoad + shardLoad - loadPerSplit;
                    assert(shardLoads[shardNum] > 0);
                    newSplitsToShards.get(splitNum).add(shardNum);
                    break;
                }
            }
            assert(Math.abs(currentLoad - loadPerSplit) < 5); // For rounding errors.
        }


        for (int splitNum = 0; splitNum < splitFactor; splitNum++) {
            List<Integer> splitShards = newSplitsToShards.get(splitNum);
            List<Integer> splitShardLoads = newSplitsToLoads.get(splitNum);
            // logger.info("{} {}", splitNum, splitShards);
            int numSplitShards = splitShards.size();
            int[] localShardLoads = new int[numSplitShards];
            int[] localShardMemoryUsages = new int[numSplitShards];
            for (int i = 0; i < numSplitShards; i++) {
                int shardNum = splitShards.get(i);
                localShardLoads[i] = splitShardLoads.get(i);
                localShardMemoryUsages[i] = shardMemoryUsages[shardNum];
            }
            int[][] localCurrentLocations = new int[serversPerSplit][numSplitShards];
            for (int serverNum = 0; serverNum < serversPerSplit; serverNum++) {
                int globalServerNum = splitNum * serversPerSplit + serverNum;
                for (int i = 0; i < numSplitShards; i++) {
                    int shardNum = splitShards.get(i);
                    localCurrentLocations[serverNum][i] = currentLocations[globalServerNum][shardNum];
                }
            }
            assert(sampleQueries.size() == 0);

            List<double[]> rs = balanceLoad(numSplitShards, serversPerSplit, localShardLoads, localShardMemoryUsages, localCurrentLocations,
                    sampleQueries, maxMemory);

            for (int serverNum = 0; serverNum < serversPerSplit; serverNum++) {
                int globalServerNum = splitNum * serversPerSplit + serverNum;
                double[] r = rs.get(serverNum);
                double[] finalR = finalRs.get(globalServerNum);
                for (int i = 0; i < numSplitShards; i++) {
                    int shardNum = splitShards.get(i);
                    finalR[shardNum] = r[i];
                }
            }
        }
        return finalRs;
    }

    /**
     * Generate an assignment of shards to servers.
     * @param numShards  Number of shards.
     * @param numServers  Number of servers.
     * @param shardLoads Amount of load on each shard.
     * @param shardMemoryUsages Memory usage of each shard.
     * @param currentLocations Entry [x][y] is 1 if server x has a copy of shard y and 0 otherwise.
     * @param sampleQueries A map from sets of shards to the number of queries that touched precisely that set.
     * @param maxMemory  Maximum server memory.
     * @return Entry [x][y] is the percentage of queries for shard y that should be routed to server x.
     * @throws IloException
     */

    public List<double[]> balanceLoad(Integer numShards, Integer numServers,
                                      int[] shardLoads, int[] shardMemoryUsages, int[][] currentLocations,
                                      Map<Set<Integer>, Integer> sampleQueries,
                                      Integer maxMemory) throws IloException {
        assert(shardLoads.length == numShards);
        assert(shardMemoryUsages.length == numShards);
        assert(currentLocations.length == numServers);
        for (int[] currentLocation : currentLocations) {
            assert (currentLocation.length == numShards);
        }

        int[][] transferCosts = currentLocations.clone();
        for(int i = 0; i < transferCosts.length; i++) {
            transferCosts[i] = Arrays.stream(transferCosts[i]).map(j -> j == 0 ? 1 : 0).toArray();
        }

        // Begin parallel objective.
        IloCplex cplex = new IloCplex();
        if (!verbose) {
            cplex.setOut(null);
        }

        List<IloNumVar[]> r = new ArrayList<>();
        List<IloNumVar[]> x = new ArrayList<>();
        for (int i = 0; i < numServers; i++) {
            r.add(cplex.numVarArray(numShards, 0, 1));
            x.add(cplex.intVarArray(numShards, 0, 1));
        }

        final int maxQuerySamples = 500;
        List<Set<Integer>> sampleQueryKeys = new ArrayList<>(sampleQueries.keySet());
        sampleQueryKeys = sampleQueryKeys.stream().filter(k -> k.size() > 1).collect(Collectors.toList());
        sampleQueryKeys = sampleQueryKeys.stream().sorted(Comparator.comparing(sampleQueries::get).reversed()).limit(maxQuerySamples).collect(Collectors.toList());

        // Minimize sum of query worst-case times, weighted by query frequency.
        int numSampleQueries = sampleQueryKeys.size();
        IloIntVar[] m = cplex.intVarArray(numSampleQueries, 0, 20); // Each entry is the maximum number of that query's shards on the same server.

        for(int serverNum = 0; serverNum < numServers; serverNum++) {
            int q = 0;
            for (Set<Integer> shards : sampleQueryKeys) {
                IloNumExpr e = cplex.constant(0);
                for (Integer shardNum : shards) {
                    e = cplex.sum(e, x.get(serverNum)[shardNum]);
                }
                cplex.addLe(e, m[q]);
                q++;
            }
        }

        int[] queryWeights = sampleQueryKeys.stream().map(sampleQueries::get).mapToInt(i ->i).toArray();
        IloObjective parallelObjective = cplex.minimize(cplex.scalProd(m, queryWeights));
        cplex.add(parallelObjective);

        setCoreConstraints(cplex, r, x, numShards, numServers, shardLoads, shardMemoryUsages, maxMemory);

        // Solve parallel objective.
        if (numSampleQueries > 0) {
            cplex.solve();
            lastM = cplex.getValues(m);
        } else {
            lastM = new double[m.length];
            Arrays.fill(lastM, 20.0);
        }

        // Begin transfer objective.
        cplex = new IloCplex();
         cplex.setParam(IloCplex.Param.MIP.Tolerances.MIPGap, 0.05);
        if (!verbose) {
            cplex.setOut(null);
        }
        r = new ArrayList<>();
        x = new ArrayList<>();
        for (int i = 0; i < numServers; i++) {
            r.add(cplex.numVarArray(numShards, 0, 1));
            x.add(cplex.intVarArray(numShards, 0, 1));
        }

        // Minimize transfer costs.
        IloNumExpr[] transferCostList = new IloNumExpr[numServers];
        for (int i = 0; i < numServers; i++) {
            transferCostList[i] = cplex.scalProd(x.get(i), transferCosts[i]);
        }
        IloObjective transferObjective = cplex.minimize(cplex.sum(transferCostList));
        cplex.add(transferObjective);

        for(int serverNum = 0; serverNum < numServers; serverNum++) {
            int q = 0;
            for (Set<Integer> shards : sampleQueryKeys) {
                IloNumExpr e = cplex.constant(0);
                for (Integer shardNum : shards) {
                    e = cplex.sum(e, x.get(serverNum)[shardNum]);
                }
                cplex.addLe(e, lastM[q]);
                q++;
            }
        }

        setCoreConstraints(cplex, r, x, numShards, numServers, shardLoads, shardMemoryUsages, maxMemory);

        // Solve transfer objective.
        cplex.solve();

        lastNumShards = numShards;
        lastNumServers = numServers;
        lastR = new ArrayList<>();
        lastX = new ArrayList<>();
        for (int i = 0; i < numServers; i++) {
            lastR.add(cplex.getValues(r.get(i)));
            lastX.add(cplex.getValues(x.get(i)));
        }
        return lastR;
    }


    // Set the load, memory, and sanity constraints.
    private static void setCoreConstraints(IloCplex cplex, List<IloNumVar[]> r, List<IloNumVar[]> x, Integer numShards, Integer numServers,
                                           int[] shardLoads, int[] shardMemoryUsages,
                                           Integer maxMemory) throws IloException {
        int actualReplicationFactor = minReplicationFactor < numServers ? minReplicationFactor : numServers;
        double averageLoad = (double) Arrays.stream(shardLoads).sum() / numServers;
        double epsilon = averageLoad / 20;

        for (int i = 0; i < numServers; i++) {
            cplex.addLe(cplex.scalProd(shardLoads, r.get(i)), averageLoad + epsilon); // Max load constraint
            cplex.addGe(cplex.scalProd(shardLoads, r.get(i)), averageLoad - epsilon); // Min load constraint
        }

        for (int i = 0; i < numServers; i++) {
            cplex.addLe(cplex.scalProd(shardMemoryUsages, x.get(i)), maxMemory); // Memory constraint
        }
        for (int i = 0; i < numServers; i++) {
            for (int j = 0; j < numShards; j++) {
                cplex.addLe(r.get(i)[j], x.get(i)[j]); // Ensure x_ij is 1 if r_ij is positive.
                if (actualReplicationFactor > 1) {
                    cplex.addLe(x.get(i)[j], cplex.sum(r.get(i)[j], 0.9999));
                }
            }
        }

        for (int j = 0; j < numShards; j++) {
            IloNumVar[] rShardServers = new IloNumVar[numServers];
            for (int i = 0; i < numServers; i++) {
                rShardServers[i] = r.get(i)[j];
            }
            cplex.addEq(cplex.sum(rShardServers), 1); // Require sum of r for each shard to be 1.
        }

        for (int j = 0; j < numShards; j++) {
            IloNumVar[] xShardServers = new IloNumVar[numServers];
            for (int i = 0; i < numServers; i++) {
                xShardServers[i] = x.get(i)[j];
            }
            cplex.addGe(cplex.sum(xShardServers), actualReplicationFactor); // Require each shard to be replicated N times.
        }
    }

    /**
     * Balance cluster load.
     * @param shardLoads Mapping from shard number to load.
     * @param shardMap Map from shard to location.
     * @return Map from shard to location.
     */
    public static Integer epsilonRatio = 20;
    public static Map<Integer, Integer> heuristicBalance(Map<Integer, Integer> shardLoads, Map<Integer, Integer> shardMap, List<Integer> serversList) {
        Set<Integer> lostShards = new HashSet<>();
        Set<Integer> gainedShards = new HashSet<>();
        Map<Integer, Integer> serverLoads = new HashMap<>();
        Map<Integer, List<Integer>> serverToShards = new HashMap<>();
        serverLoads.putAll(serversList.stream().collect(Collectors.toMap(i -> i, i -> 0)));
        serverToShards.putAll(serversList.stream().collect(Collectors.toMap(i -> i, i -> new ArrayList<>())));
        for (int shardNum: shardLoads.keySet()) {
            int shardLoad = shardLoads.get(shardNum);
            int serverNum = shardMap.get(shardNum);
            serverLoads.merge(serverNum, shardLoad, Integer::sum);
            serverToShards.get(serverNum).add(shardNum);
        }
        PriorityQueue<Integer> serverMinQueue = new PriorityQueue<>(Comparator.comparing(serverLoads::get));
        PriorityQueue<Integer> serverMaxQueue = new PriorityQueue<>(Comparator.comparing(serverLoads::get).reversed());
        serverMinQueue.addAll(serverLoads.keySet());
        serverMaxQueue.addAll(serverLoads.keySet());
        double averageLoad = shardLoads.values().stream().mapToDouble(i -> i).sum() / serverLoads.size();
        double epsilon = averageLoad / epsilonRatio;
        Map<Integer, Integer> returnMap = new HashMap<>(shardMap);
        while (serverMaxQueue.size() > 0 && serverLoads.get(serverMaxQueue.peek()) > averageLoad + epsilon) {
            Integer overLoadedServer = serverMaxQueue.remove();
            while (serverToShards.get(overLoadedServer).size() > 0 && serverLoads.get(overLoadedServer) > averageLoad + epsilon) {
                Integer underLoadedServer = serverMinQueue.remove();
                Integer mostLoadedShard = serverToShards.get(overLoadedServer).stream().filter(i -> shardLoads.get(i) > 0).max(Comparator.comparing(shardLoads::get)).orElse(null);
                assert(mostLoadedShard != null);
                serverToShards.get(overLoadedServer).remove(mostLoadedShard);
                if (serverLoads.get(underLoadedServer) + shardLoads.get(mostLoadedShard) <= averageLoad + epsilon) {
                    returnMap.put(mostLoadedShard, underLoadedServer);
                    serverLoads.merge(overLoadedServer, -1 * shardLoads.get(mostLoadedShard), Integer::sum);
                    serverLoads.merge(underLoadedServer, shardLoads.get(mostLoadedShard), Integer::sum);
                    lostShards.add(overLoadedServer);
                    gainedShards.add(underLoadedServer);
                    if (verbose) {
                        logger.info("Shard {} transferred from DS{} to DS{}", mostLoadedShard, overLoadedServer, underLoadedServer);
                    }
                }
                serverMinQueue.add(underLoadedServer);
            }
        }
        return returnMap;
    }
}
