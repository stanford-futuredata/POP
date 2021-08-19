package edu.stanford.futuredata;

import ilog.concert.IloException;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LoadBalancerExecutable {

    private static final Logger logger = LoggerFactory.getLogger(LoadBalancerExecutable.class);

    static int numShards;
    static int numServers;
    static int splitFactor = 4;
    static int maxMemory = 16;
    static int numRounds = 100;
    static int skipRounds = 20;
    static long randomSeed = 42;

    public static void main(String[] args) throws Exception {
        Options options = new Options();
        options.addOption("numShards", true, "Number of Shards?");
        options.addOption("numServers", true, "Number of Servers?");
        options.addOption("numSplits", true, "Split Factor?");
        options.addOption("benchmark", true, "Which Benchmark?");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        numShards = Integer.parseInt(cmd.getOptionValue("numShards"));
        numServers = Integer.parseInt(cmd.getOptionValue("numServers"));
        splitFactor = cmd.hasOption("numSplits") ? Integer.parseInt(cmd.getOptionValue("numSplits")) : 1;
        String benchmark = cmd.getOptionValue("benchmark");

        if (benchmark.equals("base")) {
            zipfianBenchmark();
        } else if (benchmark.equals("split")) {
            zipfianBenchmarkSplit();
        } else if (benchmark.equals("heuristic")) {
            zipfianHeuristicBenchmark();
        }

    }

    public static void zipfianBenchmark() throws IloException {
        logger.info("zipfianBenchmark");

        LoadBalancer.verbose = true;
        int[][] currentLocations = new int[numServers][numShards];
        for (int shardNum = 0; shardNum < numShards; shardNum++) {
            int serverNum = shardNum % numServers;
            currentLocations[serverNum][shardNum] = 1;
        }
        long totalTime = 0;
        int totalMovements = 0;
        Random r = new Random();
        r.setSeed(randomSeed);
        for (int roundNum = 0; roundNum < numRounds; roundNum++) {
            double zipfValue = 0.25 + r.nextDouble() * 0.5;
            int[] shardLoads = new int[numShards];
            int[] memoryUsages = new int[numShards];
            for (int shardNum = 0; shardNum < numShards; shardNum++) {
                int load = (int) Math.round(1000000.0 * (1.0 / Math.pow(shardNum + 1, zipfValue)));
                shardLoads[shardNum] = load;
                memoryUsages[shardNum] = 1;
            }

            long startTime = System.currentTimeMillis();
            List<double[]> returnR = new LoadBalancer().balanceLoad(numShards, numServers, shardLoads, memoryUsages, currentLocations, new HashMap<>(), maxMemory);
            long lbTime = System.currentTimeMillis() - startTime;
            assertEquals(numServers, returnR.size());
            double averageLoad = IntStream.of(shardLoads).sum() / (double) numServers;
            int[][] lastLocations = new int[numServers][];
            for (int i = 0; i < numServers; i++) {
                lastLocations[i] = currentLocations[i].clone();
            }
            for (int serverNum = 0; serverNum < numServers; serverNum ++) {
                double[] Rs = returnR.get(serverNum);
                double serverLoad = 0;
                for (int i = 0; i < numShards; i++) {
                    serverLoad += Rs[i] * shardLoads[i];
                    currentLocations[serverNum][i] = Rs[i] > 0 ? 1 : 0;
                }
                assertTrue(serverLoad <= averageLoad * 1.1);
                assertTrue(serverLoad >= averageLoad * 0.9);
            }
            int shardsMoved = 0;
            for(int i = 0; i < numServers; i++) {
                for(int j = 0; j < numShards; j++) {
                    if (currentLocations[i][j] == 1 && lastLocations[i][j] == 0) {
                        shardsMoved++;
                    }
                }
            }
            if (roundNum >= skipRounds) {
                totalMovements += shardsMoved;
                totalTime += lbTime;
            }
            System.out.printf("Round: %d Zipf: %.3f Shards Moved: %d LB time: %dms\n", roundNum, zipfValue, shardsMoved, lbTime);
        }
        System.out.printf("Average movements: %.2f, Average time: %dms\n", (double) totalMovements / (numRounds - skipRounds), totalTime / (numRounds - skipRounds));
    }

    public static void zipfianBenchmarkSplit() throws IloException {
        logger.info("zipfianBenchmarkSplit");
        LoadBalancer.verbose = false;
        int[][] currentLocations = new int[numServers][numShards];
        long totalTime = 0;
        int totalMovements = 0;
        List<Integer> order = IntStream.range(0, numShards).boxed().collect(Collectors.toList());
        Collections.shuffle(order);
        Random r = new Random();
        r.setSeed(randomSeed);
        for (int roundNum = 0; roundNum < numRounds; roundNum++) {
            double zipfValue = 0.25 + r.nextDouble() * 0.5;
            int[] shardLoads = new int[numShards];
            int[] memoryUsages = new int[numShards];
            for (int shardNum = 0; shardNum < numShards; shardNum++) {
                int load = (int) Math.round(1000000.0 * (1.0 / Math.pow(shardNum + 1, zipfValue)));
                shardLoads[order.get(shardNum)] = load;
                memoryUsages[shardNum] = 1;
            }

            long startTime = System.currentTimeMillis();
            List<double[]> returnR = new LoadBalancer().balanceLoad(numShards, numServers, shardLoads, memoryUsages, currentLocations, new HashMap<>(), maxMemory, splitFactor);
            long lbTime = System.currentTimeMillis() - startTime;
            assertEquals(numServers, returnR.size());
            int[][] lastLocations = new int[numServers][];
            for (int i = 0; i < numServers; i++) {
                lastLocations[i] = currentLocations[i].clone();
            }
            for (int serverNum = 0; serverNum < numServers; serverNum ++) {
                double[] Rs = returnR.get(serverNum);
                for (int i = 0; i < numShards; i++) {
                    currentLocations[serverNum][i] = Rs[i] > 0 ? 1 : 0;
                }
            }
            int shardsMoved = 0;
            for(int i = 0; i < numServers; i++) {
                for(int j = 0; j < numShards; j++) {
                    if (currentLocations[i][j] == 1 && lastLocations[i][j] == 0) {
                        shardsMoved++;
                    }
                }
            }
            if (roundNum >= skipRounds) {
                totalMovements += shardsMoved;
                totalTime += lbTime;
            }
            System.out.printf("Round: %d Zipf: %.3f Shards Moved: %d LB time: %dms\n", roundNum, zipfValue, shardsMoved, lbTime);
        }
        System.out.printf("Split Average movements: %.2f, Average time: %dms\n", (double) totalMovements / (numRounds - skipRounds), totalTime / (numRounds - skipRounds));
    }

    public static void zipfianHeuristicBenchmark() {
        logger.info("zipfianHeuristicBenchmark");

        LoadBalancer.verbose = false;
        Map<Integer, Integer> currentLocations = new HashMap<>();
        for (int i = 0; i < numShards; i++) {
            currentLocations.put(i, i % numServers);
        }
        List<Integer> serversList = new ArrayList<>();
        for (int i = 0 ; i < numServers; i++) {
            serversList.add(i);
        }
        long totalTime = 0;
        int totalMovements = 0;
        Random r = new Random();
        r.setSeed(randomSeed);
        for (int roundNum = 0; roundNum < numRounds; roundNum++) {
            double zipfValue = 0.25 + r.nextDouble() * 0.5;
            Map<Integer, Integer> shardLoads = new HashMap<>();
            int totalLoad = 0;
            for (int shardNum = 0; shardNum < numShards; shardNum++) {
                int load = (int) Math.round(1000000.0 * (1.0 / Math.pow(shardNum + 1, zipfValue)));
                shardLoads.put(shardNum, load);
                totalLoad += load;
            }

            Map<Integer, Integer> lastLocations = new HashMap<>(currentLocations);
            long startTime = System.currentTimeMillis();
            currentLocations = LoadBalancer.heuristicBalance(shardLoads, currentLocations, serversList);
            long lbTime = System.currentTimeMillis() - startTime;
            assertEquals(numShards, currentLocations.size());
            double averageLoad = totalLoad / (double) numServers;

            // Check correctness.
            if (LoadBalancer.verbose) {
                for (int serverNum = 0; serverNum < numServers; serverNum++) {
                    double serverLoad = 0;
                    for (int shardNum = 0; shardNum < numShards; shardNum++) {
                        if (currentLocations.get(shardNum) == serverNum) {
                            serverLoad += shardLoads.get(shardNum);
                        }
                    }
                    logger.info("{} {} {}", serverNum, averageLoad, serverLoad);
                }
            }

            int shardsMoved = 0;
            for(int shardNum = 0; shardNum < numShards; shardNum++) {
                if (!currentLocations.get(shardNum).equals(lastLocations.get(shardNum))) {
                    shardsMoved++;
                }
            }
            if (roundNum >= skipRounds) {
                totalMovements += shardsMoved;
                totalTime += lbTime;
            }
            System.out.printf("Round: %d Zipf: %.3f Shards Moved: %d LB time: %dms\n", roundNum, zipfValue, shardsMoved, lbTime);
        }
        System.out.printf("Average movements: %.2f, Average time: %dms\n", (double) totalMovements / (numRounds - skipRounds), totalTime / (numRounds - skipRounds));
    }
}
